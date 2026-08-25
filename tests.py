"""
Unit Test Suite for Chess Engine Modules.
Ensures correct implementation of board encoding, MCTS selections,
self-play generation, and matchmaking.
Skips PyTorch neural network tests if PyTorch is not installed.
"""

import os
import shutil
import unittest
import numpy as np
import chess

from board_encoder import encode_board, encode_move, decode_move, get_legal_move_mask
from mcts import MCTS, StockfishEvaluator

# Check if PyTorch is available
try:
    import torch
    from neural_net import ChessNet, compute_loss
    from mcts import NeuralNetEvaluator
    from self_play import play_game, ReplayBuffer
    from evaluator import evaluate_candidate
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class TestBoardEncoder(unittest.TestCase):
    def test_board_encoding_shape(self):
        board = chess.Board()
        tensor = encode_board(board)
        # Ensure correct shape (18, 8, 8) — no side-to-move plane (constant, useless)
        self.assertEqual(tensor.shape, (18, 8, 8))
        # Halfmove clock is 0 at the start: plane 17 must be all zeros
        self.assertTrue(np.all(tensor[17] == 0.0))

    def test_perspective_mirroring(self):
        # Position with Black to move
        board = chess.Board()
        board.push_san("e4") # Now Black to move
        tensor = encode_board(board)
        
        # Encoding flips perspective. e4 is a pawn move, so the halfmove
        # clock reset to 0 -> plane 17 must be all zeros.
        self.assertEqual(tensor.shape, (18, 8, 8))
        self.assertTrue(np.all(tensor[17] == 0.0))
        
        # Active player's pieces occupy planes 0-5 from the mover's perspective:
        # Black's pawn layout (mirrored) must equal White's from the start.
        white_start = encode_board(chess.Board())
        self.assertEqual(np.array_equal(white_start[0], tensor[0]), True)

    def test_move_encoding_decoding_roundtrip(self):
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        
        action = encode_move(move)
        # Verify action index is in [0, 4095]
        self.assertTrue(0 <= action < 4096)
        
        decoded = decode_move(action, board)
        self.assertEqual(move, decoded)

    def test_legal_move_mask(self):
        board = chess.Board()
        mask = get_legal_move_mask(board)
        self.assertEqual(mask.shape, (4096,))
        # Start position has 20 legal moves
        self.assertEqual(np.sum(mask), 20)


@unittest.skipIf(not HAS_TORCH, "PyTorch is not installed")
class TestNeuralNet(unittest.TestCase):
    def test_network_shapes(self):
        # Create a lightweight model for fast local CPU testing
        model = ChessNet(num_blocks=2, filters=16, in_channels=18)
        dummy_input = torch.randn(2, 18, 8, 8)
        
        policy_logits, value, moves_left = model(dummy_input)
        
        self.assertEqual(policy_logits.shape, (2, 4096))
        self.assertEqual(value.shape, (2, 1))
        self.assertEqual(moves_left.shape, (2, 1))

    def test_loss_function(self):
        policy_logits = torch.randn(4, 4096)
        value = torch.tanh(torch.randn(4, 1))
        moves_left = torch.relu(torch.randn(4, 1))
        
        target_policy = torch.softmax(torch.randn(4, 4096), dim=-1)
        target_value = torch.tensor([1.0, -1.0, 0.0, 0.5])
        target_moves_left = torch.tensor([10.0, 15.0, 20.0, 25.0])
        
        loss, p_loss, v_loss, m_loss = compute_loss(
            policy_logits, value, moves_left, 
            target_policy, target_value, target_moves_left
        )
        
        self.assertTrue(loss.item() > 0)
        self.assertTrue(p_loss.item() > 0)
        self.assertTrue(v_loss.item() > 0)


class TestMCTS(unittest.TestCase):
    @unittest.skipIf(not HAS_TORCH, "PyTorch is not installed")
    def test_mcts_search_step(self):
        # Setup lightweight model & evaluator
        model = ChessNet(num_blocks=2, filters=16, in_channels=18)
        evaluator = NeuralNetEvaluator(model)
        
        # Initialize MCTS with 5 simulations (very fast)
        mcts = MCTS(evaluator=evaluator, c_puct=1.5, num_simulations=5)
        board = chess.Board()
        
        # Verify search completes and returns a root node with stats
        root = mcts.search(board)
        self.assertIsNotNone(root)
        self.assertEqual(sum(child.visit_count for child in root.children.values()), 5)
        
        # Verify move probability vector shape
        probs = mcts.get_move_probs(board, temp=0.0)
        self.assertEqual(probs.shape, (4096,))
        self.assertEqual(np.sum(probs), 1.0)
        self.assertEqual(np.sum(probs > 0), 1) # temp=0.0 should yield one-hot max visited

    def test_mcts_deadline_stops_search(self):
        # Regression test for time-control enforcement: a deadline already in
        # the past must prevent any simulation from running.
        import time as _time
        evaluator = StockfishEvaluator(stockfish_path="invalid_path_to_stockfish_executable")
        mcts = MCTS(evaluator=evaluator, c_puct=1.5, num_simulations=500)
        board = chess.Board()
        
        root = mcts.search(board, deadline=_time.time() - 1.0)
        total = sum(child.visit_count for child in root.children.values())
        self.assertEqual(total, 0)
        
        # A future deadline allows the full simulation count.
        mcts2 = MCTS(evaluator=evaluator, c_puct=1.5, num_simulations=5)
        start = _time.time()
        root2 = mcts2.search(board, deadline=start + 60.0)
        total2 = sum(child.visit_count for child in root2.children.values())
        self.assertEqual(total2, 5)

    def test_fallback_heuristic_evaluator(self):
        # Initialize Stockfish evaluator with a fake path to trigger fallback
        evaluator = StockfishEvaluator(stockfish_path="invalid_path_to_stockfish_executable")
        board = chess.Board()
        
        # Evaluator should fallback gracefully without crashing
        policy, value = evaluator.evaluate(board)
        self.assertEqual(policy.shape, (4096,))
        self.assertTrue(-1.0 <= value <= 1.0)
        self.assertAlmostEqual(float(np.sum(policy)), 1.0, places=5)

    def test_stockfish_evaluator_integration(self):
        # Look for local stockfish binary (cross-platform)
        sf_path = shutil.which("stockfish") or ("stockfish.exe" if os.path.exists("stockfish.exe") else None)
        if sf_path is None:
            self.skipTest("Stockfish not found in PATH or working directory")
        
        # Initialize with depth 2 to keep search quick
        evaluator = StockfishEvaluator(stockfish_path=sf_path, depth=2)
        board = chess.Board()
        
        # Check evaluation returns predictions
        policy, value = evaluator.evaluate(board)
        evaluator.close()
        
        self.assertEqual(policy.shape, (4096,))
        self.assertTrue(-1.0 <= value <= 1.0)
        self.assertAlmostEqual(float(np.sum(policy)), 1.0, places=5)
        # Ensure it actually runs search and evaluates moves
        self.assertTrue(np.any(policy > 0.0))

    def test_stockfish_guided_mcts(self):
        sf_path = shutil.which("stockfish") or ("stockfish.exe" if os.path.exists("stockfish.exe") else None)
        if sf_path is None:
            self.skipTest("Stockfish not found in PATH or working directory")
            
        evaluator = StockfishEvaluator(stockfish_path=sf_path, depth=2)
        mcts = MCTS(evaluator=evaluator, c_puct=1.5, num_simulations=5)
        board = chess.Board()
        
        # Run MCTS search using Stockfish Evaluator
        root = mcts.search(board)
        evaluator.close()
        
        self.assertIsNotNone(root)
        self.assertEqual(sum(child.visit_count for child in root.children.values()), 5)


@unittest.skipIf(not HAS_TORCH, "PyTorch is not installed")
class TestSelfPlay(unittest.TestCase):
    def test_replay_buffer(self):
        buffer = ReplayBuffer(capacity=10)
        state = np.random.randn(18, 8, 8)
        policy = np.random.randn(4096)
        policy = np.abs(policy)
        policy /= np.sum(policy)
        
        buffer.add(state, policy, 1.0, 15.0)
        self.assertEqual(len(buffer), 1)
        
        states, policies, values, moves = buffer.sample_batch(batch_size=4)
        self.assertEqual(states.shape[0], 1)
        self.assertEqual(states.shape[1:], (18, 8, 8))

    def test_mini_self_play_game(self):
        model = ChessNet(num_blocks=1, filters=8, in_channels=18)
        evaluator = NeuralNetEvaluator(model)
        mcts = MCTS(evaluator=evaluator, num_simulations=2)
        
        # Generate game data using a low temp threshold to finish fast
        game_data = play_game(mcts, temp_threshold=2)
        self.assertTrue(len(game_data) > 0)
        
        # Check components of generated data
        state, policy, value, moves_left = game_data[0]
        self.assertEqual(state.shape, (18, 8, 8))
        self.assertEqual(policy.shape, (4096,))
        self.assertTrue(-1.0 <= value <= 1.0)


@unittest.skipIf(not HAS_TORCH, "PyTorch is not installed")
class TestEvaluator(unittest.TestCase):
    def test_matchmaking(self):
        model_c = ChessNet(num_blocks=1, filters=8, in_channels=18)
        model_b = ChessNet(num_blocks=1, filters=8, in_channels=18)
        eval_c = NeuralNetEvaluator(model_c)
        eval_b = NeuralNetEvaluator(model_b)
        
        # Test a 2-game evaluation match with 3 simulations per move (ultra fast)
        win_rate, elo, promoted = evaluate_candidate(eval_c, eval_b, num_games=2, num_sims=3)
        self.assertTrue(0.0 <= win_rate <= 1.0)
        self.assertTrue(isinstance(promoted, bool))


if __name__ == "__main__":
    unittest.main()
