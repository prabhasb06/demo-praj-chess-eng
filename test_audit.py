"""Regression tests for confirmed audit findings. No network access required."""
import io
import os
import pathlib
import queue
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch

import chess
import numpy as np
from alpha_beta import MATE_SCORE, alpha_beta, quiescence
from board_encoder import (POLICY_SIZE, encode_move, decode_move, encode_move_for_perspective,
                           decode_move_for_perspective, get_legal_move_mask)
from mcts import MCTS, StockfishEvaluator, MCTSNode
from engine_info import NAME, VERSION, AUTHOR
from uci_engine import parse_position, parse_time_from_go, _num

REPO = pathlib.Path(__file__).parent


class StaticEvaluator:
    def __init__(self):
        self.calls = 0

    def evaluate(self, board):
        self.calls += 1
        mask = get_legal_move_mask(board).astype(np.float32)
        return mask / max(1, mask.sum()), 0.1


class AuditCore(unittest.TestCase):
    def test_underpromotions_roundtrip_both_colors(self):
        for fen in ("8/P7/8/8/8/8/7p/4K2k w - - 0 1",
                    "4k2K/7P/8/8/8/8/p7/8 b - - 0 1"):
            board = chess.Board(fen)
            moves = list(board.legal_moves)
            actions = [encode_move(m) for m in moves]
            self.assertEqual(len(set(actions)), len(moves))
            for move in moves:
                self.assertEqual(decode_move(encode_move(move), board), move)
                action = encode_move_for_perspective(move, board.turn == chess.BLACK)
                self.assertEqual(decode_move_for_perspective(action, board), move)
            self.assertEqual(get_legal_move_mask(board).sum(), len(moves))

    def test_random_move_encoding(self):
        rng = np.random.default_rng(10)
        board = chess.Board()
        for _ in range(300):
            legal = list(board.legal_moves)
            if not legal or board.is_game_over():
                board = chess.Board()
                legal = list(board.legal_moves)
            for move in legal:
                self.assertEqual(decode_move(encode_move(move), board), move)
            board.push(legal[int(rng.integers(len(legal)))])

    def test_castling_en_passant(self):
        for fen in ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
                    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"):
            board = chess.Board(fen)
            root = MCTS(StaticEvaluator(), num_simulations=0).search(board)
            self.assertEqual({c.move for c in root.children.values()}, set(board.legal_moves))

    def test_search_expands_all_promotions(self):
        board = chess.Board("8/P7/8/8/8/8/7p/4K2k w - - 0 1")
        root = MCTS(StaticEvaluator(), num_simulations=0).search(board)
        self.assertEqual({c.move for c in root.children.values()}, set(board.legal_moves))

    def test_cache_noise_does_not_mutate(self):
        board = chess.Board()
        mcts = MCTS(StaticEvaluator(), num_simulations=0)
        expected, _ = mcts._get_evaluation(board)
        mcts.search(board, add_noise=True)
        actual, _ = mcts._get_evaluation(board)
        np.testing.assert_array_equal(expected, actual)
        actual[:] = 0
        again, _ = mcts._get_evaluation(board)
        self.assertGreater(again.sum(), 0)

    def test_cache_halfmove_and_history(self):
        ev = StaticEvaluator()
        search = MCTS(ev, num_simulations=0)
        board = chess.Board()
        search._get_evaluation(board)
        other = board.copy()
        other.halfmove_clock = 99
        search._get_evaluation(other)
        repeated = chess.Board()
        for move in "g1f3 g8f6 f3g1 f6g8".split():
            repeated.push_uci(move)
        fresh = chess.Board(repeated.fen())
        search._get_evaluation(repeated)
        search._get_evaluation(fresh)
        self.assertEqual(ev.calls, 4)

    def test_cache_evaluator_settings(self):
        ev = StaticEvaluator()
        ev.depth = 2
        search = MCTS(ev)
        search._get_evaluation(chess.Board())
        ev.depth = 9
        search._get_evaluation(chess.Board())
        self.assertEqual(ev.calls, 2)

    def test_tree_memory_bound(self):
        search = MCTS(StaticEvaluator(), num_simulations=100)
        with patch('mcts.MAX_TREE_NODES', 25):
            root = search.search(chess.Board())
        self.assertLessEqual(search.tree_nodes, 25)
        self.assertEqual(root.visit_count, 100)

    def test_cancelled_root_does_not_evaluate(self):
        ev = StaticEvaluator()
        search = MCTS(ev)
        search.cancel = lambda: True
        counts = search.get_visit_counts(chess.Board())
        self.assertEqual(ev.calls, 0)
        self.assertGreater(counts.sum(), 0)

    def test_zero_simulations_prior_fallback(self):
        class Preferred(StaticEvaluator):
            def evaluate(self, board):
                policy = np.zeros(POLICY_SIZE)
                policy[encode_move(chess.Move.from_uci("e2e4"))] = 1
                return policy, 0
        counts = MCTS(Preferred(), num_simulations=0).get_visit_counts(chess.Board())
        self.assertEqual(decode_move(int(counts.argmax()), chess.Board()).uci(), "e2e4")

    def test_tiny_temperature_is_finite(self):
        probs = MCTS(StaticEvaluator(), num_simulations=5).get_move_probs(chess.Board(), temp=1e-8)
        self.assertTrue(np.isfinite(probs).all())
        self.assertAlmostEqual(float(probs.sum()), 1)

    def test_terminal_policy_has_no_illegal_probability(self):
        board = chess.Board("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")
        probs = MCTS(StaticEvaluator(), num_simulations=1).get_move_probs(board)
        self.assertEqual(probs.sum(), 0)

    def test_mate_backpropagation_sign(self):
        board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
        root = MCTS(StaticEvaluator(), num_simulations=3).search(board)
        winning = max(root.children.values(), key=lambda c: c.visit_count)
        board.push(winning.move)
        self.assertTrue(board.is_checkmate())
        self.assertEqual(winning.value, 1)

    def test_stalemate_at_quiescence_horizon(self):
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        self.assertTrue(board.is_stalemate())
        self.assertEqual(quiescence(board, -MATE_SCORE, MATE_SCORE), 0)
        self.assertEqual(alpha_beta(board, 0, -MATE_SCORE, MATE_SCORE)[0], 0)

    def test_clock_allocation_never_exceeds_clock(self):
        for milliseconds in (0, 1, 5, 20, 99):
            budget, _ = parse_time_from_go(["go", "wtime", str(milliseconds)], chess.Board(), 30)
            self.assertLessEqual(budget, milliseconds / 1000)
        self.assertEqual(parse_time_from_go(["go", "movetime", "0"], chess.Board(), 30)[0], 0)

    def test_nonfinite_numbers(self):
        for token in ("nan", "inf", "-inf", "bad", None):
            self.assertIsNone(_num(token))

    def test_illegal_position_rejected(self):
        for line in ("position startpos moves e2e5", "position startpos moves 0000",
                     "position fen 8/8/8/8/8/8/8/8 w - - 0 1",
                     "position startpos moves e2e4 e7e5 e1e3"):
            with self.assertRaises(ValueError):
                parse_position(line.split())

    def test_position_history(self):
        board = parse_position("position startpos moves g1f3 g8f6 f3g1 f6g8".split())
        self.assertEqual(len(board.move_stack), 4)

    def test_latest_multipv_not_historical_moves(self):
        class Fake:
            proc = True
            last_complete = True
            def evaluate_position(self, *args):
                return ["info depth 1 multipv 1 score cp 900 pv g1h3",
                        "info depth 8 multipv 1 score cp 30 pv e2e4",
                        "info depth 8 multipv 2 score cp 20 pv d2d4", "bestmove e2e4"]
        ev = StockfishEvaluator.__new__(StockfishEvaluator)
        ev.engine, ev.depth, ev.multipv = Fake(), 8, 2
        ev.cancel, ev.deadline = lambda: False, None
        probs, value = ev.evaluate(chess.Board())
        self.assertGreater(probs[encode_move(chess.Move.from_uci("e2e4"))],
                           probs[encode_move(chess.Move.from_uci("g1h3"))] * 100)
        self.assertLess(value, 0.1)


class Session:
    def __init__(self):
        argv = [os.environ['PRAJ_ENGINE_EXE']] if os.environ.get('PRAJ_ENGINE_EXE') else [sys.executable, str(REPO / "uci_engine.py")]
        self.proc = subprocess.Popen(argv,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.lines = queue.Queue()
        self.reader = threading.Thread(target=self.read, daemon=True)
        self.reader.start()
        self.send("uci")
        _, self.startup_lines = self.until("uciok")
        if os.environ.get("PRAJ_STOCKFISH_PATH"):
            self.send("setoption name StockfishPath value " + os.environ["PRAJ_STOCKFISH_PATH"])
        self.send("isready")
        self.until("readyok")

    def read(self):
        for line in self.proc.stdout:
            self.lines.put(line.strip())

    def send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def until(self, prefix, timeout=5):
        end = time.monotonic() + timeout
        seen = []
        while time.monotonic() < end:
            try:
                line = self.lines.get(timeout=max(0.001, end-time.monotonic()))
            except queue.Empty:
                break
            seen.append(line)
            if line.startswith(prefix):
                return line, seen
        raise AssertionError(f"Missing {prefix}: {seen}")

    def close(self):
        self.send("quit")
        self.proc.wait(timeout=5)
        self.reader.join(timeout=1)
        for pipe in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            pipe.close()


class AuditProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = Session()

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    def setUp(self):
        self.session.send("position startpos")
        self.session.send("isready")
        self.session.until("readyok")

    def test_startup_identity_is_clean_and_correct(self):
        lines = self.session.startup_lines
        self.assertEqual(lines[0], f"{NAME} {VERSION} by {AUTHOR}")
        self.assertIn(f"id name {NAME} {VERSION}", lines)
        self.assertIn(f"id author {AUTHOR}", lines)
        self.assertFalse(any(line.startswith("info ") or "\x1b" in line for line in lines))

    def test_console_help_and_board_are_explicit_commands(self):
        self.session.send("help")
        self.session.send("isready")
        _, lines = self.session.until("readyok")
        self.assertTrue(any("go movetime 1000" in line for line in lines))
        self.session.send("position startpos moves e2e4")
        self.session.send("d")
        self.session.send("isready")
        _, lines = self.session.until("readyok")
        board = chess.Board()
        board.push_uci("e2e4")
        self.assertIn("Fen: " + board.fen(), lines)

    def best(self, go, timeout=5):
        self.session.send(go)
        return self.session.until("bestmove", timeout)[0].split()[1]

    def test_searchmoves(self):
        self.assertEqual(self.best("go movetime 30 searchmoves a2a3"), "a2a3")

    def test_underpromotion_searchmoves(self):
        self.session.send("position fen 8/P7/8/8/8/8/7p/4K2k w - - 0 1")
        self.assertEqual(self.best("go movetime 30 searchmoves a7a8n"), "a7a8n")

    def test_empty_searchmoves(self):
        self.assertEqual(self.best("go nodes 1 searchmoves"), "0000")

    def test_claimable_draw_still_legal(self):
        self.session.send("position startpos moves g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8")
        move = self.best("go movetime 30")
        self.assertIn(chess.Move.from_uci(move), chess.Board().legal_moves)

    def test_fifty_move_claim_does_not_force_null(self):
        self.session.send("position fen 6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 100 1")
        self.assertNotEqual(self.best("go movetime 30"), "0000")

    def test_instant_stop_repeated(self):
        for _ in range(15):
            self.session.send("go infinite")
            self.session.send("stop")
            line, _ = self.session.until("bestmove", 1)
            self.assertIn(chess.Move.from_uci(line.split()[1]), chess.Board().legal_moves)

    def test_stop_held_ponder(self):
        self.session.send("go ponder nodes 1")
        time.sleep(0.15)
        self.session.send("stop")
        self.session.until("bestmove", 1)

    def test_ponderhit_held_result(self):
        self.session.send("go ponder movetime 20")
        time.sleep(0.1)
        self.assertFalse(any(line.startswith("bestmove") for line in list(self.session.lines.queue)))
        self.session.send("ponderhit")
        self.session.until("bestmove", 1)

    def test_infinite_terminal_waits(self):
        self.session.send("position fen 7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")
        self.session.send("go infinite")
        time.sleep(0.08)
        self.assertFalse(any(line.startswith("bestmove") for line in list(self.session.lines.queue)))
        self.session.send("stop")
        self.assertEqual(self.session.until("bestmove", 1)[0], "bestmove 0000")

    def test_ready_while_searching(self):
        self.session.send("go infinite")
        self.session.send("isready")
        self.session.until("readyok", 0.5)
        self.session.send("stop")
        self.session.until("bestmove", 1)

    def test_no_stale_or_duplicate_moves(self):
        for i in range(8):
            self.session.send("position startpos")
            self.session.send("go infinite")
        self.session.send("position startpos moves e2e4")
        self.session.send("go movetime 30")
        line, lines = self.session.until("bestmove", 2)
        board = chess.Board()
        board.push_uci("e2e4")
        self.assertIn(chess.Move.from_uci(line.split()[1]), board.legal_moves)
        self.assertEqual(sum(l.startswith("bestmove") for l in lines), 1)
        self.session.send("isready")
        _, lines = self.session.until("readyok")
        self.assertFalse(any(l.startswith("bestmove") for l in lines))

    def test_movetime_latency(self):
        start = time.monotonic()
        self.best("go movetime 30")
        self.assertLess(time.monotonic()-start, 0.3)

    def test_depth_and_nodes(self):
        for command in ("go depth 1", "go nodes 1", "go movetime nan nodes 1"):
            start = time.monotonic()
            self.assertNotEqual(self.best(command), "0000")
            self.assertLess(time.monotonic()-start, 2)

    def test_malformed_position_transactional(self):
        self.session.send("position startpos moves e2e4")
        self.session.send("position startpos moves e2e5")
        self.assertEqual(self.best("go nodes 1 searchmoves e7e5"), "e7e5")



class AuditSubprocess(unittest.TestCase):
    def make_engine(self):
        from mcts import SubprocessStockfish
        return SubprocessStockfish([sys.executable, str(REPO / 'fixtures' / 'fake_stockfish.py')])

    def test_timeout_drains_before_next_query(self):
        engine = self.make_engine()
        self.addCleanup(engine.close)
        engine.deadline = time.monotonic() + 0.01
        first = engine.evaluate_position(chess.Board(), 2)
        self.assertEqual(first[-1], 'bestmove e2e4')
        engine.deadline = None
        second = engine.evaluate_position(chess.Board(), 2)
        self.assertEqual(second[-1], 'bestmove e2e4')
        self.assertEqual(sum(l.startswith('bestmove') for l in second), 1)
        self.assertTrue(any(l.startswith('info depth') for l in second))

    def test_unresponsive_stop_kills_and_reaps(self):
        engine = self.make_engine()
        self.addCleanup(engine.close)
        process = engine.proc
        engine.deadline = time.monotonic() + 0.01
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            engine.evaluate_position(chess.Board(), 77)
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertIsNotNone(process.poll())

    def test_crashed_backend_is_detected(self):
        engine = self.make_engine()
        self.addCleanup(engine.close)
        with self.assertRaises(OSError):
            engine.evaluate_position(chess.Board(), 78)


try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, 'PyTorch required')
class AuditTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_old_checkpoint_migration(self):
        import tempfile
        from neural_net import ChessNet, load_checkpoint
        model = ChessNet(num_blocks=1, filters=8)
        state = model.state_dict()
        original = state['fc_policy.weight'][:4096].clone()
        state['fc_policy.weight'] = original
        state['fc_policy.bias'] = state['fc_policy.bias'][:4096]
        with tempfile.TemporaryDirectory() as directory:
            filename = pathlib.Path(directory) / 'legacy.pt'
            torch.save(state, filename)
            load_checkpoint(model, filename)
        torch.testing.assert_close(model.fc_policy.weight[:4096], original)
        self.assertEqual(model.fc_policy.weight.shape[0], POLICY_SIZE)

    def test_mirrored_underpromotion_policy(self):
        from self_play import _mirror_policy
        board = chess.Board('8/P7/8/8/8/8/7p/4K2k w - - 0 1')
        policy = np.zeros(POLICY_SIZE, dtype=np.float32)
        policy[encode_move(chess.Move.from_uci('a7a8n'))] = 1
        mirrored = _mirror_policy(policy)
        self.assertEqual(mirrored[encode_move(chess.Move.from_uci('a2a1n'))], 1)
        np.testing.assert_array_equal(_mirror_policy(mirrored), policy)

    def test_empty_replay_rejected(self):
        from self_play import ReplayBuffer
        with self.assertRaises(ValueError):
            ReplayBuffer().sample_batch(2)

    def test_training_epoch_uses_full_buffer(self):
        import tempfile
        from self_play import run_self_play_training
        from board_encoder import encode_board
        mask = get_legal_move_mask(chess.Board()).astype(np.float32)
        examples = [(encode_board(chess.Board()), mask / mask.sum(), 0.0, 1.0)] * 3
        with tempfile.TemporaryDirectory() as directory:
            path = str(pathlib.Path(directory) / 'model.pt')
            with patch('self_play.DEFAULT_ARCH', dict(num_blocks=1, filters=8, in_channels=18)), \
                 patch('self_play.play_game', return_value=examples), \
                 patch('self_play.train_step', return_value=1.0) as train:
                run_self_play_training(path, num_games=1, num_epochs=1, batch_size=2, num_sims=1)
                self.assertEqual(train.call_count, 2)
                self.assertTrue(pathlib.Path(path).exists())

    def test_gradient_step_is_finite_and_updates_weights(self):
        from neural_net import ChessNet
        from self_play import train_step
        model = ChessNet(num_blocks=1, filters=8)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        before = model.fc_value2.weight.detach().clone()
        loss = train_step(model, optimizer, torch.randn(2, 18, 8, 8),
                          torch.full((2, POLICY_SIZE), 1 / POLICY_SIZE),
                          torch.tensor([1.0, -1.0]), torch.ones(2))
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before, model.fc_value2.weight))


if __name__ == "__main__":
    unittest.main()
