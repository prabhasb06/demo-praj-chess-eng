"""
MODULE 3: mcts.py
Implements Monte Carlo Tree Search (MCTS) enhanced with neural policy/value heads,
virtual loss for parallel search, Dirichlet noise for exploration, and a Stockfish-guided evaluator.
"""

import math
import random
import threading
from typing import Dict, Tuple, List, Optional
import numpy as np
import chess
import chess.engine

from board_encoder import encode_board, encode_move, decode_move, get_legal_move_mask, mirror_move

class MCTSNode:
    """
    Represents a single node in the MCTS tree.
    Tracks visit counts, total values, mean values, prior probabilities, and virtual loss.
    """
    def __init__(self, prior: float, parent: Optional['MCTSNode'] = None, move: Optional[chess.Move] = None):
        self.parent = parent
        self.move = move
        self.prior = prior  # P(s, a)
        self.visit_count = 0  # N(s, a)
        self.total_value = 0.0  # W(s, a)
        self.mean_value = 0.0  # Q(s, a)
        self.virtual_loss = 0  # V(s, a)
        self.children: Dict[int, MCTSNode] = {}  # action_index -> MCTSNode
        self.lock = threading.Lock()

    @property
    def value(self) -> float:
        """Returns Q(s,a) adjusted for virtual loss during parallel search."""
        # Treat virtual loss visits as losses (score = -1) to steer threads elsewhere
        n = self.visit_count + self.virtual_loss
        if n == 0:
            return 0.0
        w = self.total_value - self.virtual_loss
        return w / n

    def select_child(self, c_puct: float) -> Tuple[int, 'MCTSNode']:
        """
        Selects the child node that maximizes the PUCT formula:
        U(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
        """
        best_score = -float('inf')
        best_action = -1
        best_child = None

        total_visits = sum(child.visit_count for child in self.children.values())
        sqrt_total = math.sqrt(total_visits) if total_visits > 0 else 1.0

        for action, child in self.children.items():
            # Adjust prior probability based on PUCT selection formula
            u = child.value + c_puct * child.prior * sqrt_total / (1.0 + child.visit_count + child.virtual_loss)
            if u > best_score:
                best_score = u
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, action_priors: np.ndarray, legal_mask: np.ndarray):
        """
        Expands the node by creating children for all legal actions.
        action_priors: Array of size 4096 containing the moves' probabilities.
        """
        for action in range(4096):
            if legal_mask[action]:
                prior = action_priors[action]
                # Reconstruct move from action
                from_sq = action // 64
                to_sq = action % 64
                move = chess.Move(from_sq, to_sq)
                self.children[action] = MCTSNode(prior=prior, parent=self, move=move)

    def add_virtual_loss(self):
        """Adds virtual loss to discourage multiple threads from exploring identical paths."""
        with self.lock:
            self.virtual_loss += 1

    def remove_virtual_loss(self):
        """Removes virtual loss once simulation/evaluation completes."""
        with self.lock:
            self.virtual_loss -= 1

    def update(self, value: float):
        """Backpropagates the value through this node."""
        with self.lock:
            self.visit_count += 1
            self.total_value += value
            self.mean_value = self.total_value / self.visit_count


class EvaluatorBase:
    """Base class for evaluating board positions (Policy & Value heads)."""
    def evaluate(self, board: chess.Board) -> Tuple[np.ndarray, float]:
        """Returns (policy_probs, value). policy_probs has shape (4096,)."""
        raise NotImplementedError()

class NeuralNetEvaluator(EvaluatorBase):
    """Evaluates positions using the PyTorch Squeeze-and-Excitation ResNet model."""
    def __init__(self, model, device: str = "cpu"):
        import torch
        self.model = model
        self.device = device
        self.model.eval()

    def evaluate(self, board: chess.Board) -> Tuple[np.ndarray, float]:
        import torch
        # Encode board to 19-plane tensor (active player's perspective)
        tensor = encode_board(board)
        with torch.no_grad():
            tensor_t = torch.tensor(tensor, dtype=torch.float32, device=self.device).unsqueeze(0) # (1, 19, 8, 8)

            # Forward pass
            logits, value_t, _ = self.model(tensor_t)
            logits = logits.squeeze(0).cpu().numpy()
            value = float(value_t.item()) # value is in [-1, 1]

        # Mask illegal moves and apply softmax
        legal_mask = get_legal_move_mask(board)
        masked_logits = np.where(legal_mask, logits, -1e9)
        # Softmax computation
        exp_logits = np.exp(masked_logits - np.max(masked_logits))
        policy_probs = exp_logits / np.sum(exp_logits)

        return policy_probs, value

class StockfishEvaluator(EvaluatorBase):
    """
    Evaluates positions using a local Stockfish binary via UCI commands.
    Used for Stockfish-guided MCTS (Hybrid mode), avoiding any training or heavy overhead.
    """
    def __init__(self, stockfish_path: str = "stockfish", depth: int = 5, time_limit: float = 0.005):
        self.stockfish_path = stockfish_path
        self.depth = depth
        self.time_limit = time_limit
        self.engine = None
        self.lock = threading.Lock()
        
        # Try to launch engine
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        except Exception as e:
            print(f"[Warning] Could not load Stockfish engine at '{stockfish_path}': {e}.")
            print("Falling back to a material heuristic evaluator.")

    def evaluate(self, board: chess.Board) -> Tuple[np.ndarray, float]:
        # If Stockfish engine is not loaded, fallback to material heuristics
        if self.engine is None:
            return self._heuristic_evaluate(board)

        with self.lock:
            try:
                # Run quick analysis
                analysis = self.engine.analyse(board, chess.engine.Limit(depth=self.depth, time=self.time_limit), multipv=3)
                
                # Get the main evaluation score
                main_eval = analysis[0]["score"].relative
                # Convert score to centipawns or mate steps
                cp = main_eval.score(mate_score=10000)
                # Map score to value in [-1, 1] using tanh
                value = math.tanh(cp / 200.0)

                # Get policy priors from moves evaluated
                legal_mask = get_legal_move_mask(board)
                policy_probs = np.zeros(4096, dtype=np.float32)

                move_scores = []
                for pvi in analysis:
                    pv_move = pvi["pv"][0]
                    pv_score = pvi["score"].relative.score(mate_score=10000)
                    move_scores.append((pv_move, pv_score))

                # If no scores are returned, fallback to uniform
                if not move_scores:
                    legal_moves = list(board.legal_moves)
                    for move in legal_moves:
                        policy_probs[encode_move(move)] = 1.0 / len(legal_moves)
                else:
                    # Softmax scores
                    moves = [item[0] for item in move_scores]
                    scores = np.array([item[1] for item in move_scores], dtype=np.float32)
                    # Shift for numerical stability
                    scores_exp = np.exp(scores / 100.0) # temp scale of 100 centipawns
                    scores_prob = scores_exp / np.sum(scores_exp)

                    # Distribute most probability among evaluated PV moves
                    for move, prob in zip(moves, scores_prob):
                        policy_probs[encode_move(move)] = prob * 0.90
                    
                    # Remaining 10% distributed uniformly across all legal moves for exploration
                    legal_moves = list(board.legal_moves)
                    for move in legal_moves:
                        policy_probs[encode_move(move)] += 0.10 / len(legal_moves)
                
                # Ensure it sums to 1.0
                policy_probs = policy_probs / np.sum(policy_probs)
                return policy_probs, value

            except Exception as e:
                # Fallback on any communication error
                return self._heuristic_evaluate(board)

    def _heuristic_evaluate(self, board: chess.Board) -> Tuple[np.ndarray, float]:
        """Lightweight fallback evaluator based on piece values."""
        # Simple material values
        values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        # Calculate material score from active player perspective
        active_color = board.turn
        opponent_color = not active_color
        
        active_score = sum(len(board.pieces(p, active_color)) * val for p, val in values.items())
        opponent_score = sum(len(board.pieces(p, opponent_color)) * val for p, val in values.items())
        
        score_diff = active_score - opponent_score
        value = math.tanh(score_diff / 5.0) # value in [-1, 1]

        # Uniform policy over all legal moves
        legal_moves = list(board.legal_moves)
        policy_probs = np.zeros(4096, dtype=np.float32)
        if legal_moves:
            for move in legal_moves:
                policy_probs[encode_move(move)] = 1.0 / len(legal_moves)
        return policy_probs, value

    def close(self):
        if self.engine is not None:
            try:
                self.engine.quit()
            except:
                pass


class MCTS:
    """
    Performs Monte Carlo Tree Search.
    Supports parallel simulations with virtual loss, Dirichlet noise, and transposition tables.
    """
    def __init__(self, evaluator: EvaluatorBase, c_puct: float = 1.5, num_simulations: int = 800):
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.transposition_table: Dict[str, Tuple[np.ndarray, float]] = {}
        self.tt_lock = threading.Lock()

    def _get_evaluation(self, board: chess.Board) -> Tuple[np.ndarray, float]:
        """Queries the evaluator, utilizing the transposition table for caching."""
        fen = board.fen()
        with self.tt_lock:
            if fen in self.transposition_table:
                return self.transposition_table[fen]

        policy, value = self.evaluator.evaluate(board)

        with self.tt_lock:
            self.transposition_table[fen] = (policy, value)
        return policy, value

    def search(self, board: chess.Board, add_noise: bool = False) -> MCTSNode:
        """
        Runs MCTS search from the root node.
        Returns the root node containing search statistics.
        """
        import time
        import sys
        
        root = MCTSNode(prior=1.0)
        
        # 1. Expand root node
        policy, value = self._get_evaluation(board)
        legal_mask = get_legal_move_mask(board)
        
        # Apply Dirichlet noise at the root if requested (for self-play exploration)
        if add_noise and len(board.legal_moves) > 0:
            epsilon = 0.25
            alpha = 0.3
            noise = np.random.dirichlet([alpha] * len(board.legal_moves))
            
            # Mix noise into policy
            noise_idx = 0
            for action in range(4096):
                if legal_mask[action]:
                    policy[action] = (1 - epsilon) * policy[action] + epsilon * noise[noise_idx]
                    noise_idx += 1
            # Re-normalize
            policy = policy * legal_mask
            policy = policy / np.sum(policy)

        root.expand(policy, legal_mask)

        # 2. Run simulations
        start_time = time.time()
        for i in range(self.num_simulations):
            self._simulate(root, board.copy())
            
            # Print search info periodically (every 50 simulations or on completion)
            if (i + 1) % 50 == 0 or (i + 1) == self.num_simulations:
                elapsed = max(1e-3, time.time() - start_time)
                nps = int((i + 1) / elapsed)
                time_ms = int(elapsed * 1000)
                
                # Find best move from root to get score and pv
                best_child = None
                max_visits = -1
                for child in root.children.values():
                    if child.visit_count > max_visits:
                        max_visits = child.visit_count
                        best_child = child
                
                if best_child:
                    val = best_child.mean_value
                    # Convert Q to centipawns (tanh scale back to cp)
                    # Safe atanh logic
                    safe_val = max(-0.999, min(0.999, val))
                    score_cp = int(math.log((1.0 + safe_val) / (1.0 - safe_val)) * 0.5 * 150.0)
                    
                    # Extract PV line of play
                    pv = []
                    curr = root
                    visited = set()
                    while curr and curr.children:
                        bc = max(curr.children.values(), key=lambda c: c.visit_count, default=None)
                        if bc and bc.visit_count > 0:
                            m_uci = bc.move.uci()
                            if m_uci in visited:
                                break
                            visited.add(m_uci)
                            pv.append(m_uci)
                            curr = bc
                        else:
                            break
                    
                    pv_str = " ".join(pv)
                    depth = len(pv)
                    
                    # Output standard UCI info string
                    print(f"info depth {depth} score cp {score_cp} nodes {i+1} nps {nps} time {time_ms} pv {pv_str}")
                    sys.stdout.flush()
        
        return root

    def _simulate(self, root: MCTSNode, board: chess.Board):
        """Performs a single simulation: selection, expansion, evaluation, and backpropagation."""
        node = root
        search_path = [node]

        # Selection
        while len(node.children) > 0:
            action, node = node.select_child(self.c_puct)
            board.push(node.move)
            search_path.append(node)

        # Apply virtual loss to nodes on the path to simulate concurrent locks
        for n in search_path:
            n.add_virtual_loss()

        # Check for game end
        if board.is_game_over():
            outcome = board.outcome()
            # Outcome value from active player's perspective
            if outcome.winner is None:
                value = 0.0  # Draw
            elif outcome.winner == board.turn:
                value = 1.0  # Current player wins
            else:
                value = -1.0  # Current player loses
        else:
            # Expansion & Evaluation
            policy, value = self._get_evaluation(board)
            legal_mask = get_legal_move_mask(board)
            node.expand(policy, legal_mask)

        # Backpropagation
        # The value returned is for the player whose turn it is on the leaf board.
        # We must invert the value at each step up because the players alternate turns.
        # Start by inverting the value so that each node stores value from the parent's perspective.
        value = -value
        for n in reversed(search_path):
            n.remove_virtual_loss()
            n.update(value)
            value = -value  # Invert for opponent's turn

    def get_move_probs(self, board: chess.Board, temp: float = 1.0) -> np.ndarray:
        """
        Runs search and returns policy probabilities vector of shape (4096,).
        If temp = 1.0, probabilities are proportional to visit counts.
        If temp = 0.0, it is a one-hot vector corresponding to the max-visited move.
        """
        root = self.search(board, add_noise=(temp > 0.5))
        
        visits = np.zeros(4096, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visit_count

        if temp == 0.0:
            # Deterministic selection (max visits)
            best_action = np.argmax(visits)
            probs = np.zeros(4096, dtype=np.float32)
            probs[best_action] = 1.0
            return probs
        else:
            # Stochastic selection (visit counts raised to 1/temp power)
            visits = visits ** (1.0 / temp)
            total = np.sum(visits)
            if total == 0:
                # Fallback if no simulations ran
                legal_mask = get_legal_move_mask(board)
                return legal_mask / np.sum(legal_mask)
            return visits / total
