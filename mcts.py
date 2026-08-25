"""
MODULE 3: mcts.py
Implements Monte Carlo Tree Search (MCTS) enhanced with neural policy/value heads,
Dirichlet noise for exploration, an LRU transposition cache, optional wall-clock
deadlines, and a Stockfish-guided evaluator.

Note: simulations run serially on a single thread (the UCI worker pool is
single-worker). There is intentionally no virtual-loss/lock machinery — it was
dead overhead in a serial search.
"""

import math
import sys
import time
from collections import OrderedDict
from typing import Dict, Tuple, List, Optional
import numpy as np
import chess

import threading
STDOUT_LOCK = threading.Lock()
def uci_print(*args, **kwargs):
    with STDOUT_LOCK:
        print(*args, **kwargs)
        

from board_encoder import (encode_board, encode_move,
                           decode_move, get_legal_move_mask,
                           get_legal_move_mask_for_perspective)

# Maximum number of cached (policy, value) evaluations. Each entry is a 4096-float
# policy (~16 KB), so this bounds cache memory at roughly 1 GB worst case; in
# practice far less because entries are reused across moves within a game.
MAX_TT_ENTRIES = 32768


class MCTSNode:
    """
    Represents a single node in the MCTS tree.
    Tracks visit counts, total values, mean values and prior probabilities.
    """
    def __init__(self, prior: float, parent: Optional['MCTSNode'] = None, move: Optional[chess.Move] = None):
        self.parent = parent
        self.move = move
        self.prior = prior  # P(s, a)
        self.visit_count = 0  # N(s, a)
        self.total_value = 0.0  # W(s, a)
        self.mean_value = 0.0  # Q(s, a)
        self.children: Dict[int, MCTSNode] = {}  # action_index -> MCTSNode

    @property
    def value(self) -> float:
        """Returns Q(s,a)."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def select_child(self, c_puct: float) -> Tuple[int, 'MCTSNode']:
        """
        Selects the child node that maximizes the PUCT formula:
        U(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
        """
        best_score = -float('inf')
        best_action = -1
        best_child = None

        # N(s) is this node's own visit count, maintained during backpropagation.
        sqrt_total = math.sqrt(self.visit_count) if self.visit_count > 0 else 1.0

        for action, child in self.children.items():
            u = child.value + c_puct * child.prior * sqrt_total / (1.0 + child.visit_count)
            if u > best_score:
                best_score = u
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, action_priors: np.ndarray, legal_mask: np.ndarray, board: chess.Board):
        """
        Expands the node by creating children for all legal actions.
        action_priors: Array of size 4096 containing the moves' probabilities.
        """
        for action in range(4096):
            if legal_mask[action]:
                prior = action_priors[action]
                # Reconstruct move from action using decode_move to handle promotions correctly
                move = decode_move(action, board)
                self.children[action] = MCTSNode(prior=prior, parent=self, move=move)

    def update(self, value: float):
        """Backpropagates the value through this node."""
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
        self.model = model
        self.device = device
        self.model.eval()

    def evaluate(self, board: chess.Board) -> Tuple[np.ndarray, float]:
        import torch
        is_black = (board.turn == chess.BLACK)

        # Encode board to 18-plane tensor (active player's perspective)
        tensor = encode_board(board)
        with torch.no_grad():
            tensor_t = torch.tensor(tensor, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, 18, 8, 8)

            # Forward pass
            logits, value_t, _ = self.model(tensor_t)
            logits = logits.squeeze(0).cpu().numpy()
            value = float(value_t.item())  # value is in [-1, 1]

        # Mask illegal moves using perspective-aware mask (matches mirrored board)
        legal_mask = get_legal_move_mask_for_perspective(board)
        masked_logits = np.where(legal_mask, logits, -1e9)
        # Softmax computation
        exp_logits = np.exp(masked_logits - np.max(masked_logits))
        policy_probs_mirrored = exp_logits / np.sum(exp_logits)

        # Unmirror the policy back to the original (non-mirrored) action space
        # so callers can use standard encode_move() indices.
        if is_black:
            policy_probs = np.zeros(4096, dtype=np.float32)
            for action_mirrored in range(4096):
                if policy_probs_mirrored[action_mirrored] > 0:
                    from_sq = chess.square_mirror(action_mirrored // 64)
                    to_sq = chess.square_mirror(action_mirrored % 64)
                    action_orig = from_sq * 64 + to_sq
                    policy_probs[action_orig] = policy_probs_mirrored[action_mirrored]
        else:
            policy_probs = policy_probs_mirrored

        return policy_probs, value

class SubprocessStockfish:
    """A lightweight, synchronous, and robust wrapper for Stockfish via subprocess."""
    def __init__(self, path: str):
        self.path = path
        self.proc = None
        self.reader_thread = None
        self.start()

    def start(self):
        import subprocess
        try:
            self.proc = subprocess.Popen(
                [self.path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1
            )
            self.send("uci")
            self.read_until("uciok", timeout=10.0)
            self.send("isready")
            self.read_until("readyok", timeout=10.0)
        except Exception as e:
            uci_print(f"info string [Warning] Failed to start Stockfish process at {self.path}: {e}")
            
            # Clean up any partially-started process
            if self.proc:
                try:
                    self.proc.kill()
                except Exception:
                    pass
                for pipe in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
                    if pipe:
                        try:
                            pipe.close()
                        except Exception:
                            pass
            self.proc = None

    def send(self, cmd: str):
        if self.proc and self.proc.stdin:
            try:
                self.proc.stdin.write(cmd + "\n")
                self.proc.stdin.flush()
            except Exception:
                pass

    def read_until(self, expected: str, timeout: float = 30.0) -> List[str]:
        """Reads lines from Stockfish stdout until the expected token is found.
        
        Uses a background thread to avoid deadlocking if Stockfish hangs.
        Returns collected lines. If timeout expires, returns whatever was collected.
        """
        import threading
        lines: List[str] = []
        if not self.proc or not self.proc.stdout:
            return lines

        if self.reader_thread is not None and self.reader_thread.is_alive():
            uci_print("info string [Warning] Stockfish reader thread still busy from previous timeout.")
            return lines

        done = threading.Event()

        def _reader():
            try:
                while not done.is_set():
                    line = self.proc.stdout.readline()
                    if not line:
                        break
                    stripped = line.strip()
                    lines.append(stripped)
                    if stripped.startswith(expected):
                        break
            except Exception:
                pass
            finally:
                done.set()

        self.reader_thread = threading.Thread(target=_reader, daemon=True)
        self.reader_thread.start()
        self.reader_thread.join(timeout=timeout)

        if not done.is_set():
            # Timeout: Stockfish is stuck
            done.set()
            uci_print(f"info string [Warning] Stockfish read_until('{expected}') timed out after {timeout}s")
            

        return lines

    def evaluate_position(self, board: chess.Board, depth: int, multipv: int = 3) -> List[str]:
        self.send(f"setoption name MultiPV value {multipv}")
        self.send(f"position fen {board.fen()}")
        self.send(f"go depth {depth}")
        return self.read_until("bestmove")

    def close(self):
        if self.proc:
            try:
                self.send("quit")
                self.proc.wait(timeout=1)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            # Close pipes to avoid ResourceWarning about unclosed file handles
            for pipe in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
                if pipe:
                    try:
                        pipe.close()
                    except Exception:
                        pass
            self.proc = None


class StockfishEvaluator(EvaluatorBase):
    """
    Evaluates positions using a local Stockfish binary via synchronous subprocess UCI commands.
    Used for Stockfish-guided MCTS (Hybrid mode), avoiding any training or heavy overhead.
    """
    def __init__(self, stockfish_path: str = "stockfish", depth: int = 5):
        self.stockfish_path = stockfish_path
        self.depth = depth
        self.engine = None
        
        # Try to launch engine
        self.engine = SubprocessStockfish(self.stockfish_path)
        if self.engine.proc is None:
            uci_print("info string [Warning] Could not load Stockfish engine. Falling back to a material heuristic evaluator.")
            
            self.engine = None

    def _restart_engine(self):
        """Restarts the Stockfish process to recover from any communication or sync errors."""
        if self.engine is not None:
            self.engine.close()
        self.engine = SubprocessStockfish(self.stockfish_path)
        if self.engine.proc is None:
            self.engine = None

    def evaluate(self, board: chess.Board) -> Tuple[np.ndarray, float]:
        # If Stockfish engine is not loaded, fallback to material heuristics
        if self.engine is None or self.engine.proc is None:
            return self._heuristic_evaluate(board)

        try:
            # Query Stockfish synchronously
            lines = self.engine.evaluate_position(board, self.depth, multipv=3)

            # Parse lines
            move_scores_dict = {}
            best_eval = 0
            best_eval_set = False

            for line in lines:
                if line.startswith("info"):
                    tokens = line.split()

                    # Extract multipv
                    multipv = 1
                    if "multipv" in tokens:
                        multipv_idx = tokens.index("multipv") + 1
                        if multipv_idx < len(tokens):
                            multipv = int(tokens[multipv_idx])

                    # Extract score
                    score_val = 0
                    if "score" in tokens:
                        score_idx = tokens.index("score") + 1
                        if score_idx + 1 < len(tokens):
                            score_type = tokens[score_idx] # "cp" or "mate"
                            score_value_str = tokens[score_idx + 1]
                            try:
                                score_value = int(score_value_str)
                                if score_type == "mate":
                                    score_val = 10000 if score_value > 0 else -10000
                                else:
                                    score_val = score_value
                            except Exception:
                                pass

                    # Extract PV move
                    if "pv" in tokens:
                        pv_idx = tokens.index("pv") + 1
                        if pv_idx < len(tokens):
                            pv_move_str = tokens[pv_idx]
                            try:
                                pv_move = chess.Move.from_uci(pv_move_str)
                                # Only add if it's a legal move for the current board
                                if pv_move in board.legal_moves:
                                    move_scores_dict[pv_move] = score_val
                            except Exception:
                                pass

                    # Save the score of multipv 1 as main eval
                    if multipv == 1:
                        best_eval = score_val
                        best_eval_set = True

            if not best_eval_set:
                return self._heuristic_evaluate(board)

            move_scores = list(move_scores_dict.items())

            # Map score to value in [-1, 1] using tanh
            value = math.tanh(best_eval / 200.0)

            # Get policy priors from moves evaluated
            policy_probs = np.zeros(4096, dtype=np.float32)

            # If no scores are returned, fallback to uniform
            if not move_scores:
                legal_moves = list(board.legal_moves)
                for move in legal_moves:
                    policy_probs[encode_move(move)] = 1.0 / len(legal_moves)
            else:
                # Softmax scores with numerical stability
                moves = list(move_scores_dict.keys())
                scores = list(move_scores_dict.values())

                # Convert scores (centipawns) to probabilities
                # Use a softmax with a temperature scale
                scaled_scores = np.array(scores, dtype=np.float32) / 100.0

                # Prevent np.exp overflow by subtracting the max score
                max_score = np.max(scaled_scores)
                exp_scores = np.exp(scaled_scores - max_score)
                scores_prob = exp_scores / np.sum(exp_scores)

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
            uci_print(f"info string [Warning] Stockfish evaluation error: {e}. Restarting Stockfish process...")
            
            self._restart_engine()
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
            self.engine.close()


class MCTS:
    """
    Performs Monte Carlo Tree Search.
    Supports Dirichlet noise, an LRU-bounded transposition table,
    wall-clock deadlines and hybrid Alpha-Beta prior boosting for tactical awareness.
    """
    def __init__(self, evaluator: EvaluatorBase, c_puct: float = 1.5, num_simulations: int = 800):
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.transposition_table: 'OrderedDict[str, Tuple[np.ndarray, float]]' = OrderedDict()
        self.abort_search = False

    def clear_cache(self):
        """Drops all cached evaluations (call when the evaluator changes or a new game starts)."""
        self.transposition_table.clear()

    def _get_evaluation(self, board: chess.Board) -> Tuple[np.ndarray, float]:
        """Queries the evaluator, utilizing the LRU transposition table for caching."""
        fen = board.fen()
        cached = self.transposition_table.get(fen)
        if cached is not None:
            self.transposition_table.move_to_end(fen)
            return cached

        policy, value = self.evaluator.evaluate(board)

        self.transposition_table[fen] = (policy, value)
        while len(self.transposition_table) > MAX_TT_ENTRIES:
            self.transposition_table.popitem(last=False)
        return policy, value

    def search(self, board: chess.Board, add_noise: bool = False,
               noise_epsilon: float = 0.25, noise_alpha: float = 0.3,
               ab_candidates: 'Optional[List[tuple]]' = None,
               deadline: Optional[float] = None) -> MCTSNode:
        """
        Runs MCTS search from the root node.
        Returns the root node containing search statistics.
        
        Args:
            board: Current board position.
            add_noise: Whether to add Dirichlet exploration noise at root.
            noise_epsilon: Weight of Dirichlet noise vs policy (0.0-1.0).
            noise_alpha: Dirichlet distribution alpha parameter.
            ab_candidates: Optional list of (chess.Move, score_cp) tuples from
                          Alpha-Beta filter to boost in the MCTS prior.
            deadline: Optional absolute time.time() value; no new simulation is
                      started once it is exceeded (time-control safety).
        """
        import sys
        
        root = MCTSNode(prior=1.0)
        
        # 1. Expand root node
        policy, value = self._get_evaluation(board)
        legal_mask = get_legal_move_mask(board)
        
        # 1b. Boost priors for Alpha-Beta candidate moves
        if ab_candidates and len(ab_candidates) > 0:
            # Give AB candidates a significant prior boost proportional to their rank
            ab_boost_total = 0.40  # Allocate 40% of total prior mass to AB candidates
            remaining_mass = 1.0 - ab_boost_total
            
            # Scale existing policy down
            policy = policy * remaining_mass
            
            # Distribute boost among AB candidates (weighted by rank)
            n_cands = len(ab_candidates)
            for rank, (move, score_cp) in enumerate(ab_candidates):
                action = encode_move(move)
                if legal_mask[action]:
                    # Higher-ranked moves get more boost (linear decay)
                    weight = (n_cands - rank) / sum(range(1, n_cands + 1))
                    policy[action] += ab_boost_total * weight
            
            # Re-normalize
            policy = policy * legal_mask
            total = np.sum(policy)
            if total > 0:
                policy = policy / total
        
        # 1c. Apply Dirichlet noise at the root for exploration
        legal_moves = list(board.legal_moves)
        if add_noise and len(legal_moves) > 0:
            noise = np.random.dirichlet([noise_alpha] * len(legal_moves))
            
            # Mix noise into policy
            noise_idx = 0
            for action in range(4096):
                if legal_mask[action]:
                    policy[action] = (1 - noise_epsilon) * policy[action] + noise_epsilon * noise[noise_idx]
                    noise_idx += 1
            # Re-normalize
            policy = policy * legal_mask
            policy = policy / np.sum(policy)

        root.expand(policy, legal_mask, board)

        # 2. Run simulations
        start_time = time.time()
        for i in range(self.num_simulations):
            if self.abort_search:
                break
            if deadline is not None and time.time() >= deadline:
                break
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
                    uci_print(f"info depth {depth} score cp {score_cp} nodes {i+1} nps {nps} time {time_ms} pv {pv_str}")
                    
        
        return root

    def _simulate(self, root: MCTSNode, board: chess.Board):
        """Performs a single simulation: selection, expansion, evaluation, and backpropagation."""
        node = root
        search_path = [node]

        # Selection
        while len(node.children) > 0:
            action, next_node = node.select_child(self.c_puct)
            if next_node is None:
                break
            node = next_node
            board.push(node.move)
            search_path.append(node)

        # Check for game end
        if board.is_game_over(claim_draw=True):
            outcome = board.outcome(claim_draw=True)
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
            node.expand(policy, legal_mask, board)

        # Backpropagation
        # The value returned is for the player whose turn it is on the leaf board.
        # We must invert the value at each step up because the players alternate turns.
        # Start by inverting the value so that each node stores value from the parent's perspective.
        value = -value
        for n in reversed(search_path):
            n.update(value)
            value = -value  # Invert for opponent's turn

    def get_visit_counts(self, board: chess.Board, add_noise: bool = False,
                         noise_epsilon: float = 0.25, noise_alpha: float = 0.3,
                         ab_candidates: 'Optional[List[tuple]]' = None,
                         deadline: Optional[float] = None) -> np.ndarray:
        """
        Runs MCTS search and returns the raw visit count array of shape (4096,).
        Move selection (temperature, top-K) is handled externally by DifficultyController.
        
        Args:
            board: Current board position.
            add_noise: Whether to add Dirichlet exploration noise.
            noise_epsilon: Noise weight.
            noise_alpha: Dirichlet alpha.
            ab_candidates: Alpha-Beta candidate moves to boost.
            deadline: Optional absolute time.time() after which no new simulations start.
        
        Returns:
            np.ndarray of shape (4096,) with visit counts per action.
        """
        root = self.search(board, add_noise=add_noise,
                          noise_epsilon=noise_epsilon, noise_alpha=noise_alpha,
                          ab_candidates=ab_candidates, deadline=deadline)
        
        visits = np.zeros(4096, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        
        return visits

    def get_move_probs(self, board: chess.Board, temp: float = 1.0) -> np.ndarray:
        """
        Runs search and returns policy probabilities vector of shape (4096,).
        If temp = 1.0, probabilities are proportional to visit counts.
        If temp = 0.0, it is a one-hot vector corresponding to the max-visited move.
        
        Note: For the hybrid engine with difficulty control, prefer using
        get_visit_counts() + select_move_with_variety() from difficulty.py instead.
        This method is retained for backward compatibility with evaluator.py and self_play.py.
        """
        root = self.search(board, add_noise=(temp > 0.5))
        
        visits = np.zeros(4096, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visit_count

        total_visits = np.sum(visits)

        if temp == 0.0:
            # Deterministic selection (max visits)
            if total_visits == 0:
                # Fallback if no simulations ran or all visits are 0
                legal_mask = get_legal_move_mask(board)
                if root.children:
                    # Select the move with the highest prior probability
                    best_action = max(root.children.keys(), key=lambda act: root.children[act].prior)
                else:
                    best_action = int(np.argmax(legal_mask)) if np.any(legal_mask) else 0
            else:
                best_action = np.argmax(visits)
                
            probs = np.zeros(4096, dtype=np.float32)
            probs[best_action] = 1.0
            return probs
        else:
            # Stochastic selection (visit counts raised to 1/temp power)
            if total_visits == 0:
                # Fallback if no simulations ran
                legal_mask = get_legal_move_mask(board)
                return legal_mask / np.sum(legal_mask) if np.any(legal_mask) else np.ones(4096) / 4096
                
            visits = visits ** (1.0 / temp)
            total = np.sum(visits)
            return visits / total
