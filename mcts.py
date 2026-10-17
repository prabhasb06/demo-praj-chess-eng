"""Serial MCTS with cancellable Stockfish advice and bounded sparse evaluation cache."""
import math
import queue
import subprocess
import sys
import threading
import time
import weakref
from collections import OrderedDict

import chess
import numpy as np
from board_encoder import (POLICY_SIZE, encode_board, encode_move, decode_move,
                           encode_move_for_perspective, get_legal_move_mask)

STDOUT_LOCK = threading.Lock()
MAX_TT_ENTRIES = 8192
MAX_TREE_NODES = 65536


def uci_print(*args, **kwargs):
    with STDOUT_LOCK:
        print(*args, **kwargs, flush=True)


class MCTSNode:
    def __init__(self, prior, parent=None, move=None):
        self.parent = weakref.proxy(parent) if parent is not None else None
        self.move, self.prior = move, float(prior)
        self.visit_count = 0
        self.total_value = self.mean_value = 0.0
        self.children = {}

    @property
    def value(self):
        return self.mean_value

    def select_child(self, c_puct):
        scale = math.sqrt(max(1, self.visit_count))
        return max(self.children.items(), key=lambda item:
                   item[1].value + c_puct * item[1].prior * scale / (1 + item[1].visit_count))

    def expand(self, action_priors, legal_mask, board):
        for move in board.legal_moves:
            action = encode_move(move)
            if legal_mask[action]:
                self.children[action] = MCTSNode(action_priors[action], self, move)

    def update(self, value):
        self.visit_count += 1
        self.total_value += value
        self.mean_value = self.total_value / self.visit_count


class EvaluatorBase:
    def evaluate(self, board):
        raise NotImplementedError


class NeuralNetEvaluator(EvaluatorBase):
    def __init__(self, model, device="cpu"):
        self.model, self.device = model, device
        self.model.eval()

    def evaluate(self, board):
        import torch
        legal = list(board.legal_moves)
        probs = np.zeros(POLICY_SIZE, dtype=np.float32)
        if not legal:
            return probs, -1.0 if board.is_checkmate() else 0.0
        self.model.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(encode_board(board), device=self.device).unsqueeze(0)
            logits, value, _ = self.model(tensor)
            logits = logits[0].cpu().numpy()
        indices = [encode_move_for_perspective(m, board.turn == chess.BLACK) for m in legal]
        scores = logits[indices].astype(np.float64)
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()
        for move, weight in zip(legal, weights):
            probs[encode_move(move)] = weight
        return probs, float(value.item())


class SubprocessStockfish:
    """One reader owns stdout for the process lifetime; stop drains through bestmove."""
    def __init__(self, path):
        self.path, self.proc = path, None
        self.lines = queue.Queue()
        self.deadline = None
        self.cancel = lambda: False
        self.reader_thread = None
        self.last_complete = False
        try:
            self.proc = subprocess.Popen([path] if isinstance(path, str) else path, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.reader_thread = threading.Thread(target=self._reader, args=(self.proc,), daemon=True)
            self.reader_thread.start()
            self.send("uci")
            self.read_until("uciok", 5)
            self.send("setoption name Threads value 1")
            self.send("setoption name Hash value 64")
            self.send("isready")
            self.read_until("readyok", 5)
        except (OSError, TimeoutError):
            self.close()

    def _reader(self, proc):
        try:
            for line in proc.stdout:
                self.lines.put(line.strip())
        finally:
            self.lines.put(None)

    def send(self, command):
        if self.proc is None or self.proc.poll() is not None:
            raise OSError("Stockfish is not running")
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def read_until(self, expected, timeout=30.0):
        end = time.monotonic() + timeout
        stopping = False
        result = []
        while True:
            now = time.monotonic()
            cancelled = expected == "bestmove" and (self.cancel() or
                (self.deadline is not None and now >= self.deadline))
            if (cancelled or now >= end) and not stopping:
                if expected != "bestmove":
                    raise TimeoutError("Stockfish handshake timed out")
                self.send("stop")
                stopping = True
                end = now + 0.25
            if stopping and now >= end:
                self.close()
                raise TimeoutError("Stockfish did not stop")
            try:
                line = self.lines.get(timeout=0.002)
            except queue.Empty:
                continue
            if line is None:
                raise OSError("Stockfish exited")
            result.append(line)
            if line.split()[:1] == [expected]:
                self.last_complete = not stopping
                return result

    def evaluate_position(self, board, depth, multipv=3):
        self.send(f"setoption name MultiPV value {multipv}")
        # Preserve game history for repetition-aware evaluations.
        moves = " ".join(m.uci() for m in board.move_stack)
        self.send(f"position fen {board.root().fen()} moves {moves}")
        self.send(f"go depth {max(1, depth)}")
        return self.read_until("bestmove")

    def close(self):
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            proc.stdin.write("quit\n")
            proc.stdin.flush()
            proc.wait(timeout=0.3)
        except (OSError, subprocess.TimeoutExpired):
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=1)
        if self.reader_thread:
            self.reader_thread.join(timeout=1)
        for pipe in (proc.stdin, proc.stdout):
            if pipe:
                try:
                    pipe.close()
                except OSError:
                    pass  # An already-dead child can reject the final stdin flush.


class StockfishEvaluator(EvaluatorBase):
    def __init__(self, stockfish_path="stockfish", depth=5, multipv=3):
        self.stockfish_path, self.depth, self.multipv = stockfish_path, depth, multipv
        self.engine = SubprocessStockfish(stockfish_path)
        self.deadline = None
        self.cancel = lambda: False
        self.cacheable = True

    def _heuristic_evaluate(self, board):
        from alpha_beta import evaluate_board
        policy = get_legal_move_mask(board).astype(np.float32)
        if policy.sum():
            policy /= policy.sum()
        return policy, math.tanh(evaluate_board(board) / 600.0)

    def evaluate(self, board):
        self.cacheable = False
        if self.cancel() or (self.deadline is not None and time.monotonic() >= self.deadline):
            return self._heuristic_evaluate(board)
        if self.engine is None or self.engine.proc is None:
            return self._heuristic_evaluate(board)
        self.engine.cancel, self.engine.deadline = self.cancel, self.deadline
        try:
            lines = self.engine.evaluate_position(board, self.depth, self.multipv)
        except (OSError, TimeoutError):
            self.engine.close()
            uci_print("info string Stockfish query failed; using static evaluation")
            return self._heuristic_evaluate(board)
        # Retain only deepest completed information for each MultiPV rank.
        slots = {}
        for line in lines:
            tokens = line.split()
            if not tokens or tokens[0] != "info" or "score" not in tokens or "pv" not in tokens:
                continue
            if "lowerbound" in tokens or "upperbound" in tokens:
                continue
            try:
                rank = int(tokens[tokens.index("multipv") + 1]) if "multipv" in tokens else 1
                depth = int(tokens[tokens.index("depth") + 1])
                idx = tokens.index("score")
                kind, score = tokens[idx + 1], int(tokens[idx + 2])
                move = chess.Move.from_uci(tokens[tokens.index("pv") + 1])
                if move not in board.legal_moves:
                    continue
                if kind == "mate":
                    cp = (10000 - min(1000, abs(score))) * (1 if score > 0 else -1)
                    value = (1 - min(100, abs(score)) * 0.0005) * (1 if score > 0 else -1)
                else:
                    cp, value = score, 0.95 * math.tanh(score / 600.0)
                if rank not in slots or depth >= slots[rank][0]:
                    slots[rank] = (depth, move, cp, value)
            except (ValueError, IndexError):
                continue
        if 1 not in slots:
            return self._heuristic_evaluate(board)
        primary = slots[1]
        current = [s for s in slots.values() if s[0] == primary[0]]
        scores = np.array([s[2] for s in current], dtype=np.float64) / 80.0
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()
        policy = get_legal_move_mask(board).astype(np.float32)
        policy *= 0.02 / max(1, policy.sum())
        for entry, weight in zip(current, weights):
            policy[encode_move(entry[1])] += 0.98 * weight
        self.cacheable = self.engine.last_complete
        return policy, primary[3]

    def close(self):
        if self.engine:
            self.engine.close()


class MCTS:
    def __init__(self, evaluator, c_puct=1.5, num_simulations=800):
        self.evaluator, self.c_puct, self.num_simulations = evaluator, c_puct, num_simulations
        self.transposition_table = OrderedDict()
        self.abort_search = False
        self.cancel = lambda: False
        self.last_root = None
        self.deadline = None
        self.max_depth = None
        self.depth_reached = 0
        self.progress = None
        self.tree_nodes = 0

    def clear_cache(self):
        self.transposition_table.clear()

    def _get_evaluation(self, board):
        # Halfmove clock and reversible history affect NN input and draw scores.
        history = tuple(m.uci() for m in board.move_stack[-board.halfmove_clock:]) if board.halfmove_clock else ()
        key = (' '.join(board.fen().split()[:5]), history, getattr(self.evaluator, "depth", None),
               getattr(self.evaluator, "multipv", None))
        cached = self.transposition_table.get(key)
        if cached is not None:
            self.transposition_table.move_to_end(key)
            indices, weights, value = cached
            policy = np.zeros(POLICY_SIZE, dtype=np.float32)
            policy[indices] = weights
            return policy, value
        policy, value = self.evaluator.evaluate(board)
        policy = np.asarray(policy, dtype=np.float32).copy()
        mask = get_legal_move_mask(board)
        policy = np.where(mask & np.isfinite(policy), np.maximum(policy, 0), 0)
        if policy.sum() == 0:
            policy = mask.astype(np.float32)
        if policy.sum():
            policy /= policy.sum()
        value = float(np.clip(value, -1, 1)) if np.isfinite(value) else 0.0
        if getattr(self.evaluator, "cacheable", True):
            indices = np.flatnonzero(policy).astype(np.uint16)
            self.transposition_table[key] = (indices, policy[indices].copy(), value)
            if len(self.transposition_table) > MAX_TT_ENTRIES:
                self.transposition_table.popitem(last=False)
        return policy, value

    def _stopped(self):
        return self.abort_search or self.cancel() or (self.deadline is not None and time.monotonic() >= self.deadline)

    def search(self, board, **kwargs):
        settings = {name: getattr(self.evaluator, name) for name in
                    ('depth', 'multipv', 'deadline', 'cancel') if hasattr(self.evaluator, name)}
        try:
            return self._search(board, **kwargs)
        finally:
            for name, value in settings.items():
                setattr(self.evaluator, name, value)

    def _search(self, board, add_noise=False, noise_epsilon=0.25, noise_alpha=0.3,
               ab_candidates=None, deadline=None, root_moves=None, max_depth=None):
        # Legacy callers supply epoch deadlines; internal clocks are monotonic.
        self.deadline = (time.monotonic() + deadline - time.time()
                         if deadline is not None and deadline > 1000000000 else deadline)
        self.max_depth = max_depth
        self.depth_reached = 0
        root = self.last_root = MCTSNode(1)
        mask = get_legal_move_mask(board)
        if root_moves is not None:
            allowed = np.zeros(POLICY_SIZE, dtype=bool)
            for move in root_moves:
                allowed[encode_move(move)] = True
            mask &= allowed
        if not mask.any():
            return root
        if isinstance(self.evaluator, StockfishEvaluator):
            self.evaluator.cancel = lambda: self.abort_search or self.cancel()
            self.evaluator.deadline = self.deadline
            original_depth, original_multipv = self.evaluator.depth, self.evaluator.multipv
            if self.deadline is not None:
                self.evaluator.deadline = time.monotonic() + max(0, self.deadline - time.monotonic()) * 0.6
        if self._stopped():
            policy = mask.astype(np.float32) / mask.sum()
            root.evaluation_value = 0.0
        else:
            policy, root.evaluation_value = self._get_evaluation(board)
        if isinstance(self.evaluator, StockfishEvaluator):
            self.evaluator.deadline = self.deadline
            self.evaluator.depth = min(original_depth, 6)
            self.evaluator.multipv = 1
        policy = policy.copy() * mask
        if ab_candidates:
            for rank, (move, _) in enumerate(ab_candidates):
                if mask[encode_move(move)]:
                    policy[encode_move(move)] += 0.4 / (rank + 1)
        if policy.sum() == 0:
            policy = mask.astype(np.float32)
        policy /= policy.sum()
        indices = np.flatnonzero(mask)
        if add_noise:
            policy[indices] = ((1 - noise_epsilon) * policy[indices]
                               + noise_epsilon * np.random.dirichlet([noise_alpha] * len(indices)))
        root.expand(policy, mask, board)
        self.tree_nodes = 1 + len(root.children)
        # A verified immediate mate dominates search, including underpromotions.
        for child in root.children.values():
            board.push(child.move)
            mate = board.is_checkmate()
            board.pop()
            if mate:
                root.proven_move = child.move
                child.update(1.0)
                root.update(-1.0)
                return root
        next_info = time.monotonic() + 0.25
        for _ in range(self.num_simulations):
            if self._stopped():
                break
            self._simulate(root, board.copy())
            if self.progress and time.monotonic() >= next_info:
                self.progress(root)
                next_info = time.monotonic() + 0.25
        return root

    def _simulate(self, root, board):
        node, path = root, [root]
        while node.children and (self.max_depth is None or len(path) <= self.max_depth):
            if self._stopped():
                return
            _, node = node.select_child(self.c_puct)
            board.push(node.move)
            path.append(node)
        outcome = board.outcome(claim_draw=False)
        self.depth_reached = max(self.depth_reached, len(path) - 1)
        if outcome is not None:
            value = 0.0 if outcome.winner is None else (1.0 if outcome.winner == board.turn else -1.0)
        else:
            if self.max_depth is not None and len(path) - 1 >= self.max_depth:
                from alpha_beta import evaluate_board
                policy = get_legal_move_mask(board).astype(np.float32)
                value = math.tanh(evaluate_board(board) / 600.0)
            else:
                if isinstance(self.evaluator, StockfishEvaluator) and self.max_depth is not None:
                    self.evaluator.depth = max(1, min(6, self.max_depth - len(path) + 1))
                policy, value = self._get_evaluation(board)
            if self._stopped():
                return
            # Claiming is an option, not a forced end to a winning line.
            if board.can_claim_fifty_moves() or (board.halfmove_clock >= 7 and board.can_claim_threefold_repetition()):
                value = max(0.0, value)
            if self.max_depth is None or len(path) - 1 < self.max_depth:
                mask = get_legal_move_mask(board)
                if self.tree_nodes + int(mask.sum()) <= MAX_TREE_NODES:
                    node.expand(policy, mask, board)
                    self.tree_nodes += len(node.children)
        value = -value
        for node in reversed(path):
            node.update(value)
            value = -value

    def get_visit_counts(self, board, **kwargs):
        root = self.search(board, **kwargs)
        visits = np.zeros(POLICY_SIZE, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        if not visits.any():
            for action, child in root.children.items():
                visits[action] = child.prior
        return visits

    def get_move_probs(self, board, temp=1.0):
        if temp < 0 or not math.isfinite(temp):
            raise ValueError("Temperature must be finite and nonnegative")
        visits = self.get_visit_counts(board, add_noise=temp > 0.5)
        probs = np.zeros(POLICY_SIZE, dtype=np.float32)
        indices = np.flatnonzero(visits)
        if not len(indices):
            return probs
        if temp == 0:
            probs[int(np.argmax(visits))] = 1
        else:
            logits = np.log(visits[indices].astype(np.float64)) / temp
            weights = np.exp(logits - logits.max())
            probs[indices] = weights / weights.sum()
        return probs
