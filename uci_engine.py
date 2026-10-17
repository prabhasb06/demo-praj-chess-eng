"""UCI frontend. A single worker owns evaluators; each search owns its cancellation state."""
import os
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import chess
import numpy as np

from board_encoder import decode_move
from engine_info import NAME, VERSION, AUTHOR
from difficulty import DifficultyController, select_move_with_variety
from mcts import MCTS, StockfishEvaluator, uci_print


def _num(token):
    try:
        return int(token)
    except (TypeError, ValueError, OverflowError):
        try:
            return int(float(token))
        except (TypeError, ValueError, OverflowError):
            return None


GO_VALUES = {"wtime", "btime", "winc", "binc", "movestogo", "movetime", "depth", "nodes", "mate"}
GO_WORDS = GO_VALUES | {"go", "infinite", "ponder", "searchmoves"}


def parse_go(tokens, board):
    result = {}
    i = 1
    while i < len(tokens):
        word = tokens[i]
        if word in GO_VALUES and i + 1 < len(tokens):
            number = _num(tokens[i + 1])
            if number is not None:
                result[word] = max(0, number)
            i += 2
        elif word == "searchmoves":
            moves = []
            i += 1
            while i < len(tokens) and tokens[i] not in GO_WORDS:
                try:
                    move = board.parse_uci(tokens[i])
                    if move and move not in moves:
                        moves.append(move)
                except ValueError:
                    pass
                i += 1
            result[word] = moves
        else:
            if word in {"infinite", "ponder"}:
                result[word] = True
            i += 1
    return result


def parse_time_from_go(tokens, board, max_think_time):
    limits = parse_go(tokens, board)
    if limits.get("infinite"):
        return None, True
    if "movetime" in limits:
        return max(0.0, limits["movetime"] / 1000.0), False
    remaining = limits.get("wtime" if board.turn else "btime")
    if remaining is None:
        return None, False
    increment = limits.get("winc" if board.turn else "binc", 0)
    # No minimum allocation that can exceed a nearly exhausted clock.
    reserve = min(50, remaining * 0.1)
    available = max(0, remaining - reserve)
    moves = max(1, limits.get("movestogo", 30))
    allocation = min(available / moves + increment * 0.8, available * 0.2)
    return min(max_think_time, allocation / 1000), False


def parse_position(tokens):
    if len(tokens) < 2:
        raise ValueError("Missing position")
    if tokens[1] == "startpos":
        board, index = chess.Board(), 2
    elif tokens[1] == "fen":
        index = tokens.index("moves") if "moves" in tokens else len(tokens)
        board = chess.Board(" ".join(tokens[2:index]))
    else:
        raise ValueError("Unknown position format")
    if not board.is_valid():
        raise ValueError("Invalid chess position")
    if index < len(tokens):
        if tokens[index] != "moves":
            raise ValueError("Expected moves")
        for token in tokens[index + 1:]:
            move = board.parse_uci(token)
            if not move:
                raise ValueError("Null moves are not game moves")
            board.push(move)
    return board


def _parse_setoption(tokens):
    lower = [t.lower() for t in tokens]
    if "name" not in lower:
        return None, None
    start = lower.index("name") + 1
    end = lower.index("value") if "value" in lower else len(tokens)
    return " ".join(lower[start:end]), " ".join(tokens[end + 1:]) if end < len(tokens) else None


def default_stockfish_path(base):
    """Resolve relative to the application, independently of a GUI's directory."""
    bundled = os.path.join(base, "engines", "stockfish.exe")
    return bundled if os.path.isfile(bundled) else os.path.join(base, "stockfish.exe")


@dataclass
class SearchJob:
    board: chess.Board
    tokens: list
    params: object
    variety: bool
    started: float = field(default_factory=time.monotonic)
    stop: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    superseded: bool = False
    emitted: bool = False


class UciEngine:
    def __init__(self):
        base = os.path.dirname(sys.executable if getattr(sys, "frozen", False) else __file__)
        self.path = default_stockfish_path(base)
        self.difficulty = DifficultyController(10)
        self.variety = False
        self.board = chess.Board()
        self.lock = threading.RLock()
        self.job = None
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.evaluator = None
        self.mcts = None
        self.backend_notice = None
        self.pending = self.pool.submit(self.initialize)

    def initialize(self):
        if self.evaluator:
            self.evaluator.close()
        self.evaluator = StockfishEvaluator(self.path)
        self.mcts = MCTS(self.evaluator)
        self.backend_notice = None

    def report_backend(self, job):
        """Report evaluator changes during search, without startup diagnostics."""
        engine = self.evaluator.engine
        assisted = engine is not None and engine.proc is not None and engine.proc.poll() is None
        state = "stockfish" if assisted else "static"
        with self.lock:
            if job.superseded or self.backend_notice == state:
                return
            if assisted:
                uci_print("info string Evaluation: Stockfish-assisted MCTS")
            else:
                uci_print("info string Evaluation: static fallback; configure StockfishPath for assisted play")
            self.backend_notice = state

    def supersede(self):
        with self.lock:
            if self.job:
                self.job.superseded = True
                self.job.stop.set()
                self.job.release.set()

    def emit(self, job, move, root=None):
        with self.lock:
            if job.superseded or job.emitted:
                return
            job.emitted = True
            self.info(job, root, move)
            uci_print(f"bestmove {move.uci()}")

    def info(self, job, root, move=None):
        with self.lock:
            if job.superseded:
                return
            elapsed = max(1, int((time.monotonic() - job.started) * 1000))
            nodes = root.visit_count if root is not None else 0
            value = getattr(root, 'evaluation_value', 0.0)
            if move is None and root and root.children:
                child = max(root.children.values(), key=lambda c: c.visit_count + 12 * c.prior)
                move = child.move
                value = child.value if child.visit_count else value
            score = 'mate 1' if getattr(root, 'proven_move', None) else 'cp ' + str(int(600 * math.atanh(max(-0.99, min(0.99, value)))))
            depth = self.mcts.depth_reached if root else 0
            uci_print(f"info depth {depth} score {score} nodes {nodes} nps {int(nodes*1000/elapsed)} time {elapsed} pv {move.uci() if move else ''}")

    def search(self, job):
        if job.superseded:
            return
        legal = list(job.board.legal_moves)
        limits = parse_go(job.tokens, job.board)
        if "searchmoves" in limits:
            legal = [m for m in legal if m in limits["searchmoves"]]
        best = legal[0] if legal else chess.Move.null()
        root = None
        try:
            self.report_backend(job)
            if legal and not job.stop.is_set():
                params = job.params
                budget, infinite = parse_time_from_go(job.tokens, job.board, params.max_think_time)
                if budget is None and not infinite:
                    budget = params.max_think_time
                deadline = None if infinite else job.started + max(0, budget - min(0.01, budget * 0.1))
                self.mcts.abort_search = False
                self.mcts.cancel = job.stop.is_set
                self.mcts.progress = lambda root: self.info(job, root)
                self.mcts.c_puct = params.c_puct
                self.mcts.num_simulations = max(1, limits.get("nodes", params.num_simulations))
                if infinite and "nodes" not in limits:
                    self.mcts.num_simulations = 1000000000
                depth = limits.get("depth")
                if "mate" in limits:
                    depth = max(1, 2 * limits["mate"] - 1)
                self.evaluator.depth = max(1, min(params.stockfish_depth, depth)) if depth else params.stockfish_depth
                self.evaluator.multipv = params.stockfish_multipv
                counts = self.mcts.get_visit_counts(job.board,
                    add_noise=job.variety and params.add_noise,
                    noise_epsilon=params.noise_epsilon, noise_alpha=params.noise_alpha,
                    deadline=deadline, root_moves=legal, max_depth=max(1, depth) if depth is not None else None)
                root = self.mcts.last_root
                if job.variety:
                    # Apply variety only to allowed roots.
                    restricted = counts.copy()
                    from board_encoder import encode_move
                    allowed = {encode_move(m) for m in legal}
                    action = select_move_with_variety(restricted, params, job.board)
                    if action not in allowed:
                        action = max(allowed, key=lambda a: counts[a])
                else:
                    # Root advice acts as a small virtual sample, preventing one
                    # speculative visit from overturning the stronger root search.
                    action = max(root.children, key=lambda a:
                        root.children[a].visit_count + 12 * root.children[a].prior)
                move = decode_move(action, job.board)
                if getattr(root, 'proven_move', None) is not None:
                    move = root.proven_move
                if move in legal:
                    best = move
        except Exception as error:
            uci_print(f"info string Search failed: {type(error).__name__}: {error}")
        # Completed ponder/infinite searches must hold even terminal/fallback results.
        if ("ponder" in limits or limits.get("infinite")) and not job.stop.is_set():
            job.release.wait()
        self.emit(job, best, root)

    def handle(self, line):
        tokens = line.split()
        if not tokens:
            return True
        command = tokens[0]
        if command == "uci":
            uci_print(f"id name {NAME} {VERSION}")
            uci_print(f"id author {AUTHOR}")
            uci_print("option name Difficulty type spin default 10 min 1 max 10")
            uci_print("option name MoveVariety type check default false")
            uci_print("option name Ponder type check default false")
            uci_print(f"option name StockfishPath type string default {self.path}")
            uci_print("option name Clear Hash type button")
            uci_print("uciok")
        elif command == "help":
            uci_print(f"{NAME} is a UCI chess engine by {AUTHOR}.")
            uci_print("Use a chess GUI for a graphical board, or enter UCI commands below.")
            uci_print("  uci                       Show identity and available options")
            uci_print("  isready                   Check engine readiness")
            uci_print("  position startpos         Set the starting position")
            uci_print("  position startpos moves e2e4 e7e5")
            uci_print("  go movetime 1000          Search for one second")
            uci_print("  stop                      Stop the current search")
            uci_print("  d                         Display the current board and FEN")
            uci_print("  setoption name StockfishPath value C:\\path\\to\\stockfish.exe")
            uci_print("  quit                      Exit the engine")
        elif command == "d":
            border = "  +---+---+---+---+---+---+---+---+"
            rows = [border]
            for rank in range(7, -1, -1):
                pieces = [self.board.piece_at(chess.square(file, rank)) for file in range(8)]
                rows.append(str(rank + 1) + " | " + " | ".join(
                    piece.symbol() if piece else " " for piece in pieces) + " |")
                rows.append(border)
            rows.extend(("    a   b   c   d   e   f   g   h", "", "Fen: " + self.board.fen()))
            uci_print("\n".join(rows))
        elif command == "isready":
            # Ready must wait for configuration, but never wait for an active search.
            self.pending.add_done_callback(lambda future: uci_print("readyok"))
        elif command == "setoption":
            name, value = _parse_setoption(tokens)
            if name == "difficulty" and value is not None:
                number = _num(value)
                if number is not None:
                    self.difficulty.set_level(number)
            elif name == "movevariety" and value is not None:
                self.variety = value.lower() == "true"
            elif name == "stockfishpath" and value:
                if not os.path.isfile(value):
                    uci_print("info string Rejected StockfishPath: file does not exist")
                else:
                    self.supersede()
                    self.path = os.path.abspath(value)
                    self.pending = self.pool.submit(self.initialize)
            elif name == "clear hash":
                self.supersede()
                self.pending = self.pool.submit(self.clear_cache)
        elif command == "ucinewgame":
            self.supersede()
            self.board = chess.Board()
            self.pending = self.pool.submit(self.clear_cache)
        elif command == "position":
            self.supersede()
            try:
                self.board = parse_position(tokens)
            except ValueError as error:
                # Transactional: never commit an illegal or partially applied board.
                uci_print(f"info string Rejected position: {error}")
        elif command == "go":
            self.supersede()
            job = SearchJob(self.board.copy(), tokens, self.difficulty.get_params(), self.variety)
            with self.lock:
                self.job = job
            self.pool.submit(self.search, job)
        elif command == "stop":
            with self.lock:
                if self.job:
                    self.job.stop.set()
                    self.job.release.set()
        elif command == "ponderhit":
            with self.lock:
                if self.job and "ponder" in self.job.tokens:
                    self.job.release.set()
        elif command == "quit":
            return False
        return True

    def clear_cache(self):
        self.mcts.clear_cache()
        engine = self.evaluator.engine
        if engine and engine.proc is not None:
            engine.send("ucinewgame")
            engine.send("isready")
            engine.read_until("readyok", 5)

    def close(self):
        self.supersede()
        self.pool.shutdown(wait=True, cancel_futures=True)
        if self.evaluator:
            self.evaluator.close()


def main():
    uci_print(f"{NAME} {VERSION} by {AUTHOR}")
    engine = UciEngine()
    try:
        for line in sys.stdin:
            try:
                if not engine.handle(line):
                    break
            except (ValueError, OSError) as error:
                uci_print(f"info string Command rejected: {error}")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
