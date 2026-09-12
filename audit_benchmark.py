"""Reproducible UCI quality/latency suite; scores are not an absolute Elo rating."""
import argparse
import json
import pathlib
import statistics
import sys
import time

import chess
import chess.engine

POSITIONS = [
    ("start", chess.STARTING_FEN),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("scholar", "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 3"),
    ("backrank", "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"),
    ("rook_underpromotion", "8/1P6/8/8/8/8/7p/5K1k w - - 0 1"),
    ("knight_underpromotion", "8/5P1k/8/6K1/8/8/8/8 w - - 0 1"),
    ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ("middlegame", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"),
]


def suite(directory, milliseconds):
    directory = pathlib.Path(directory).resolve()
    rows = []
    with chess.engine.SimpleEngine.popen_uci(
        [sys.executable, str(directory / "uci_engine.py")], timeout=10
    ) as engine, chess.engine.SimpleEngine.popen_uci(
        str(directory / "stockfish.exe"), timeout=10
    ) as judge:
        judge.configure({"Threads": 1, "Hash": 64})
        for name, fen in POSITIONS:
            board = chess.Board(fen)
            assert board.is_valid(), (name, board.status())
            judge.configure({"Clear Hash": None})
            reference = judge.analyse(board, chess.engine.Limit(depth=16))
            started = time.perf_counter()
            result = engine.play(board, chess.engine.Limit(time=milliseconds / 1000), game=object())
            elapsed = (time.perf_counter() - started) * 1000
            legal = result.move in board.legal_moves
            loss = None
            if legal:
                judge.configure({"Clear Hash": None})
                chosen = judge.analyse(board, chess.engine.Limit(depth=16), root_moves=[result.move])
                loss = max(0, reference["score"].pov(board.turn).score(mate_score=10000)
                           - chosen["score"].pov(board.turn).score(mate_score=10000))
            row = dict(name=name, move=str(result.move), reference=str(reference["pv"][0]),
                       legal=legal, elapsed_ms=round(elapsed, 2), cp_loss=loss)
            rows.append(row)
            print(json.dumps(row), flush=True)
    return dict(movetime_ms=milliseconds, positions=rows,
                mean_cp_loss=statistics.mean(r["cp_loss"] for r in rows if r["cp_loss"] is not None),
                max_elapsed_ms=max(r["elapsed_ms"] for r in rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-dir", default=str(pathlib.Path(__file__).parent))
    parser.add_argument("--movetime", type=int, default=150)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = suite(args.engine_dir, args.movetime)
    pathlib.Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
