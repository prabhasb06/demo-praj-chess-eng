"""Paired-opening UCI match harness with history, per-game resets and PGN/JSON results."""
import argparse
import json
import pathlib
import sys
import time
from contextlib import ExitStack
import chess
import chess.engine
import chess.pgn

OPENINGS = [
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6",
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4",
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7",
    "c2c4 e7e5 b1c3 g8f6 g2g3 d7d5",
    "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6",
]


def command(directory):
    return [sys.executable, str(pathlib.Path(directory).resolve() / "uci_engine.py")]


def match(engine_dir, opponent_dir, games, movetime, sf_depth, max_plies, output):
    rows, pgns = [], []
    with ExitStack() as stack:
        engine = stack.enter_context(chess.engine.SimpleEngine.popen_uci(command(engine_dir), timeout=5))
        rival_cmd = command(opponent_dir) if opponent_dir else str(pathlib.Path(engine_dir).resolve() / "stockfish.exe")
        rival = stack.enter_context(chess.engine.SimpleEngine.popen_uci(rival_cmd, timeout=5))
        if not opponent_dir:
            rival.configure({"Threads": 1, "Hash": 64})
        for game_number in range(games):
            board = chess.Board()
            for token in OPENINGS[(game_number // 2) % len(OPENINGS)].split():
                board.push_uci(token)
            initial = board.copy()
            white = game_number % 2 == 0
            game_id = object()
            reason, score = None, None
            times = []
            start = time.monotonic()
            while board.ply() < max_plies and not board.is_game_over(claim_draw=True):
                current = engine if board.turn == white else rival
                limit = (chess.engine.Limit(time=movetime / 1000)
                         if current is engine or opponent_dir else chess.engine.Limit(depth=sf_depth))
                tick = time.monotonic()
                try:
                    result = current.play(board, limit, game=game_id)
                    elapsed = (time.monotonic() - tick) * 1000
                    if result.move not in board.legal_moves:
                        raise ValueError("Illegal/null move in playable position")
                    if current is engine:
                        times.append(elapsed)
                    board.push(result.move)
                except (chess.engine.EngineError, TimeoutError, ValueError) as error:
                    score = 0.0 if current is engine else 1.0
                    reason = f"{'engine' if current is engine else 'opponent'} failure: {error}"
                    break
            if score is None:
                outcome = board.outcome(claim_draw=True)
                if outcome is None:
                    reason, score = "ply-cap (adjudicated draw; not a proven chess draw)", 0.5
                elif outcome.winner is None:
                    reason, score = outcome.termination.name, 0.5
                else:
                    reason, score = outcome.termination.name, float(outcome.winner == white)
            game = chess.pgn.Game.from_board(board)
            game.headers["White"] = "candidate" if white else "baseline"
            game.headers["Black"] = "baseline" if white else "candidate"
            game.headers["Result"] = ("1/2-1/2" if score == 0.5 else
                                     "1-0" if (score == 1) == white else "0-1")
            game.headers["Termination"] = reason
            game.headers["TimeControl"] = f"{movetime}ms per move"
            pgns.append(str(game))
            row = dict(game=game_number+1, candidate_white=white, opening=initial.fen(),
                       score=score, reason=reason, plies=board.ply(),
                       seconds=round(time.monotonic()-start, 2),
                       max_candidate_ms=round(max(times, default=0), 2))
            rows.append(row)
            print(json.dumps(row), flush=True)
            # Persist after every game so interrupted experiments retain evidence.
            report = dict(games=rows, movetime_ms=movetime, opponent=opponent_dir or f"Stockfish depth {sf_depth}",
                          wins=sum(r["score"] == 1 for r in rows),
                          draws=sum(r["score"] == 0.5 for r in rows),
                          losses=sum(r["score"] == 0 for r in rows),
                          score=sum(r["score"] for r in rows) / len(rows))
            pathlib.Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
            pathlib.Path(output).with_suffix(".pgn").write_text("\n\n".join(pgns), encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-dir", default=str(pathlib.Path(__file__).parent))
    ap.add_argument("--opponent-dir")
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--movetime", type=int, default=150)
    ap.add_argument("--sf-depth", type=int, default=2)
    ap.add_argument("--max-plies", type=int, default=240)
    ap.add_argument("--output", default="match_results.json")
    args = ap.parse_args()
    if args.games <= 0 or args.movetime <= 0 or args.max_plies <= 6:
        ap.error("games/movetime must be positive and max-plies greater than 6")
    print(json.dumps(match(args.engine_dir, args.opponent_dir, args.games, args.movetime,
                           args.sf_depth, args.max_plies, args.output)), flush=True)
