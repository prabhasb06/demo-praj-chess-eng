"""
benchmark.py — Measure the hybrid engine's playing strength.

Plays N games: Praj hybrid (python uci_engine.py) VS pure stockfish.exe
limited to a fixed search depth. Alternates colors. Self-refereed with
python-chess (mate / stalemate / insufficient material / 50-move /
threefold / ply cap = draw).

Usage:
    python benchmark.py --games 4 --movetime 300 --sf-depth 2 --max-plies 80
"""
import argparse
import chess
import os
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.abspath(__file__))


class UciProc:
    def __init__(self, argv, cwd, name):
        self.name = name
        self.p = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True)
        self.lines = []
        threading.Thread(target=self._reader, daemon=True).start()
        self.send("uci")
        self.wait(lambda l: l.startswith("uciok"), 15)
        self.send("isready")
        self.wait(lambda l: l.startswith("readyok"), 15)

    def _reader(self):
        for ln in self.p.stdout:
            self.lines.append(ln.rstrip())

    def send(self, cmd):
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def wait(self, pred, timeout):
        end = time.time() + timeout
        seen = 0
        while time.time() < end:
            fresh = self.lines[seen:]
            seen = len(self.lines)
            for ln in fresh:
                if pred(ln):
                    return True
            time.sleep(0.02)
        return False

    def bestmove(self, go_cmd, timeout=120.0):
        marker = len(self.lines)
        self.send(go_cmd)
        end = time.time() + timeout
        while time.time() < end:
            for ln in self.lines[marker:]:
                if ln.startswith("bestmove"):
                    tok = ln.split()[1]
                    return None if tok == "0000" else chess.Move.from_uci(tok)
            time.sleep(0.02)
        return None

    def quit(self):
        try:
            self.send("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def play_game(eng, sf, sf_depth, movetime_ms, max_plies, eng_white):
    board = chess.Board()
    plies = 0
    start_fen = board.fen()

    for p, is_eng in ((eng, eng_white), (sf, not eng_white)):
        p.send(f"position startpos")
    # engine gets full position command each move; stockfish likewise

    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        if board.turn == chess.WHITE:
            mover = eng if eng_white else sf
        else:
            mover = sf if eng_white else eng

        pos_cmd = f"position fen {board.fen()}"
        mover.send(pos_cmd)
        if mover is sf:
            mv = mover.bestmove(f"go depth {sf_depth}", timeout=60)
        else:
            mv = mover.bestmove(f"go movetime {movetime_ms}", timeout=60 + movetime_ms / 100)

        if mv is None or mv not in board.legal_moves:
            return ("illegal/timeout loss", board)

        board.push(mv)
        plies += 1

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return ("draw-plycap", board)
    if outcome.winner is None:
        return ("draw", board)
    winner_is_engine = (outcome.winner == chess.WHITE) == eng_white
    return ("engine-win" if winner_is_engine else "sf-win", board)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=4, help="total games (alternating colors)")
    ap.add_argument("--movetime", type=int, default=300, help="engine movetime ms")
    ap.add_argument("--sf-depth", type=int, default=2, help="pure Stockfish fixed depth")
    ap.add_argument("--max-plies", type=int, default=80, help="adjudicate draw after N plies")
    args = ap.parse_args()

    eng = UciProc([sys.executable, "uci_engine.py"], REPO, "Praj-Hybrid")
    sf_bin = os.path.join(REPO, "stockfish.exe")
    sf = UciProc([sf_bin], REPO, f"Stockfish-d{args.sf_depth}")

    tally = {"engine-win": 0, "sf-win": 0, "draw": 0}
    t0 = time.time()
    try:
        for g in range(args.games):
            eng_white = (g % 2 == 0)
            result, final_board = play_game(eng, sf, args.sf_depth,
                                            args.movetime, args.max_plies, eng_white)
            key = result if result in tally else ("draw" if result.startswith("draw") else result)
            if key == "illegal/timeout loss":
                key = "sf-win"
            tally[key] += 1
            print(f"game {g+1}: engine={'W' if eng_white else 'B'} -> {result} "
                  f"({final_board.fen().split()[0][:30]}...) "
                  f"[W {tally['engine-win']} / L {tally['sf-win']} / D {tally['draw']}]",
                  flush=True)
    finally:
        eng.quit()
        sf.quit()

    n = max(1, sum(tally.values()))
    score = (tally["engine-win"] * 1.0 + tally["draw"] * 0.5) / n
    print("=" * 50)
    print(f"RESULT vs Stockfish depth {args.sf_depth}: "
          f"W {tally['engine-win']} / L {tally['sf-win']} / D {tally['draw']}  "
          f"score {score:.0%}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
