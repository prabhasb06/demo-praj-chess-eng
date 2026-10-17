"""Integration checks against a fully extracted ready-to-run Windows download."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import chess
from prepare_release import sha256
from test_audit import Session


@unittest.skipUnless(os.environ.get("PRAJ_ENGINE_EXE"), "Extracted release executable required")
class BundledEngine(unittest.TestCase):
    def test_assistance_is_automatic_from_an_unrelated_directory(self):
        self.assertNotIn("PRAJ_STOCKFISH_PATH", os.environ,
                         "This check must not manually configure the evaluator")
        with tempfile.TemporaryDirectory() as folder:
            session = Session(cwd=folder)
            try:
                session.send("position startpos")
                session.send("go movetime 500")
                line, seen = session.until("bestmove", timeout=10)
                self.assertIn("info string Evaluation: Stockfish-assisted MCTS", seen)
                self.assertFalse(any("fallback" in item.lower() or "failed" in item.lower() for item in seen))
                self.assertIn(chess.Move.from_uci(line.split()[1]), chess.Board().legal_moves)
            finally:
                session.close()

    def test_bundled_binary_matches_recorded_official_origin(self):
        root = Path(os.environ["PRAJ_ENGINE_EXE"]).resolve().parent
        origin = json.loads((root / "licenses" / "Stockfish" / "ORIGIN.json").read_text())
        self.assertEqual(origin["tag"], "sf_18")
        self.assertEqual(sha256(root / "engines" / "stockfish.exe"), origin["binary_sha256"])
        self.assertEqual(len(origin["embedded_networks"]), 2)


if __name__ == "__main__":
    unittest.main()
