# Praj MCTS Chess Engine 2.1

A Python UCI engine combining MCTS move selection with Stockfish policy/value advice. The September 2026 audit fixes protocol races, search timing, draw handling, underpromotions, caching and training bugs. See [AUDIT_REPORT.md](AUDIT_REPORT.md) for evidence and limitations.

## Run

Use Python 3.11 or later:

```powershell
python -m pip install -r requirements-engine.txt
python main.py
```

On Windows, add `dist/praj_mcts_2_1.exe` to your chess GUI. Keep `stockfish.exe` beside that executable. The old `dist/uci_engine.exe` is retained and is **not** the audited build. The new executable runs the Stockfish-assisted engine without needing Python or PyTorch installed.

On other platforms, put the matching native Stockfish executable at `stockfish.exe` beside the Python source (the filename is historical), or set the UCI `StockfishPath` option to its absolute path. Linux CI is configured, but this audit was executed on Windows.

## UCI options and limits

| Option | Default | Meaning |
|---|---|---|
| Difficulty | 10 | Parameter preset, range 1–10; **not a calibrated Elo rating** |
| MoveVariety | false | Enable temperature, exploration noise and configured blunder probabilities |
| Ponder | false | Standard GUI capability option; actual pondering starts only with `go ponder` |
| StockfishPath | stockfish.exe | Local evaluator executable |
| Clear Hash | button | Clear evaluation caches between searches |

Supports `position startpos/fen ... moves ...`, `movetime`, clock/increment allocation, `depth`, `nodes`, `mate`, `searchmoves`, `infinite`, `stop`, `ponderhit`, `ucinewgame`, `isready` and `quit`.

`depth` caps tree plies and advisory lookahead; it does not imply exhaustive alpha-beta coverage. `nodes` counts completed MCTS simulations (also the reported UCI nodes), not the internal nodes searched by the advisory Stockfish process. `mate N` bounds mate search to 2N−1 plies; finding a mate is not guaranteed. Finite searches can finish early. Completed ponder/infinite results are held until the appropriate release command. A new position/go supersedes old work, suppressing stale output. Invalid positions are rejected transactionally with an `info string` diagnostic.

Deadlines reach Stockfish; stopping drains its final `bestmove` before the next query. A backend that refuses to stop is terminated. If unavailable, static evaluation preserves legal play, with substantially reduced and uncalibrated strength. Set a valid `StockfishPath` to restart the backend.

## Search and model format

The move policy has **4,480 actions**: the original 4,096 square-pair actions plus 384 distinct knight/bishop/rook promotion actions for both colors. Queen promotions retain their original indices. All training/inference paths use the same encoding. The board encoder remains 18 planes.

`neural_net.load_checkpoint` migrates old 4,096-action checkpoints in memory by copying each matching queen-promotion row into the new underpromotion rows. Original files remain untouched. Further training is needed to learn distinct promotion preferences.

MCTS has an 8,192-entry sparse evaluation cache and a 65,536-node tree cap. Cache keys preserve the halfmove clock, reversible history and evaluator configuration. The standalone alpha-beta tactical analyzer is available in `alpha_beta.py`; the default UCI path uses Stockfish-guided MCTS.

## Tests

Install neural dependencies to run all tests:

```powershell
python -m pip install -r requirements.txt
python -m unittest tests test_audit -v
python alpha_beta.py
```

The audit used an isolated `.venv-audit` with CPU PyTorch. Without PyTorch, neural tests explicitly skip. `python test_uci.py` runs the protocol suite. To test a packaged build:

```powershell
$env:PRAJ_ENGINE_EXE = "$PWD\dist\praj_mcts_2_1.exe"
python -m unittest test_audit.AuditProtocol -v
```

## Benchmarks

```powershell
python audit_benchmark.py --movetime 150 --output quality.json
python benchmark.py --opponent-dir C:\path\to\baseline --games 12 --movetime 150 --output match.json
python benchmark.py --games 12 --movetime 150 --sf-depth 2 --output stockfish_match.json
```

Matches use paired openings with reversed colors, preserve move history and save PGN/JSON. Ply-cap draws are explicitly marked. Short matches and a shallow Stockfish opponent do not establish an absolute Elo.

## Training

`bash train_pipeline.sh` generates self-play examples, trains over complete replay-buffer epochs, evaluates paired openings and promotes candidates at the configured score threshold. Its default workload is large; reduce the constants for a smoke run. The 55% promotion threshold is a heuristic, not a statistical significance test. Full-scale training and GPU operation were not exercised in this audit.

The C++ file is a nonfunctional reference sketch, excluded from CMake by default. It is not an alternative production engine.

Project source is MIT-licensed. Stockfish is a separate dependency with its own license.
