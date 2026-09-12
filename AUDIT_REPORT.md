# Chess engine audit — 12 September 2026

## Result

A substantially stronger, tested **release candidate** is available as `dist/praj_mcts_2_1.exe`. Keep the existing `dist/stockfish.exe` beside it. No changes were committed or pushed, and the previous executable was preserved.

Baseline: commit `cd7577803385c87f3c218349b4cdbcf4090ae526`, clean working tree at audit start. A runnable original-source snapshot is at `C:\Users\prabh\chess_engine_audit_baseline_20260912`.

## Verified fixes

| Area | Finding and correction |
|---|---|
| Search cancellation | Shared abort flags could erase a pending stop. Each search now owns immutable settings and its own stop/release events. |
| Races and stale output | Position/configuration changes could race evaluation and completion. One worker owns evaluator/cache mutations; completion is locked and superseded work cannot emit moves. |
| Ponder/infinite | Completed or terminal results could leak early, and stopping a held ponder result could go unanswered. All completion paths now obey release/stop. |
| Time controls | A 100 ms allocation floor could exceed the remaining clock; root/leaf Stockfish queries ignored deadlines. Monotonic deadlines, cancellation polling and bounded stop handling now reach the backend. |
| Process failures | Per-query readers could leave stale output after timeout. A lifetime reader drains through bestmove; crashed/hung processes are reaped. Fault injection also exposed and fixed Windows pipe-close errors. |
| Position legality | `Board.push` accepted syntactically valid illegal moves. Parsing now validates every move and FEN and commits the board only after complete success. |
| Draws | Claimable draws were treated as mandatory endings, returning null moves in playable positions. Claims are optional in search; root legal moves remain available. Quiescence now recognizes terminal draws. |
| Promotions | All four promotions shared one action. All promotions now round-trip distinctly through UCI, search, neural inference and training. |
| Root strength | Historical shallow MultiPV moves retained misleading scores. Only current depth/rank advice is retained. Root advice receives limited virtual weight so a tiny sample cannot casually overturn it. |
| Evaluation | Mate scores are separated from saturated material values; verified immediate mates take priority. Incomplete deadline-interrupted simulations do not override completed evidence. |
| Cache correctness | EPD-only keys conflated halfmove/history and difficulty settings; noise could mutate cached policy arrays. Context-aware keys and isolated sparse arrays fix both. |
| Memory/performance | Dense cache could allocate about 512 MiB for policy arrays alone, and infinite trees had no bound. Cache is sparse/bounded, tree size is capped, parent cycles are removed, and expansion iterates legal moves. |
| Alpha-beta ranking | A move proven far worse could regain a high rank through a static fallback score. Fallback estimates are now capped by their search bound; checked horizon positions use legal evasions. |
| Numerical stability | Tiny temperatures could overflow visit powers; difficulty sampling used raw-count softmax. Log-space sampling now yields finite distributions. |
| Training | Promotion shell comparison included all logs, so PROMOTE never matched; script imports depended on caller CWD; an epoch trained only one minibatch. These are fixed, with paired evaluation openings and checkpoint migration coverage. |
| Release/testing | Replaced the misleading UCI smoke test with assertions, added fault injection and CI, and excluded the invalid C++ sketch from default builds. |

## Measurements

Two 12-game matches against the preserved original each finished **12 wins, 0 draws, 0 losses** at 150 ms requested per move, using six paired openings. All 24 games ended by checkmate; no illegal moves, timeout forfeits or ply-cap adjudications occurred. The final match is in [match_results_final.json](match_results_final.json) and [match_results_final.pgn](match_results_final.pgn); the earlier experiment is also retained.

The final tree-memory cap was added after these strength runs and verified separately; no further strength parameters were changed.

On the fixed eight-position diagnostic suite:

| Measure | Original | Revised |
|---|---:|---:|
| Mean estimated centipawn loss | 14.25 | 2.50 |
| Maximum observed response time | 218.63 ms | 159.90 ms |
| Starting-position choice | Nh3 | e4 |

See [baseline_quality.json](baseline_quality.json) and [after_quality.json](after_quality.json). The reference is the same local Stockfish at depth 16. Scores from separate finite-depth queries can differ even for the same move; this small suite is diagnostic, not an exact quality or Elo measurement. Latency includes protocol/setup overhead, and this was a normal Windows desktop rather than an isolated benchmark machine.

Validation: **55 tests passed** across the original/regression suites with CPU PyTorch enabled; the standalone alpha-beta self-test passed **24/24**. The packaged executable passed the **14 UCI protocol tests**. Backend crash, refusal-to-stop, stale output, promotion encoding, legal random positions, memory limits, checkpoint migration and a real gradient update are covered. Build: Python 3.13.14, PyInstaller 6.20.0, CPU PyTorch 2.14.0.

## Release limits

This is a release candidate, not proof of zero bugs. Absolute Elo remains unmeasured; the previous 600–2400 labels were parameter guesses and are no longer presented as ratings. Longer tournaments against multiple independently rated engines, broader endgame suites and extended GUI soak tests are still needed.

UCI `nodes` measures MCTS simulations, not advisory Stockfish nodes. `depth` bounds selective search rather than promising exhaustive depth coverage. Pondering may hold a completed result; it does not reuse an entire search tree. Without Stockfish, the fallback remains legal but much weaker.

The Windows engine and CPU training components were exercised locally. Linux CI is added but has not run remotely; GPU training and a full training campaign were not run. The compiled executable excludes PyTorch; neural training remains a source/Python workflow. The C++ sketch is deliberately not production-ready.

Protocol behavior was checked against the [UCI specification](https://backscattering.de/chess/uci/), and the independent match driver uses [python-chess engine APIs](https://python-chess.readthedocs.io/en/latest/engine.html).
