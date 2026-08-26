# Praj MCTS Chess Engine — Bug Report

> **STATUS: RESOLVED** — see [Resolution Log](#resolution-log) at the bottom.
> All critical/medium items fixed and empirically verified (9/9 UCI protocol tests).

Audit date: 2026-08-25
Audited against: `main` branch (`5511003`), local clone `C:\Users\prabh\chess_engine`
Verification: static review of all modules + empirical runs
(`alpha_beta.py` self-test: 24/24 PASS · `tests.py`: 14/14 OK (6 skipped, no PyTorch) ·
`test_uci.py`: exposed missing bestmove · `debug.log`: confirmed GUI desync + historical crash)

---

## CRITICAL — break engine behaviour under real UCI GUIs

### BUG-1: `stop` command never answers `bestmove`
- **File:** `uci_engine.py:470-472` (handler), abort checks at `uci_engine.py:169` and `212`
- **Detail:** `stop` only sets `abort_event` / `mcts.abort_search`. The search worker's
  post-phase guards (`if abort_event.is_set() or _superseded(...): return`) then exit
  **without printing anything**. The UCI spec requires exactly one `bestmove` after every `stop`.
- **Effect:** GUIs that send `stop` hang waiting for the move; tournament managers forfeit
  the engine on time.
- **Also related:** race window at `uci_engine.py:135-137` — a `stop` arriving between job
  submit and worker start gets cleared before it is ever observed.
- **Fix sketch:** track an explicit `state["stop"]` flag set by the `stop` handler and
  cleared by each new `go`; change the worker's silent-abort condition to supersession-only,
  so a still-current aborted search falls through to move selection and emits `bestmove`
  (with a zero-visits fallback: AB best move or random legal move).

### BUG-2: `go depth N` / `go nodes N` / `go infinite` are ignored
- **File:** `parse_time_from_go`, `uci_engine.py:35-82`
- **Detail:** parser reads only `wtime/btime/winc/binc/movestogo/movetime`. Any other
  `go` form returns `None`, so the engine silently substitutes `params.max_think_time`.
- **Evidence:** `debug.log` shows the GUI repeatedly sending `go depth 20`; at the current
  default Difficulty 10 that means up to **30 s per move**, causing the GUI to give up and
  spam `position`+`go` (the desync pattern throughout the log).
- **Effect:** engine ignores analysis requests; appears frozen; `go infinite` illegally
  self-terminates (~30 s) with an unrequested bestmove.
- **Fix sketch:** parse `depth`, `nodes`, `infinite`. Map depth→bounded budget
  (e.g. `max(0.8, depth*0.25)` seconds, still capped by `max_think_time`);
  `infinite` → unlimited deadline + lift the simulation cap so the search runs until `stop`.

### BUG-3: MCTS output bypasses the stdout lock → corrupt UCI lines
- **File:** raw `print()` calls in `mcts.py:574` (periodic `info depth … pv …`),
  and warning prints at `mcts.py:166`, `225`, `269`, `385`
- **Detail:** `uci_engine.py` serialises stdout via `_stdout_lock`, but the search worker
  (which lives in mcts.py) writes directly. Two threads writing concurrently can interleave.
- **Effect:** interleaved/garbled protocol lines; random GUI parse failures.
- **Fix sketch:** expose a single shared lock (e.g. `STDOUT_LOCK` in `mcts.py`) plus a
  `_uci_out()` helper; have `uci_engine._uci_print` use the same lock.

---

## MEDIUM — races and resource leaks

### BUG-4: `setoption` mutates shared objects during a running search
- **File:** `uci_engine.py:356-372` (Difficulty), `uci_engine.py:378-397` (StockfishPath)
- **Detail:** unlike `go`/`position`/`ucinewgame`, these handlers do not invalidate the
  in-flight search first. Main thread closes/replaces the Stockfish process or changes
  `mcts.c_puct` while the worker is mid-search.
- **Effect:** exceptions inside the running search, stale evaluator references, rare crash.
- **Fix sketch:** supersede first (`abort_event.set(); state["current_search_id"] += 1`)
  before mutating.

### BUG-5: Stockfish pipe double-reader leak after a timeout
- **File:** `SubprocessStockfish.read_until`, `mcts.py:190-228`
- **Detail:** on timeout the spawned reader thread stays blocked in `readline()`. The next
  query starts a *second* reader on the same pipe; incoming lines split arbitrarily between
  the two consumers.
- **Effect:** lost/duplicated Stockfish responses → evaluation fallbacks, sync drift.
- **Fix sketch:** keep a reference to the reader thread; refuse to spawn a second one while
  the previous is alive (return empty → caller falls back to heuristic eval).

---

## LOW — correctness/polish

| ID | Bug | Location |
|----|-----|----------|
| BUG-6 | Quiescence recurs past its depth limit while in check (evasion branch skips the `depth_limit <= 0` early-out) | `alpha_beta.py:394-410` |
| BUG-7 | `stockfish_multipv` difficulty parameter never used — `multipv=3` hardcoded | `mcts.py:288`, `difficulty.py` table |
| BUG-8 | Mobility term counts only side-to-move (tempo bias) and regenerates `legal_moves` twice per leaf | `alpha_beta.py:270-278` |
| BUG-9 | Transposition cache keyed on full FEN incl. halfmove/fullmove counters → fewer hits than possible | `mcts.py:444` |
| BUG-10 | Failed `position` command silently keeps the previous board (GUI/engine desync) | `uci_engine.py:430-440` |
| BUG-11 | No repetition avoidance inside MCTS simulations (can walk into threefold within the tree) | `mcts.py:_simulate` |
| BUG-12 | `read_until` timeout leaves the SF process un-stopped; next query may attach mid-search output | `mcts.py:230-234` |

---

## Documentation mismatches (README.md vs live code)

| README says | Live code does |
|---|---|
| Clone URL `your-username/praj-chess-engine` | actual repo: `prabhasb06/demo-praj-chess-eng` |
| `Difficulty` default **7** | advertises `default 10` |
| `MoveVariety` default **true** | defaults to `false` |
| "Alpha-Beta tactical filter detects forced mates…" + AB Depth column | AB filter is dead code when the Stockfish evaluator runs (`use_ab` requires a non-Stockfish evaluator; main() always creates one). Mate play relies purely on MCTS+Stockfish. |

---

## Verified working (no action needed)

- Backpropagation sign convention & PUCT perspective (value stored from mover's viewpoint)
- Supersede-once-bestmove logic for rapid `go` storms
- FEN/startpos/moves parsing; promotion collapse handling incl. `_pick_promotion` fallback
- Clock allocation math (1/30 + increment, capped at 1/5 of remaining)
- Alpha-beta two-pass root scheme (null-window + verification re-search)
- Historical `stdin_reader` UnboundLocalError from old builds is fixed in current source
  (**note:** stale binaries `dist\uci_engine.exe` / `uci_engine.exe` predate this — rebuild)

## Recommended fix order

1. **BUG-1 + BUG-6** (stop handling) — protocol-critical
2. **BUG-2** (depth/nodes/infinite) — removes the 30 s/move GUI stalls seen in debug.log
3. **BUG-3** (shared stdout lock)
4. **BUG-4**, **BUG-5** (races/leaks)
5. Low items opportunistically

---

## Resolution Log

| Item | Status | Commit / Note |
|------|--------|---------------|
| BUG-1 stop → bestmove | FIXED + verified | `aa8aa74` (stop_requested fall-through) |
| BUG-2 go depth/nodes/infinite | FIXED + verified | `aa8aa74`; depth→bounded budget, infinite runs until stop |
| BUG-3 shared stdout lock | FIXED + verified | `aa8aa74` + `e495234` (flush restored; line buffering forced) |
| BUG-4 setoption race | FIXED + verified | supersede kept for StockfishPath; Difficulty applied lock-free mid-search |
| BUG-5 SF double-reader leak | FIXED | reader-thread guard (`aa8aa74`) |
| REG-A engine mute under GUIs | FIXED + verified | `e495234` — flush + line buffering (regression from aa8aa74) |
| REG-B setoption Difficulty NameError | FIXED + verified | `e495234` — removed undefined `is_infinite` reference |
| BUG-6 stop race pre-worker-start | FIXED + verified | `e495234` — pending stop honoured before flag clear |
| BUG-7 dead stockfish_multipv | FIXED | this commit — evaluator.multipv wired from difficulty params |
| BUG-8 mobility double enumeration | PARTIAL | this commit — single enumeration per leaf; single-side tempo bias kept intentionally (cheap eval) |
| BUG-9 TT keyed on clocks | FIXED | this commit — cache key now `board.epd()` |
| BUG-10 stale board on bad position | FIXED | this commit — revert to startpos on parse failure |
| BUG-11 repetition choices in tree | PARTIALLY ADDRESSED | scoring was already correct via `claim_draw=True`; this commit adds a **root repetition guard** (priors of moves that instantly recreate a threefold are crushed ×10⁻³, so draws only happen when nothing better exists). Full repetition avoidance remains future work |
| BUG-12 SF pipe unusable after timeout | FIXED | this commit — `stop` sent on timeout, blocked reader drains and exits |
| README mismatches | FIXED | this commit — URL, defaults, AB-filter wording, time-control section |

### Strength upgrades (hybrid architecture preserved — MCTS decides, Stockfish advises)

| Upgrade | Detail |
|---------|--------|
| Mate-distance-aware values | Forced mates now map to `±(1.0 … 0.92)` by distance-to-mate instead of flattening to ±1.0 — the engine prefers the *shortest* mate |
| Root repetition guard | See BUG-11 above |
| Stockfish Hash 256 MB | SF's transposition table persists across our repeated queries (default was 16 MB) — deeper advice per simulation at no protocol cost |

Verified post-upgrade: py_compile clean · tests 14/14 · UCI suite 9/9 ·
mate-in-1 found in 0.5 s (Scholar's + back-rank) · clock game honors budget
(2.4 s used of ~2.8 s allocated at 60+5).

### Follow-up round (same day)

| Item | Detail |
|------|--------|
| **Pondering support** | Full `go ponder` / `ponderhit` implementation: held bestmove delivered on hit (0.03 s), zero leaks when opponent deviates, stop-during-ponder still answers per spec. Un-pondered work is reused via the EPD TT + SF hash |
| **benchmark.py** | Self-contained strength measurement vs pure Stockfish at fixed depth; smoke run W1/D1 (75%) at depth 2 |
| Verification | py_compile · tests 14/14 · UCI suite 9/9 · ponder suite 7/7 · benchmark smoke |

Verification battery at resolution time: `py_compile` clean · `alpha_beta.py`
self-test 24/24 · `tests.py` 14/14 (6 skipped, no PyTorch) · UCI suite 9/9
(handshake over buffered pipe, setoption before/during search, instant-stop
race, mid-search stop, movetime/depth bounds, rapid storm, line integrity).
