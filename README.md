# Praj MCTS Chess Engine

A hybrid chess engine combining **Monte Carlo Tree Search (MCTS)** with **Alpha-Beta tactical filtering** and an optional **neural network evaluator** (Squeeze-and-Excitation ResNet). Supports configurable difficulty levels (ELO ~600–2400) and UCI protocol for use with any chess GUI.

## Features

- **Hybrid Search**: Alpha-Beta tactical filter (mates, candidate seeding) for non-Stockfish evaluators and lower levels; the default Stockfish-guided MCTS handles tactics via deep Stockfish evaluations
- **Neural Network**: SE-ResNet architecture with policy, value, and moves-left heads
- **Stockfish Integration**: Optional Stockfish-guided MCTS evaluation for strong play
- **10 Difficulty Levels**: Configurable ELO from ~600 to ~2400 with blunder simulation, temperature scheduling, and search depth scaling
- **UCI Protocol**: Compatible with En Croissant, Arena, CuteChess, and other UCI GUIs
- **Self-Play Training**: AlphaZero-style reinforcement learning pipeline

## Architecture

```
┌─────────────────────────────────────────┐
│              UCI Interface              │
│            (uci_engine.py)              │
├──────────┬──────────┬───────────────────┤
│ Alpha-   │  MCTS    │   Difficulty      │
│ Beta     │ Search   │   Controller      │
│ Filter   │(mcts.py) │ (difficulty.py)   │
│(alpha_   │          │                   │
│ beta.py) │          │                   │
├──────────┴──────────┴───────────────────┤
│         Evaluator Backend               │
│  ┌─────────────┐  ┌──────────────────┐  │
│  │  Neural Net  │  │    Stockfish     │  │
│  │ (SE-ResNet)  │  │   (subprocess)   │  │
│  └─────────────┘  └──────────────────┘  │
├─────────────────────────────────────────┤
│       Board Encoder (18-plane)          │
│         (board_encoder.py)              │
└─────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/prabhasb06/demo-praj-chess-eng.git
cd demo-praj-chess-eng
pip install -r requirements.txt
```

### Running the Engine

```bash
python main.py
```

Or on Windows:
```bash
run_engine.bat
```

### Using with a Chess GUI

1. Open your preferred UCI chess GUI (e.g., [En Croissant](https://encroissant.org/), [Arena](http://www.playwitharena.de/), [CuteChess](https://cutechess.com/))
2. Add a new engine and point it to `main.py` (or `python main.py`)
3. The engine will respond to standard UCI commands

### UCI Options

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| `Difficulty` | spin | 10 | 1–10 | Engine strength (1=~600 ELO, 10=~2400 ELO) |
| `MoveVariety` | check | false | — | Enable move variety via temperature/blunder simulation |
| `StockfishPath` | string | stockfish.exe | — | Path to Stockfish binary for hybrid evaluation |

## Difficulty Levels

| Level | ~ELO | Simulations | AB Depth* | Temperature | Blunder% |
|-------|------|-------------|----------|-------------|----------|
| 1 | 600 | 20 | 1 | 2.00 | 30% |
| 2 | 800 | 40 | 1 | 1.50 | 20% |
| 3 | 1000 | 80 | 2 | 1.20 | 12% |
| 4 | 1200 | 150 | 2 | 1.00 | 7% |
| 5 | 1400 | 250 | 3 | 0.80 | 3% |
| 6 | 1600 | 400 | 3 | 0.50 | 1% |
| 7 | 1800 | 600 | 3 | 0.30 | 0% |
| 8 | 2000 | 800 | 4 | 0.15 | 0% |
| 9 | 2200 | 1200 | 4 | 0.05 | 0% |
| 10 | 2400 | 2000 | 4 | 0.00 | 0% |

\* AB depth applies when the Alpha-Beta filter is active (non-Stockfish evaluator).
With the default Stockfish-guided evaluator, search strength is driven by
simulation count, Stockfish depth, and MultiPV instead.

## Time Controls

The engine supports `go movetime`, `go wtime/btime/winc/binc [+ movestogo]`,
`go depth N` (mapped to a bounded think time) and `go infinite` (search until
`stop`). Every `go`/`stop` pair is answered with exactly one `bestmove`.

**Pondering** is supported (`go ponder` / `ponderhit`) — the engine thinks on
the opponent's clock. Un-pondered thinking is never wasted: evaluations are
cached (EPD-keyed transposition table + persistent Stockfish hash) and reused
by the next search.

## Measuring Strength

```bash
python benchmark.py --games 4 --movetime 300 --sf-depth 2 --max-plies 80
```

Plays alternating-color games of the hybrid engine against a *pure*
`stockfish.exe` limited to a fixed depth, self-refereed (mate, draws,
50-move, threefold, ply-cap adjudication), and prints a W/L/D summary.

## Training (Self-Play)

The engine supports AlphaZero-style self-play training:

```bash
bash train_pipeline.sh
```

This will:
1. Generate self-play games using the current best model
2. Train a candidate model on the generated data
3. Evaluate the candidate against the current best
4. Promote the candidate if it wins ≥55% of evaluation games

## Running Tests

```bash
python -m unittest tests.py -v
python alpha_beta.py  # Built-in self-test
```

## Project Structure

| File | Description |
|------|-------------|
| `main.py` | Entry point |
| `uci_engine.py` | UCI protocol handler |
| `mcts.py` | Monte Carlo Tree Search |
| `alpha_beta.py` | Alpha-Beta minimax with quiescence search |
| `board_encoder.py` | Board state → 18-plane tensor encoding |
| `neural_net.py` | SE-ResNet neural network |
| `evaluator.py` | Model evaluation & matchmaking |
| `difficulty.py` | Difficulty level parameter mapping |
| `self_play.py` | Self-play data generation & training |
| `train_pipeline.sh` | Full training pipeline script |
| `uci_engine.cpp` | C++ engine prototype (non-functional sketch) |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
