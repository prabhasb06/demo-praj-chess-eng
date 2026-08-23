# Praj MCTS Chess Engine

A hybrid chess engine combining **Monte Carlo Tree Search (MCTS)** with **Alpha-Beta tactical filtering** and an optional **neural network evaluator** (Squeeze-and-Excitation ResNet). Supports configurable difficulty levels (ELO ~600–2400) and UCI protocol for use with any chess GUI.

## Features

- **Hybrid Search**: Alpha-Beta tactical filter detects forced mates and large advantages before MCTS strategic search
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
git clone https://github.com/your-username/praj-chess-engine.git
cd praj-chess-engine
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
| `Difficulty` | spin | 7 | 1–10 | Engine strength (1=~600 ELO, 10=~2400 ELO) |
| `MoveVariety` | check | true | — | Enable move variety via temperature/blunder simulation |
| `StockfishPath` | string | stockfish.exe | — | Path to Stockfish binary for hybrid evaluation |

## Difficulty Levels

| Level | ~ELO | Simulations | AB Depth | Temperature | Blunder% |
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
