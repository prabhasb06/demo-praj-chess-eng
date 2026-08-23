# Contributing to Praj MCTS Chess Engine

Thank you for your interest in contributing! Here's how to get started.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/praj-chess-engine.git
   cd praj-chess-engine
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the tests to make sure everything works:
   ```bash
   python -m unittest tests.py -v
   ```

## Making Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Run the test suite:
   ```bash
   python -m unittest tests.py -v
   python alpha_beta.py  # Built-in self-test
   ```
4. Commit with a descriptive message:
   ```bash
   git commit -m "Add: description of your change"
   ```

## Code Style

- Follow PEP 8 for Python code
- Use type hints where possible
- Add docstrings to all public functions and classes
- Keep functions focused and under 50 lines where practical
- Use descriptive variable names

## Pull Request Process

1. Update documentation if your change affects the public API
2. Add tests for new functionality
3. Ensure all existing tests pass
4. Submit a PR with a clear description of what and why

## Reporting Issues

When reporting bugs, please include:
- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Any error messages or logs

## Areas for Contribution

- Improving the neural network architecture
- Adding opening book support
- Endgame tablebase integration
- Performance optimizations in MCTS
- Better time management
- Cross-platform support improvements
- Documentation and examples
