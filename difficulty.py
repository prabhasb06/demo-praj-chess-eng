"""
Difficulty controller for a hybrid MCTS + Alpha-Beta chess engine.

Maps difficulty levels (1-10) to concrete search parameters that govern
Alpha-Beta depth, MCTS simulation count, move selection temperature,
blunder probability, Stockfish evaluator settings, and exploration noise.

Usage:
    controller = DifficultyController(level=5)
    params = controller.get_params()

    # After MCTS, select a move with appropriate variety:
    action = select_move_with_variety(visit_counts, params, board)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

import chess
import numpy as np

from board_encoder import encode_move


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DifficultyParams:
    """All search parameters controlled by difficulty level.

    Attributes:
        level:            Difficulty level (1 = easiest, 10 = strongest).
        ab_depth:         Alpha-Beta filter depth (1-4).
        num_simulations:  MCTS simulation count per move.
        temperature:      Move-selection temperature (0.0 = deterministic).
        top_k:            Select from the top-K moves by MCTS visit count.
        blunder_pct:      Probability of choosing a random legal move (0.0-0.3).
        stockfish_depth:  Depth for Stockfish evaluator queries.
        stockfish_multipv: MultiPV for Stockfish queries.
        c_puct:           MCTS exploration constant (UCB).
        max_think_time:   Maximum wall-clock time in seconds for the search.
        add_noise:        Whether to inject Dirichlet noise at the MCTS root.
        noise_epsilon:    Dirichlet noise weight (0.0-0.5).
        noise_alpha:      Dirichlet alpha parameter.
    """

    level: int
    ab_depth: int
    num_simulations: int
    temperature: float
    top_k: int
    blunder_pct: float
    stockfish_depth: int
    stockfish_multipv: int
    c_puct: float
    max_think_time: float
    add_noise: bool
    noise_epsilon: float
    noise_alpha: float


# ---------------------------------------------------------------------------
# Parameter table  (indexed by level 1-10)
# ---------------------------------------------------------------------------

_PARAM_TABLE: List[DifficultyParams] = [
    # Level 1  –  ELO ~600
    DifficultyParams(
        level=1, ab_depth=1, num_simulations=20,
        temperature=2.0, top_k=5, blunder_pct=0.30,
        stockfish_depth=3, stockfish_multipv=2,
        c_puct=2.5, max_think_time=1.0,
        add_noise=True, noise_epsilon=0.50, noise_alpha=1.0,
    ),
    # Level 2  –  ELO ~800
    DifficultyParams(
        level=2, ab_depth=1, num_simulations=40,
        temperature=1.5, top_k=4, blunder_pct=0.20,
        stockfish_depth=3, stockfish_multipv=2,
        c_puct=2.2, max_think_time=1.5,
        add_noise=True, noise_epsilon=0.40, noise_alpha=0.8,
    ),
    # Level 3  –  ELO ~1000
    DifficultyParams(
        level=3, ab_depth=2, num_simulations=80,
        temperature=1.2, top_k=4, blunder_pct=0.12,
        stockfish_depth=4, stockfish_multipv=3,
        c_puct=2.0, max_think_time=2.0,
        add_noise=True, noise_epsilon=0.35, noise_alpha=0.6,
    ),
    # Level 4  –  ELO ~1200
    DifficultyParams(
        level=4, ab_depth=2, num_simulations=150,
        temperature=1.0, top_k=3, blunder_pct=0.07,
        stockfish_depth=4, stockfish_multipv=3,
        c_puct=1.8, max_think_time=3.0,
        add_noise=True, noise_epsilon=0.30, noise_alpha=0.5,
    ),
    # Level 5  –  ELO ~1400
    DifficultyParams(
        level=5, ab_depth=3, num_simulations=250,
        temperature=0.8, top_k=3, blunder_pct=0.03,
        stockfish_depth=5, stockfish_multipv=3,
        c_puct=1.6, max_think_time=4.0,
        add_noise=True, noise_epsilon=0.25, noise_alpha=0.4,
    ),
    # Level 6  –  ELO ~1600
    DifficultyParams(
        level=6, ab_depth=3, num_simulations=400,
        temperature=0.5, top_k=2, blunder_pct=0.01,
        stockfish_depth=5, stockfish_multipv=4,
        c_puct=1.5, max_think_time=5.0,
        add_noise=True, noise_epsilon=0.20, noise_alpha=0.3,
    ),
    # Level 7  –  ELO ~1800
    DifficultyParams(
        level=7, ab_depth=3, num_simulations=600,
        temperature=0.3, top_k=2, blunder_pct=0.00,
        stockfish_depth=6, stockfish_multipv=4,
        c_puct=1.4, max_think_time=7.0,
        add_noise=True, noise_epsilon=0.15, noise_alpha=0.3,
    ),
    # Level 8  –  ELO ~2000
    DifficultyParams(
        level=8, ab_depth=4, num_simulations=800,
        temperature=0.15, top_k=2, blunder_pct=0.00,
        stockfish_depth=7, stockfish_multipv=5,
        c_puct=1.3, max_think_time=10.0,
        add_noise=True, noise_epsilon=0.10, noise_alpha=0.3,
    ),
    # Level 9  –  ELO ~2200
    DifficultyParams(
        level=9, ab_depth=4, num_simulations=1200,
        temperature=0.05, top_k=1, blunder_pct=0.00,
        stockfish_depth=8, stockfish_multipv=5,
        c_puct=1.2, max_think_time=15.0,
        add_noise=False, noise_epsilon=0.00, noise_alpha=0.3,
    ),
    # Level 10  –  ELO ~2400
    DifficultyParams(
        level=10, ab_depth=4, num_simulations=2000,
        temperature=0.0, top_k=1, blunder_pct=0.00,
        stockfish_depth=10, stockfish_multipv=5,
        c_puct=1.0, max_think_time=30.0,
        add_noise=False, noise_epsilon=0.00, noise_alpha=0.3,
    ),
]

# ELO boundaries aligned with the parameter table.
_ELO_BOUNDARIES: List[int] = [600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400]


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class DifficultyController:
    """Maps difficulty levels (1-10) to search parameters.

    Examples:
        >>> ctrl = DifficultyController(level=3)
        >>> p = ctrl.get_params()
        >>> p.ab_depth
        2
        >>> ctrl.set_level(10)
        >>> ctrl.get_params().num_simulations
        2000
    """

    def __init__(self, level: int = 7) -> None:
        """Initialise with a clamped difficulty level.

        Args:
            level: Desired difficulty (1-10). Values outside this range are
                   clamped automatically.
        """
        self.level: int = max(1, min(10, level))

    def set_level(self, level: int) -> None:
        """Update the difficulty level (clamped to 1-10).

        Args:
            level: New difficulty level.
        """
        self.level = max(1, min(10, level))

    def get_params(self) -> DifficultyParams:
        """Return the :class:`DifficultyParams` for the current level.

        Returns:
            A frozen snapshot of parameters corresponding to ``self.level``.
        """
        return _PARAM_TABLE[self.level - 1]

    @staticmethod
    def from_elo(elo: int) -> int:
        """Convert an ELO rating to the nearest difficulty level.

        The mapping uses the midpoints between consecutive ELO anchors in the
        parameter table.  Ratings below 600 map to level 1; ratings above 2400
        map to level 10.

        Args:
            elo: An integer ELO rating.

        Returns:
            An integer difficulty level in the range [1, 10].
        """
        if elo <= _ELO_BOUNDARIES[0]:
            return 1
        if elo >= _ELO_BOUNDARIES[-1]:
            return 10

        # Choose based on midpoints between consecutive anchors.
        for idx in range(len(_ELO_BOUNDARIES) - 1):
            midpoint = (_ELO_BOUNDARIES[idx] + _ELO_BOUNDARIES[idx + 1]) // 2
            if elo <= midpoint:
                return idx + 1
        return len(_ELO_BOUNDARIES)


# ---------------------------------------------------------------------------
# Move selection with variety
# ---------------------------------------------------------------------------

def select_move_with_variety(
    visit_counts: np.ndarray,
    params: DifficultyParams,
    board: chess.Board,
) -> int:
    """Select a move index from MCTS visit counts, respecting difficulty params.

    The selection pipeline:
      1. **Blunder check** – with probability ``params.blunder_pct`` a
         uniformly random *legal* move index is returned immediately.
      2. **Top-K filtering** – the ``params.top_k`` actions with the highest
         visit counts are isolated.
      3. **Temperature sampling** –
         * If ``temperature == 0.0`` the best move among the top-K is chosen
           (ties broken randomly).
         * If ``temperature > 0.0`` a softmax distribution with the given
           temperature is constructed over the top-K visit counts and a move
           is sampled from it.

    Args:
        visit_counts: A numpy array of shape ``(4096,)`` containing the
            visit count for each possible action in the MCTS tree.
        params: The :class:`DifficultyParams` governing selection behaviour.
        board: The current :class:`chess.Board` position (used for the legal
            move list when a blunder is triggered).

    Returns:
        The integer action index of the selected move.

    Raises:
        ValueError: If no legal moves are available on the board.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise ValueError("No legal moves available on the board.")

    # ------------------------------------------------------------------
    # 1. Blunder roll – return a random legal move index
    # ------------------------------------------------------------------
    if params.blunder_pct > 0.0 and random.random() < params.blunder_pct:
        move = random.choice(legal_moves)
        return encode_move(move)

    # ------------------------------------------------------------------
    # 2. Top-K by visit count
    # ------------------------------------------------------------------
    # Work with a copy so we don't mutate the caller's array.
    counts = visit_counts.copy().astype(np.float64)

    # Identify the top-K action indices (descending order of visits).
    k = min(params.top_k, int(np.count_nonzero(counts)))
    if k == 0:
        # Fallback: no visits recorded – pick a random legal move.
        move = random.choice(legal_moves)
        return encode_move(move)

    top_k_indices = np.argpartition(counts, -k)[-k:]
    top_k_indices = top_k_indices[np.argsort(counts[top_k_indices])[::-1]]
    top_k_counts = counts[top_k_indices]

    # ------------------------------------------------------------------
    # 3. Temperature-based selection
    # ------------------------------------------------------------------
    if params.temperature == 0.0:
        # Deterministic: pick the best (ties broken randomly).
        max_count = top_k_counts[0]
        best_indices = top_k_indices[top_k_counts == max_count]
        return int(np.random.choice(best_indices))

    # Apply softmax with temperature.
    logits = np.log(top_k_counts) / params.temperature
    # Numerical stability: subtract the max before exp.
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)

    chosen = np.random.choice(top_k_indices, p=probs)
    return int(chosen)


# ---------------------------------------------------------------------------
# CLI – pretty-print the full parameter table
# ---------------------------------------------------------------------------

def _print_params_table() -> None:
    """Print a formatted table of all 10 difficulty levels."""
    header = (
        f"{'Lvl':>3}  {'ELO':>4}  {'AB':>2}  {'Sims':>5}  {'Temp':>5}  "
        f"{'TopK':>4}  {'Blndr%':>6}  {'SFd':>3}  {'MPV':>3}  "
        f"{'cPUCT':>5}  {'Time':>5}  {'Noise':>5}  {'eps':>5}  {'alpha':>5}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for p in _PARAM_TABLE:
        elo = _ELO_BOUNDARIES[p.level - 1]
        print(
            f"{p.level:>3}  {elo:>4}  {p.ab_depth:>2}  {p.num_simulations:>5}  "
            f"{p.temperature:>5.2f}  {p.top_k:>4}  {p.blunder_pct:>6.2f}  "
            f"{p.stockfish_depth:>3}  {p.stockfish_multipv:>3}  "
            f"{p.c_puct:>5.1f}  {p.max_think_time:>5.1f}  "
            f"{'Y' if p.add_noise else 'N':>5}  "
            f"{p.noise_epsilon:>5.2f}  {p.noise_alpha:>5.1f}"
        )
    print(sep)


if __name__ == "__main__":
    _print_params_table()
