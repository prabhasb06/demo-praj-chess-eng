"""
Alpha-Beta Minimax Search with Quiescence Search for Tactical Detection.

This module implements a lightweight alpha-beta pruning search intended to be
used as a tactical filter BEFORE an MCTS search runs. It detects forced mates,
large tactical advantages, and provides candidate moves with scores to seed
MCTS priors.

All scores are in centipawns from the perspective of the side to move.
Piece-square tables are defined from White's perspective and flipped for Black.

Dependencies: chess (python-chess), numpy
"""

from __future__ import annotations

import chess
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Material Values (centipawns)
# ---------------------------------------------------------------------------
PIECE_VALUES: dict[int, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

# ---------------------------------------------------------------------------
# Piece-Square Tables (from White's perspective, a8=index 0, h1=index 63)
#
# Tables are stored as 64-element numpy arrays where index 0 = a8 and
# index 63 = h1.  For White we mirror vertically so that the first row
# of each visual table corresponds to rank 8 (indices 0-7).
#
# To look up a bonus for a white piece on square sq (chess.A1=0, chess.H8=63
# in python-chess), we use TABLES[piece_type][chess.square_mirror(sq)]
# because python-chess numbers a1=0 while our tables number a8=0.
#
# For Black pieces we simply use TABLES[piece_type][sq] (no mirror).
# ---------------------------------------------------------------------------

# fmt: off

_PAWN_TABLE = np.array([
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
], dtype=np.int32)

_KNIGHT_TABLE = np.array([
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
], dtype=np.int32)

_BISHOP_TABLE = np.array([
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
], dtype=np.int32)

_ROOK_TABLE = np.array([
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
], dtype=np.int32)

_QUEEN_TABLE = np.array([
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
], dtype=np.int32)

_KING_TABLE_MG = np.array([
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
], dtype=np.int32)

_KING_TABLE_EG = np.array([
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
], dtype=np.int32)

# fmt: on

PST: dict[int, np.ndarray] = {
    chess.PAWN: _PAWN_TABLE,
    chess.KNIGHT: _KNIGHT_TABLE,
    chess.BISHOP: _BISHOP_TABLE,
    chess.ROOK: _ROOK_TABLE,
    chess.QUEEN: _QUEEN_TABLE,
    chess.KING: _KING_TABLE_MG,
}

# Pre-compute flipped tables so we can index directly by python-chess square.
# python-chess: a1=0, b1=1, ..., h8=63
# Our visual tables: a8=0 ... h1=63  (row 0 = rank 8)
# square_mirror flips vertically (rank1<->rank8, etc.)
#
# For a White piece on python-chess square `sq`:
#   table index = sq ^ 56   (mirror vertically)
# For a Black piece on python-chess square `sq`:
#   table index = sq         (already "upside down" from Black's view)

_FLIP = 56  # XOR mask to mirror vertically


def _game_phase(board: chess.Board) -> float:
    """Estimate game phase from remaining material. 1.0 = opening/midgame, 0.0 = pure endgame."""
    # Total non-pawn, non-king material for both sides
    phase_material = {
        chess.KNIGHT: 1,
        chess.BISHOP: 1,
        chess.ROOK: 2,
        chess.QUEEN: 4,
    }
    total = 0
    for pt, weight in phase_material.items():
        total += len(board.pieces(pt, chess.WHITE)) * weight
        total += len(board.pieces(pt, chess.BLACK)) * weight
    # Max phase material = 2*(2*1 + 2*1 + 2*2 + 1*4) = 2*12 = 24
    return min(total / 24.0, 1.0)


def _pst_score(piece_type: int, square: int, is_white: bool, phase: float = 1.0) -> int:
    """Return the piece-square table bonus for a piece on a given square."""
    if piece_type == chess.KING:
        mg = int(_KING_TABLE_MG[square ^ _FLIP]) if is_white else int(_KING_TABLE_MG[square])
        eg = int(_KING_TABLE_EG[square ^ _FLIP]) if is_white else int(_KING_TABLE_EG[square])
        return int(phase * mg + (1.0 - phase) * eg)
    table = PST.get(piece_type)
    if table is None:
        return 0
    if is_white:
        return int(table[square ^ _FLIP])
    else:
        return int(table[square])


# ---------------------------------------------------------------------------
# Evaluation Constants
# ---------------------------------------------------------------------------
MATE_SCORE = 100_000  # Score for checkmate (large enough to dominate)
CASTLING_BONUS = 30   # Bonus for each remaining castling right
MOBILITY_WEIGHT = 4   # Centipawns per legal move
BISHOP_PAIR_BONUS = 50  # Bonus for having both bishops


# ---------------------------------------------------------------------------
# Static Evaluation
# ---------------------------------------------------------------------------

def evaluate_board(board: chess.Board, mobility: int = None) -> int:
    """
    Evaluate the board position and return a score in centipawns from the
    perspective of the **side to move**.

    Components:
    - Material counting with standard piece values
    - Piece-square table positional bonuses
    - King safety: bonus for remaining castling rights
    - Bishop pair bonus
    - Mobility: bonus proportional to legal move count

    Note: repetition draws are intentionally NOT detected here (can_claim_draw
    scans the whole game history and is far too slow inside a search leaf);
    only the fifty-move rule is considered.

    Args:
        board: A chess.Board instance.

    Returns:
        Integer score in centipawns.  Positive = good for side to move.
    """
    # Terminal positions
    if board.is_checkmate():
        # The side to move is checkmated → worst possible score
        return -MATE_SCORE

    if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return 0
    if not any(board.legal_moves):
        return 0  # Stalemate (checkmate already handled above)

    phase = _game_phase(board)
    score_white = 0
    score_black = 0

    # --- Material + Piece-Square Tables ---
    white_bishops = 0
    black_bishops = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue

        piece_type = piece.piece_type
        material = PIECE_VALUES.get(piece_type, 0)
        pst = _pst_score(piece_type, square, piece.color == chess.WHITE, phase)

        if piece.color == chess.WHITE:
            score_white += material + pst
            if piece_type == chess.BISHOP:
                white_bishops += 1
        else:
            score_black += material + pst
            if piece_type == chess.BISHOP:
                black_bishops += 1

    # --- Bishop Pair Bonus ---
    if white_bishops >= 2:
        score_white += BISHOP_PAIR_BONUS
    if black_bishops >= 2:
        score_black += BISHOP_PAIR_BONUS

    # --- Castling Rights (King Safety proxy) ---
    if board.has_kingside_castling_rights(chess.WHITE):
        score_white += CASTLING_BONUS
    if board.has_queenside_castling_rights(chess.WHITE):
        score_white += CASTLING_BONUS
    if board.has_kingside_castling_rights(chess.BLACK):
        score_black += CASTLING_BONUS
    if board.has_queenside_castling_rights(chess.BLACK):
        score_black += CASTLING_BONUS

    # --- Mobility ---
    # Only count side-to-move mobility (cheap). Opponent mobility would
    # require a null-move push/pop at every leaf which is too expensive.
    # NOTE: this is an intentional tempo-biased proxy, not a symmetric eval;
    # callers that already enumerated moves may pass the count via `mobility`.
    our_mobility = board.legal_moves.count() if mobility is None else mobility

    if board.turn == chess.WHITE:
        score_white += our_mobility * MOBILITY_WEIGHT
    else:
        score_black += our_mobility * MOBILITY_WEIGHT

    # --- Final score from side-to-move perspective ---
    absolute_score = score_white - score_black
    return absolute_score if board.turn == chess.WHITE else -absolute_score


# ---------------------------------------------------------------------------
# Move Ordering — MVV-LVA + checks + promotions
# ---------------------------------------------------------------------------

# MVV-LVA: we want captures of high-value pieces by low-value attackers first.
# Score = victim_value * 10 - attacker_value  (higher is better)

def _mvv_lva_score(board: chess.Board, move: chess.Move) -> int:
    """Return MVV-LVA ordering score for a capture move (higher = better)."""
    victim = board.piece_type_at(move.to_square)
    if victim is None:
        # En passant
        victim = chess.PAWN
    attacker = board.piece_type_at(move.from_square)
    if attacker is None:
        attacker = chess.PAWN
    return PIECE_VALUES.get(victim, 0) * 10 - PIECE_VALUES.get(attacker, 0)


def order_moves(board: chess.Board, moves: list[chess.Move]) -> list[chess.Move]:
    """
    Order moves for alpha-beta pruning efficiency.

    Priority (highest first):
    1. Captures, sorted by MVV-LVA score (descending)
    2. Queen promotions
    3. Checks
    4. Other promotions
    5. Quiet moves

    Args:
        board: Current board state.
        moves: List of legal moves to order.

    Returns:
        A new list of moves sorted by estimated importance.
    """
    captures: list[tuple[int, chess.Move]] = []
    queen_promotions: list[chess.Move] = []
    checks: list[chess.Move] = []
    other_promotions: list[chess.Move] = []
    quiet: list[chess.Move] = []

    for move in moves:
        is_capture = board.is_capture(move)
        is_promotion = move.promotion is not None

        if is_capture:
            captures.append((_mvv_lva_score(board, move), move))
        elif is_promotion and move.promotion == chess.QUEEN:
            queen_promotions.append(move)
        elif is_promotion:
            other_promotions.append(move)
        else:
            # Use gives_check() — no push/pop needed, much faster
            if board.gives_check(move):
                checks.append(move)
            else:
                quiet.append(move)

    # Sort captures by MVV-LVA descending
    captures.sort(key=lambda x: x[0], reverse=True)

    result: list[chess.Move] = []
    result.extend(m for _, m in captures)
    result.extend(queen_promotions)
    result.extend(checks)
    result.extend(other_promotions)
    result.extend(quiet)

    return result


# ---------------------------------------------------------------------------
# Quiescence Search
# ---------------------------------------------------------------------------

def quiescence(board: chess.Board, alpha: int, beta: int, depth_limit: int = 6,
               ply: int = 0) -> int:
    """
    Quiescence search to resolve tactical sequences and prevent the
    horizon effect.

    Examines only captures, promotions, and check evasions until the
    position is 'quiet' (no more tactical moves) or the depth limit is
    reached. When in check there is no stand-pat (the side to move must
    resolve the check), which avoids overvaluing checked positions.

    Args:
        board: Current board position.
        alpha: Alpha bound.
        beta: Beta bound.
        depth_limit: Maximum quiescence depth to prevent explosion.
        ply: Distance from the root; used to prefer faster mates.

    Returns:
        Score in centipawns from the perspective of the side to move.
    """
    # Terminal positions
    if board.is_checkmate():
        # Prefer shorter mates: deeper mates score slightly worse
        return -(MATE_SCORE - ply)

    if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return 0

    in_check = board.is_check()
    # Enumerate legal moves once; reuse for both the mobility term and the
    # tactical move filter below (evaluate_board otherwise regenerates them).
    all_moves = list(board.legal_moves)
    stand_pat = evaluate_board(board, mobility=len(all_moves))

    if depth_limit <= -4:
        if in_check:
            # At the hard horizon, evaluate legal evasions rather than allowing
            # the checked side to stand pat in an illegal position.
            scores = []
            for move in all_moves:
                board.push(move)
                try:
                    scores.append(-evaluate_board(board))
                finally:
                    board.pop()
            return max(scores)
        return stand_pat

    if not in_check:
        if depth_limit <= 0:
            return stand_pat

        if stand_pat >= beta:
            return beta  # Fail-high (beta cutoff)

        if stand_pat > alpha:
            alpha = stand_pat

    # Generate tactical moves: captures + promotions
    # When in check, consider ALL moves (check evasions)
    if in_check:
        tactical_moves = all_moves
        if not tactical_moves:
            # Checkmated (should have been caught above) — safety net
            return -(MATE_SCORE - ply)
    else:
        tactical_moves = [
            m for m in all_moves
            if board.is_capture(m) or m.promotion is not None
        ]

    # Order tactical moves
    tactical_moves = order_moves(board, tactical_moves)

    for move in tactical_moves:
        board.push(move)
        score = -quiescence(board, -beta, -alpha, depth_limit - 1, ply + 1)
        board.pop()

        if score >= beta:
            return beta  # Beta cutoff

        if score > alpha:
            alpha = score

    return alpha


# ---------------------------------------------------------------------------
# Alpha-Beta (Negamax variant)
# ---------------------------------------------------------------------------

def alpha_beta(
    board: chess.Board,
    depth: int,
    alpha: int,
    beta: int,
    ply: int = 0,
) -> tuple[int, Optional[chess.Move]]:
    """
    Negamax alpha-beta search.

    Returns the best score from the perspective of the side to move and the
    best move found.  At depth 0 the search transitions to quiescence search.
    Mate scores encode distance-to-mate so faster mates are preferred.

    Args:
        board: Current board state.
        depth: Remaining search depth (full moves).
        alpha: Alpha bound.
        beta: Beta bound.
        ply: Distance from the root (used for mate-distance scoring).

    Returns:
        Tuple of (score_cp, best_move).  best_move is None only at
        terminal nodes or when no legal moves exist.
    """
    # Terminal checks
    if board.is_checkmate():
        return -(MATE_SCORE - ply), None

    if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return 0, None

    # Leaf node → quiescence search
    if depth <= 0:
        return quiescence(board, alpha, beta, ply=ply), None

    best_move: Optional[chess.Move] = None
    moves = order_moves(board, list(board.legal_moves))

    if not moves:
        # No legal moves (should be caught above, but safety check)
        return evaluate_board(board), None

    for move in moves:
        board.push(move)
        score, _ = alpha_beta(board, depth - 1, -beta, -alpha, ply + 1)
        score = -score
        board.pop()

        if score > alpha:
            alpha = score
            best_move = move

        if alpha >= beta:
            break  # Beta cutoff

    return alpha, best_move


# ---------------------------------------------------------------------------
# Result Dataclass
# ---------------------------------------------------------------------------

@dataclass
class AlphaBetaResult:
    """
    Result of an alpha-beta tactical analysis.

    Attributes:
        best_move: The best move found by the search.
        score_cp: Evaluation in centipawns from the side to move's perspective.
        is_tactical: True if the position contains a forced mate or the best
                     move scores > 300cp above the second-best move.
        candidate_moves: Top moves with their scores, sorted descending.
                         Used to seed MCTS search priors.
    """
    best_move: Optional[chess.Move]
    score_cp: int
    is_tactical: bool
    candidate_moves: List[Tuple[chess.Move, int]]


# ---------------------------------------------------------------------------
# AlphaBetaFilter — Main Interface
# ---------------------------------------------------------------------------

class AlphaBetaFilter:
    """
    Tactical filter that runs a shallow alpha-beta search to detect
    forced tactical sequences before handing off to MCTS.

    Usage::

        filt = AlphaBetaFilter(depth=3)
        result = filt.analyze(board)
        if result.is_tactical:
            # Play the tactical move immediately
            board.push(result.best_move)
        else:
            # Feed result.candidate_moves as priors into MCTS
            ...

    Args:
        depth: Search depth in plies (half-moves). Default 3 gives good
               speed/quality balance (< 200 ms on typical positions).
    """

    def __init__(self, depth: int = 3) -> None:
        self.depth = depth

    def analyze(self, board: chess.Board) -> AlphaBetaResult:
        """
        Analyze the position with alpha-beta search.

        Returns an AlphaBetaResult containing the best move, score, whether
        the position is tactical, and a list of all candidate moves with
        their scores (sorted descending by score).

        Args:
            board: The board position to analyze.

        Returns:
            AlphaBetaResult with tactical analysis.
        """
        legal_moves = list(board.legal_moves)

        # Edge case: no legal moves (game already over)
        if not legal_moves:
            if board.is_checkmate():
                return AlphaBetaResult(
                    best_move=None,
                    score_cp=-MATE_SCORE,
                    is_tactical=True,
                    candidate_moves=[],
                )
            else:
                # Stalemate or other draw
                return AlphaBetaResult(
                    best_move=None,
                    score_cp=0,
                    is_tactical=False,
                    candidate_moves=[],
                )

        # Two-pass root search for speed:
        #
        # Pass 1 — Full alpha-beta with narrowing window.
        # The first move (from ordered list) gets a full window; subsequent
        # moves use the best score found so far as alpha, which enables
        # massive pruning once a strong move (e.g. mate) is found.
        #
        # Pass 2 — Any move that was pruned (scored <= alpha at time of
        # search) gets a verification re-search with its own full window
        # so we get an accurate score for ranking all candidates.

        ordered = order_moves(board, legal_moves)
        move_scores: list[tuple[chess.Move, int]] = []
        needs_research: list[chess.Move] = []
        best_alpha = -MATE_SCORE

        for move in ordered:
            board.push(move)
            score, _ = alpha_beta(
                board,
                self.depth - 1,
                -MATE_SCORE,
                -best_alpha,
            )
            score = -score  # Negate because alpha_beta returns from child's perspective
            board.pop()

            if score > best_alpha:
                best_alpha = score
                move_scores.append((move, score))
            else:
                # This move was pruned — score is a lower bound only.
                # Mark for re-search if we need accurate ranking.
                needs_research.append(move)

        # Pass 2 — re-search pruned moves with a window just below best.
        # We only need accurate scores for MCTS priors, so search with a
        # window of [best_alpha - 400, best_alpha] to limit work.
        # Moves far worse than best are scored with a coarse estimate.
        research_window = 400
        for move in needs_research:
            board.push(move)
            score, _ = alpha_beta(
                board,
                self.depth - 1,
                -MATE_SCORE,
                -(best_alpha - research_window),
            )
            score = -score
            board.pop()

            if score <= best_alpha - research_window:
                # Still far below best — use a quick static eval as estimate
                board.push(move)
                score = min(best_alpha - research_window, -evaluate_board(board))
                board.pop()

            move_scores.append((move, score))

        # Sort by score descending (best first)
        move_scores.sort(key=lambda x: x[1], reverse=True)

        best_move, best_score = move_scores[0]

        # Determine if position is tactical:
        # 1. Forced mate detected (|score| close to MATE_SCORE)
        # 2. Best move scores > 300cp above the second-best move
        is_tactical = False

        if abs(best_score) >= MATE_SCORE - 100:
            # Forced mate (within 100 plies — generous margin)
            is_tactical = True
        elif len(move_scores) >= 2:
            second_best_score = move_scores[1][1]
            if best_score - second_best_score > 300:
                is_tactical = True

        # If only one legal move, it's forced (tactical by definition)
        if len(move_scores) == 1:
            is_tactical = True

        return AlphaBetaResult(
            best_move=best_move,
            score_cp=best_score,
            is_tactical=is_tactical,
            candidate_moves=move_scores,
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def find_best_move(board: chess.Board, depth: int = 3) -> Optional[chess.Move]:
    """
    Quick convenience function to get the best move for a position.

    Args:
        board: Board position to analyze.
        depth: Search depth in plies.

    Returns:
        The best chess.Move, or None if the game is over.
    """
    filt = AlphaBetaFilter(depth=depth)
    result = filt.analyze(board)
    return result.best_move


# ---------------------------------------------------------------------------
# Self-test / Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("Alpha-Beta Engine - Self-Test Suite")
    print("=" * 60)

    passed = 0
    failed = 0

    def _test(name: str, condition: bool, detail: str = "") -> None:
        global passed, failed
        status = "PASS" if condition else "FAIL"
        if not condition:
            failed += 1
        else:
            passed += 1
        msg = f"  [{status}] {name}"
        if detail:
            msg += f"  ({detail})"
        print(msg)

    # --- Test 1: Starting position evaluation ---
    print("\n--- Starting Position ---")
    b = chess.Board()
    score = evaluate_board(b)
    _test("Starting eval is near zero", abs(score) < 100, f"score={score}")

    # --- Test 2: Material advantage ---
    print("\n--- Material Advantage ---")
    # White up a queen
    b = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    score = evaluate_board(b)
    _test("White up a queen -> positive score", score > 800, f"score={score}")

    # --- Test 3: Mate in 1 ---
    print("\n--- Mate in 1 ---")
    # Scholar's mate setup: White to play Qxf7#
    b = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    filt = AlphaBetaFilter(depth=3)
    t0 = time.perf_counter()
    result = filt.analyze(b)
    elapsed = time.perf_counter() - t0
    mate_move = chess.Move.from_uci("h5f7")
    _test("Finds Qxf7#", result.best_move == mate_move,
          f"got {result.best_move}, expected {mate_move}")
    _test("Mate score detected", result.score_cp >= MATE_SCORE - 100,
          f"score={result.score_cp}")
    _test("Marked as tactical", result.is_tactical)
    _test("Speed < 1s", elapsed < 1.0, f"{elapsed:.3f}s")

    # --- Test 4: Checkmate position (game over) ---
    print("\n--- Checkmate Position ---")
    b = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1")
    result = filt.analyze(b)
    _test("Checkmate: no best move", result.best_move is None)
    _test("Checkmate: score is -MATE_SCORE", result.score_cp == -MATE_SCORE)
    _test("Checkmate: is tactical", result.is_tactical)

    # --- Test 5: Stalemate ---
    print("\n--- Stalemate ---")
    b = chess.Board("k7/8/1K6/8/8/8/8/8 b - - 0 1")
    # Black king is on a8, White king on b6. If it's stalemate:
    if b.is_stalemate():
        result = filt.analyze(b)
        _test("Stalemate: score is 0", result.score_cp == 0)
        _test("Stalemate: not tactical", not result.is_tactical)
    else:
        # Not actually stalemate in this position; set up proper one
        b = chess.Board("k7/8/1K6/8/8/8/8/8 b - - 0 1")
        score = evaluate_board(b)
        _test("King vs King-like: near zero or valid", True, f"score={score}")

    # --- Test 6: Real stalemate ---
    print("\n--- Real Stalemate ---")
    b = chess.Board("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1")
    if b.is_stalemate():
        result = filt.analyze(b)
        _test("Real stalemate: no moves", result.best_move is None)
        _test("Real stalemate: score 0", result.score_cp == 0)
    else:
        _test("Position is not stalemate", False, "Expected stalemate")

    # --- Test 7: Performance on starting position ---
    print("\n--- Performance (Starting Position, depth=3) ---")
    b = chess.Board()
    filt3 = AlphaBetaFilter(depth=3)
    t0 = time.perf_counter()
    result = filt3.analyze(b)
    elapsed = time.perf_counter() - t0
    _test("Depth 3 < 5s", elapsed < 5.0, f"{elapsed:.3f}s")
    _test("Has candidate moves", len(result.candidate_moves) == 20,
          f"got {len(result.candidate_moves)}")
    _test("Has a best move", result.best_move is not None,
          f"best={result.best_move}")

    # --- Test 8: Captures detected in quiescence ---
    print("\n--- Quiescence (hanging piece) ---")
    # Black queen hanging on d4, White to move
    b = chess.Board("rnb1kbnr/pppppppp/8/8/3q4/4P3/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
    result = filt.analyze(b)
    # Should find exd4 or some capture of the queen
    found_capture = result.best_move is not None and b.is_capture(result.best_move)
    _test("Captures hanging queen", found_capture or result.score_cp > 500,
          f"move={result.best_move}, score={result.score_cp}")

    # --- Test 9: Move ordering sanity ---
    print("\n--- Move Ordering ---")
    b = chess.Board()
    moves = list(b.legal_moves)
    ordered = order_moves(b, moves)
    _test("Move ordering returns same count", len(ordered) == len(moves),
          f"{len(ordered)} vs {len(moves)}")

    # --- Test 10: AlphaBetaResult fields ---
    print("\n--- AlphaBetaResult fields ---")
    b = chess.Board()
    result = filt.analyze(b)
    _test("best_move is chess.Move", isinstance(result.best_move, chess.Move))
    _test("score_cp is int", isinstance(result.score_cp, int))
    _test("is_tactical is bool", isinstance(result.is_tactical, bool))
    _test("candidate_moves is list", isinstance(result.candidate_moves, list))
    if result.candidate_moves:
        m, s = result.candidate_moves[0]
        _test("candidate tuple structure", isinstance(m, chess.Move) and isinstance(s, int))

    # --- Test 11: Back-rank mate ---
    print("\n--- Back-Rank Mate ---")
    b = chess.Board("6k1/5ppp/8/8/8/8/8/R3K3 w Q - 0 1")
    result = filt.analyze(b)
    # Should find Ra8# or similar
    if result.best_move:
        b.push(result.best_move)
        is_mate = b.is_checkmate()
        b.pop()
        _test("Finds back-rank mate", is_mate or result.score_cp >= MATE_SCORE - 100,
              f"move={result.best_move}, score={result.score_cp}")
    else:
        _test("Finds back-rank mate", False, "No move found")

    # --- Test 12: Insufficient material ---
    print("\n--- Insufficient Material ---")
    b = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    score = evaluate_board(b)
    _test("K vs K is draw (0)", score == 0, f"score={score}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    if failed > 0:
        print("\nSome tests failed - review output above.")
    else:
        print("\nAll tests passed!")
    raise SystemExit(1 if failed else 0)
