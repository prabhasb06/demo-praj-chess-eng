"""
MODULE 1: board_encoder.py
Handles the representation and encoding of a chess position into an 8x8x18 float tensor,
as well as move encoding/decoding for the 4096-dimensional policy space.
"""

import numpy as np
import chess

# Mathematical Encoding Formulation:
# A chess board state S is mapped to a tensor X of shape (18, 8, 8).
# To maintain a single learning perspective, if it is Black's turn, we mirror the board
# vertically and swap colors before encoding.
# The planes are defined as:
# Planes 0-5:   Active player pieces (Pawn, Knight, Bishop, Rook, Queen, King)
# Planes 6-11:  Opponent pieces (Pawn, Knight, Bishop, Rook, Queen, King)
# Plane 12:     Active player castling rights - Kingside (O-O)
# Plane 13:     Active player castling rights - Queenside (O-O-O)
# Plane 14:     Opponent castling rights - Kingside (o-o)
# Plane 15:     Opponent castling rights - Queenside (o-o-o)
# Plane 16:     En Passant target square
# Plane 17:     Halfmove clock (draw rule counter), scaled as halfmove_clock / 100.0
#
# Note: there is no side-to-move plane. Because the encoding always flips to the
# active player's perspective, the side to move is implicitly "plane owner" and a
# constant plane would carry zero information.

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

def encode_board(board: chess.Board) -> np.ndarray:
    """
    Encodes a chess.Board state into a (18, 8, 8) numpy float32 array.
    Applies perspective flipping: if Black to move, mirrors board and swaps colors.
    """
    # 1. Perspective flipping: if Black's turn, mirror the board
    is_black = (board.turn == chess.BLACK)
    working_board = board.mirror() if is_black else board.copy()

    # Initialize tensor of shape (18, 8, 8)
    tensor = np.zeros((18, 8, 8), dtype=np.float32)

    # Helper to map 0-63 square to (row, col)
    # python-chess square 0 is a1 (bottom left, rank 1, file A)
    # Let's map rank 1 to row 7, rank 8 to row 0, file A to col 0, file H to col 7
    def get_coords(sq: int):
        row = 7 - chess.square_rank(sq)
        col = chess.square_file(sq)
        return row, col

    # 2. Encode pieces (Planes 0-5 for active player, 6-11 for opponent)
    # Since working_board is always from the active player's perspective (White to move)
    # Active pieces = White pieces, Opponent pieces = Black pieces
    for i, piece_type in enumerate(PIECE_TYPES):
        # Active player (White)
        for sq in working_board.pieces(piece_type, chess.WHITE):
            row, col = get_coords(sq)
            tensor[i, row, col] = 1.0
        # Opponent player (Black)
        for sq in working_board.pieces(piece_type, chess.BLACK):
            row, col = get_coords(sq)
            tensor[i + 6, row, col] = 1.0

    # 3. Castling rights (Planes 12-15)
    # Plane 12: Active Kingside
    if working_board.has_kingside_castling_rights(chess.WHITE):
        tensor[12, :, :] = 1.0
    # Plane 13: Active Queenside
    if working_board.has_queenside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    # Plane 14: Opponent Kingside
    if working_board.has_kingside_castling_rights(chess.BLACK):
        tensor[14, :, :] = 1.0
    # Plane 15: Opponent Queenside
    if working_board.has_queenside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0

    # 4. En Passant square (Plane 16)
    if working_board.has_legal_en_passant():
        ep_square = working_board.ep_square
        row, col = get_coords(ep_square)
        tensor[16, row, col] = 1.0

    # 5. Halfmove clock (Plane 17)
    # Scaled to [0, 1] range. Max halfmove clock in chess is 100 for 50-move rule.
    tensor[17, :, :] = min(working_board.halfmove_clock, 100) / 100.0

    return tensor

def encode_move(move: chess.Move) -> int:
    """
    Encodes a chess.Move into an integer index in the range [0, 4095].
    Formulation: index = from_square * 64 + to_square
    Note: Underpromotions are implicitly mapped to this coordinate grid.
    For execution/decoding, we reconstruct the chess.Move object.
    """
    return move.from_square * 64 + move.to_square


def encode_move_for_perspective(move: chess.Move, is_black: bool) -> int:
    """
    Encodes a chess.Move into an action index, mirroring squares when encoding
    for Black so the action space aligns with the mirrored board encoding.
    """
    if is_black:
        from_sq = chess.square_mirror(move.from_square)
        to_sq = chess.square_mirror(move.to_square)
    else:
        from_sq = move.from_square
        to_sq = move.to_square
    return from_sq * 64 + to_sq


def decode_move(action: int, board: chess.Board) -> chess.Move:
    """
    Decodes an action index in [0, 4095] into a chess.Move.
    Requires the current board to determine if the move is a promotion.
    Checks all legal promotions for the from/to pair and defaults to Queen.
    """
    from_square = action // 64
    to_square = action % 64

    # Default move
    move = chess.Move(from_square, to_square)

    # Check if this is a pawn promotion
    # Moving pawn to rank 8 (for white) or rank 1 (for black)
    piece = board.piece_at(from_square)
    if piece and piece.piece_type == chess.PAWN:
        to_rank = chess.square_rank(to_square)
        if (piece.color == chess.WHITE and to_rank == 7) or (piece.color == chess.BLACK and to_rank == 0):
            # Check which promotions are legal for this from/to pair.
            # Prefer Queen, but accept any legal promotion.
            preference = [chess.QUEEN, chess.KNIGHT, chess.ROOK, chess.BISHOP]
            for promo in preference:
                candidate = chess.Move(from_square, to_square, promotion=promo)
                if candidate in board.legal_moves:
                    return candidate
            # Fallback: default to Queen promotion even if not in legal moves
            # (caller should validate legality separately)
            move.promotion = chess.QUEEN

    return move


def decode_move_for_perspective(action: int, board: chess.Board) -> chess.Move:
    """
    Decodes an action index from the perspective-mirrored action space.
    If it's Black's turn, unmirrors the squares before decoding.
    """
    from_square = action // 64
    to_square = action % 64

    if board.turn == chess.BLACK:
        from_square = chess.square_mirror(from_square)
        to_square = chess.square_mirror(to_square)

    # Re-encode as a standard action and use the standard decoder
    return decode_move(from_square * 64 + to_square, board)


def get_legal_move_mask(board: chess.Board) -> np.ndarray:
    """
    Returns a boolean mask of shape (4096,) where True represents legal moves.
    Uses raw square indices (no perspective mirroring).
    """
    mask = np.zeros(4096, dtype=bool)
    for move in board.legal_moves:
        action = encode_move(move)
        mask[action] = True
    return mask


def get_legal_move_mask_for_perspective(board: chess.Board) -> np.ndarray:
    """
    Returns a boolean mask of shape (4096,) in the perspective-mirrored action space.
    When it's Black's turn, move squares are mirrored to match the mirrored board encoding.
    """
    mask = np.zeros(4096, dtype=bool)
    is_black = (board.turn == chess.BLACK)
    for move in board.legal_moves:
        action = encode_move_for_perspective(move, is_black)
        mask[action] = True
    return mask


def mirror_move(move: chess.Move) -> chess.Move:
    """
    Mirrors a move vertically. Used to map moves between the original board and the mirrored board.
    """
    from_sq = chess.square_mirror(move.from_square)
    to_sq = chess.square_mirror(move.to_square)
    return chess.Move(from_sq, to_sq, promotion=move.promotion)
