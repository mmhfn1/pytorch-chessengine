"""
move_encoding.py
=================
Implements the bijection between a `chess.Move` and an index into the
flat 4672-dimensional policy vector produced by the network, following
the scheme used by AlphaZero / Leela Chess Zero:

    8x8 "from" squares  x  73 "move planes"  =  4672

The 73 move-planes per origin square are laid out as:

    [ 0 .. 55]  "queen-like" sliding moves: 8 compass directions
                 x 7 possible distances (1..7 squares)
    [56 .. 63]  knight-jump moves (8 possible knight offsets)
    [64 .. 72]  underpromotions: 3 forward diagonals/straight
                 x 3 promotion pieces (knight, bishop, rook)
                 (promotion to queen is encoded implicitly by a normal
                  queen-like move landing on the back rank)

Because the policy is always expressed from the perspective of the
side to move, every move is first expressed in "white's frame" by
vertically mirroring the board whenever it is black's turn. This is
exactly mirrored by `encoder.py`'s board-history encoding, so the
network only ever has to learn "my perspective" patterns.
"""

from typing import Optional
import chess

# 8 compass directions as (delta_file, delta_rank), ordered consistently.
_DIRECTIONS = [
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1),
]  # N, NE, E, SE, S, SW, W, NW

# The 8 possible (delta_file, delta_rank) knight jumps.
_KNIGHT_DELTAS = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
]

# Underpromotion directions: forward-left, forward, forward-right
# (always "forward" in the mover's own, already-mirrored, frame).
_UNDERPROMO_DIRS = [(-1, 1), (0, 1), (1, 1)]
_UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

POLICY_SIZE = 64 * 73  # = 4672


def _mirror_square(square: int) -> int:
    """Flip a square vertically (rank r -> rank 7-r), file unchanged."""
    return chess.square_mirror(square)


def move_to_index(move: chess.Move, board: chess.Board) -> Optional[int]:
    """
    Map a legal move (in the *actual* board orientation) to its index
    in the flat 4672-way policy vector, expressed in the current
    side-to-move's own frame of reference.

    Returns None if the move cannot be represented (should not happen
    for any legal chess move).
    """
    from_sq, to_sq = move.from_square, move.to_square

    # Re-express everything from "white to move" perspective.
    if board.turn == chess.BLACK:
        from_sq = _mirror_square(from_sq)
        to_sq = _mirror_square(to_sq)

    ff, fr = chess.square_file(from_sq), chess.square_rank(from_sq)
    tf, tr = chess.square_file(to_sq), chess.square_rank(to_sq)
    df, dr = tf - ff, tr - fr

    if move.promotion is not None and move.promotion != chess.QUEEN:
        # --- underpromotion -------------------------------------------------
        try:
            dir_idx = _UNDERPROMO_DIRS.index((df, dr))
            piece_idx = _UNDERPROMO_PIECES.index(move.promotion)
        except ValueError:
            return None
        plane = 64 + dir_idx * 3 + piece_idx

    elif (df, dr) in _KNIGHT_DELTAS:
        # --- knight move -----------------------------------------------------
        plane = 56 + _KNIGHT_DELTAS.index((df, dr))

    else:
        # --- queen-like sliding move (also covers queen promotions) ---------
        distance = max(abs(df), abs(dr))
        if distance == 0:
            return None
        step = (df // distance, dr // distance)
        try:
            dir_idx = _DIRECTIONS.index(step)
        except ValueError:
            return None
        plane = dir_idx * 7 + (distance - 1)

    return from_sq * 73 + plane


def index_to_move(index: int, board: chess.Board) -> Optional[chess.Move]:
    """
    Inverse of `move_to_index`: given a policy index and the *current*
    board (used only to determine whether a queen-like move that lands
    on the back rank is actually a pawn promotion, and to mirror back
    into the real board orientation), reconstruct the corresponding
    `chess.Move`. Returns None for invalid indices.

    NOTE: this does not check legality; combine with
    `board.legal_moves` (or simply attempt `board.push`) to validate.
    """
    if not (0 <= index < POLICY_SIZE):
        return None

    from_sq_mirrored, plane = divmod(index, 73)
    ff, fr = chess.square_file(from_sq_mirrored), chess.square_rank(from_sq_mirrored)

    promotion = None

    if plane < 56:
        dir_idx, dist = divmod(plane, 7)
        df, dr = _DIRECTIONS[dir_idx]
        distance = dist + 1
        tf, tr = ff + df * distance, fr + dr * distance
    elif plane < 64:
        df, dr = _KNIGHT_DELTAS[plane - 56]
        tf, tr = ff + df, fr + dr
    else:
        sub = plane - 64
        dir_idx, piece_idx = divmod(sub, 3)
        df, dr = _UNDERPROMO_DIRS[dir_idx]
        tf, tr = ff + df, fr + dr
        promotion = _UNDERPROMO_PIECES[piece_idx]

    if not (0 <= tf <= 7 and 0 <= tr <= 7):
        return None

    to_sq_mirrored = chess.square(tf, tr)

    # Mirror back into the real board orientation if it's black to move.
    if board.turn == chess.BLACK:
        from_sq = _mirror_square(from_sq_mirrored)
        to_sq = _mirror_square(to_sq_mirrored)
    else:
        from_sq, to_sq = from_sq_mirrored, to_sq_mirrored

    # If this is a pawn reaching the back rank via a queen-like plane
    # (i.e. promotion wasn't already set by the underpromotion branch),
    # it's an implicit queen promotion.
    if promotion is None:
        piece = board.piece_at(from_sq)
        if piece is not None and piece.piece_type == chess.PAWN:
            back_rank = 7 if piece.color == chess.WHITE else 0
            if chess.square_rank(to_sq) == back_rank:
                promotion = chess.QUEEN

    return chess.Move(from_sq, to_sq, promotion=promotion)


def legal_move_mask(board: chess.Board):
    """
    Returns (indices, moves): a list of valid policy indices for every
    currently legal move, paired with the `chess.Move` objects, in the
    same order. Useful for masking the raw policy logits before
    softmax/normalization so illegal moves get zero probability.
    """
    indices = []
    moves = []
    for move in board.legal_moves:
        idx = move_to_index(move, board)
        if idx is not None:
            indices.append(idx)
            moves.append(move)
    return indices, moves
