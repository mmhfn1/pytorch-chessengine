"""
heuristics.py
==============
Hand-crafted positional heuristics that quantify "attacking style" —
this is the concrete mechanism behind the requested "custom evaluation
bias". None of this replaces the learned value head; instead it
produces a single scalar `aggression_score` in [-1, 1] for a position
which is used two ways:

  1. As an auxiliary training target for a small extra head on the
     network (see network.py's AggressionHead) — over many self-play
     games the network learns to *predict* this style score directly
     from the board, which lets it generalize the bias to unseen
     positions instead of only memorizing it.

  2. As a direct, hand-coded blend into the value used inside MCTS
     (see mcts.py), so the search itself is nudged toward sharp,
     attacking lines even before the auxiliary head has converged.

The four ingredients, matching the user's request:
  - material_activity : reward active/centralized pieces over passive,
                          undeveloped, or purely defensive ones.
  - king_safety_diff   : reward exposing the OPPONENT's king while
                          keeping our own reasonably sheltered.
  - sac_potential      : reward being materially down while having
                          strong piece activity / king pressure — the
                          classic "sound sacrifice for initiative"
                          pattern.
  - mobility           : reward higher total legal-move mobility,
                          a generic proxy for active piece play.

All four are normalized to roughly [-1, 1] and combined with the
weights in `AggressionConfig`.
"""

from __future__ import annotations
import chess
from .config import AggressionConfig

# Standard material values (in pawns) used only for the heuristic —
# NOT used anywhere as a hard evaluation, just to detect "material
# down but active" sacrifice patterns.
_PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

# Centralization bonus table (distance-from-center based), file/rank 0..7.
_CENTER_BONUS = [
    [0.0, 0.1, 0.2, 0.3, 0.3, 0.2, 0.1, 0.0],
    [0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1],
    [0.2, 0.3, 0.4, 0.5, 0.5, 0.4, 0.3, 0.2],
    [0.3, 0.4, 0.5, 0.6, 0.6, 0.5, 0.4, 0.3],
    [0.3, 0.4, 0.5, 0.6, 0.6, 0.5, 0.4, 0.3],
    [0.2, 0.3, 0.4, 0.5, 0.5, 0.4, 0.3, 0.2],
    [0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1],
    [0.0, 0.1, 0.2, 0.3, 0.3, 0.2, 0.1, 0.0],
]


def _material_total(board: chess.Board, color: chess.Color) -> float:
    return sum(
        _PIECE_VALUES[p.piece_type]
        for p in board.piece_map().values()
        if p.color == color
    )


def material_activity(board: chess.Board, perspective: chess.Color) -> float:
    """
    Rewards pieces (knights/bishops/rooks/queen) placed actively
    (toward the center, or advanced into enemy territory) rather than
    sitting passively on the back rank. Pawns and king excluded.
    Returns a value roughly in [-1, 1].
    """
    score = 0.0
    count = 0
    for square, piece in board.piece_map().items():
        if piece.piece_type in (chess.PAWN, chess.KING):
            continue
        file, rank = chess.square_file(square), chess.square_rank(square)
        bonus = _CENTER_BONUS[rank][file]
        # Advancement bonus: pieces pushed into the opponent's half score higher.
        advanced = rank if perspective == chess.WHITE else (7 - rank)
        advance_bonus = max(0.0, (advanced - 3) / 4.0) * 0.4
        piece_score = bonus + advance_bonus
        if piece.color != perspective:
            piece_score = -piece_score
        score += piece_score
        count += 1
    if count == 0:
        return 0.0
    return max(-1.0, min(1.0, score / max(count, 6)))


def _king_exposure(board: chess.Board, color: chess.Color) -> float:
    """
    Rough proxy for how exposed a king is: pawn shield intact-ness,
    castling rights still available, and number of attackers the
    opponent has on squares around the king. Higher = more exposed.
    Returns a value in [0, 1].
    """
    king_sq = board.king(color)
    if king_sq is None:
        return 0.0

    exposure = 0.0

    # Lost castling rights without having castled yet is risky.
    has_kingside = board.has_kingside_castling_rights(color)
    has_queenside = board.has_queenside_castling_rights(color)
    started_on_e_file = chess.square_file(king_sq) == 4
    if not has_kingside and not has_queenside and started_on_e_file:
        exposure += 0.3  # king stuck in the center, rights gone, hasn't castled

    # Pawn shield: count friendly pawns on the 3 files around the king,
    # one or two ranks in front of it.
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    forward = 1 if color == chess.WHITE else -1
    shield_pawns = 0
    shield_slots = 0
    for df in (-1, 0, 1):
        f = king_file + df
        if not (0 <= f <= 7):
            continue
        for dr in (1, 2):
            r = king_rank + forward * dr
            if not (0 <= r <= 7):
                continue
            shield_slots += 1
            sq = chess.square(f, r)
            p = board.piece_at(sq)
            if p is not None and p.piece_type == chess.PAWN and p.color == color:
                shield_pawns += 1
    if shield_slots > 0:
        exposure += 0.4 * (1.0 - shield_pawns / shield_slots)

    # Direct attacker pressure: how many enemy pieces attack squares
    # immediately around the king.
    attackers = 0
    ring_squares = 0
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == 0 and dr == 0:
                continue
            f, r = king_file + df, king_rank + dr
            if not (0 <= f <= 7 and 0 <= r <= 7):
                continue
            ring_squares += 1
            sq = chess.square(f, r)
            if board.is_attacked_by(not color, sq):
                attackers += 1
    if ring_squares > 0:
        exposure += 0.3 * (attackers / ring_squares)

    return max(0.0, min(1.0, exposure))


def king_safety_differential(board: chess.Board, perspective: chess.Color) -> float:
    """
    Positive when the OPPONENT's king is more exposed than ours —
    exactly the "reward exposed enemy kings / attacking storm setups"
    bias requested. Returns a value in [-1, 1].
    """
    my_exposure = _king_exposure(board, perspective)
    opp_exposure = _king_exposure(board, not perspective)
    return max(-1.0, min(1.0, opp_exposure - my_exposure))


def mobility(board: chess.Board, perspective: chess.Color) -> float:
    """
    Mobility differential: (our legal-move count - opponent's),
    normalized. Requires temporarily flipping the side to move to
    count the opponent's mobility, which is safe since we restore it.
    """
    our_turn_is_perspective = board.turn == perspective
    my_moves = board.legal_moves.count() if our_turn_is_perspective else None
    board_copy = board.copy(stack=False)
    if board_copy.turn != perspective:
        board_copy.push(chess.Move.null())
    my_count = board_copy.legal_moves.count() if board_copy.turn == perspective else my_moves

    board_copy2 = board.copy(stack=False)
    if board_copy2.turn == perspective:
        board_copy2.push(chess.Move.null())
    opp_count = board_copy2.legal_moves.count()

    diff = my_count - opp_count
    return max(-1.0, min(1.0, diff / 20.0))


def sacrifice_potential(board: chess.Board, perspective: chess.Color) -> float:
    """
    Detects the "materially down but dangerous" pattern: positive only
    when we are behind on material yet our pieces are active and/or
    the enemy king is exposed — i.e. a *justified* sacrifice for
    initiative, not just "blundered a piece for nothing".
    Returns a value in [0, 1] (never negative — being down material
    with NO compensation simply contributes 0, it is not separately
    punished here since the value head already handles raw material).
    """
    my_material = _material_total(board, perspective)
    opp_material = _material_total(board, not perspective)
    deficit = opp_material - my_material  # positive if we're behind
    if deficit <= 0:
        return 0.0
    deficit_norm = min(1.0, deficit / 5.0)  # cap influence at ~5 pawns of deficit

    compensation = max(
        0.0, material_activity(board, perspective)
    ) * 0.5 + max(0.0, king_safety_differential(board, perspective)) * 0.5

    return max(0.0, min(1.0, deficit_norm * compensation))


def aggression_score(board: chess.Board, perspective: chess.Color,
                      cfg: AggressionConfig = AggressionConfig()) -> float:
    """
    Combines all four ingredients into the single scalar "attacking
    style" score used both as an auxiliary training target and as a
    direct MCTS value blend. Result is clamped to [-1, 1].
    """
    score = (
        cfg.material_activity_weight * material_activity(board, perspective)
        + cfg.king_safety_weight * king_safety_differential(board, perspective)
        + cfg.sac_potential_weight * sacrifice_potential(board, perspective)
        + cfg.mobility_weight * mobility(board, perspective)
    )
    total_weight = (
        cfg.material_activity_weight + cfg.king_safety_weight
        + cfg.sac_potential_weight + cfg.mobility_weight
    )
    if total_weight > 0:
        score /= total_weight
    return max(-1.0, min(1.0, score))
