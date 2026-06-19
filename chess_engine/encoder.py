"""
encoder.py
===========
Converts a `chess.Board` (plus its recent history) into the
multi-channel tensor fed to the network, following the AlphaZero/
LeelaChessZero input representation:

    For T=8 timesteps (current position + 7 most recent):
        12 planes : piece positions (6 piece types x 2 colors)
         2 planes : repetition counters (2-fold / 3-fold flags)
        -> 14 planes per timestep x 8 timesteps = 112 planes

    Plus 7 constant ("meta") planes for the *current* position:
        1 : side to move (all-ones if white, all-zeros if black --
            though since we always encode from the mover's own
            perspective this is close to constant; kept for fidelity
            with the original architecture and to help the value head)
        1 : total move number (normalized)
        4 : castling rights (white king-side, white queen-side,
            black king-side, black queen-side)
        1 : no-progress / halfmove clock (normalized, for 50-move rule)

    Total channels = 112 + 7 = 119  (matches config.NetworkConfig default)

Just like the move encoding, the board is always expressed from the
current side-to-move's perspective: when it is black's turn, every
plane is vertically mirrored (rank r -> 7-r) and black/white piece
planes are swapped, so the network is always "looking" at the board
as if it were playing white. This roughly halves the input pattern
space the network has to learn and is standard practice.
"""

from __future__ import annotations
from collections import deque
from typing import Deque, Optional

import numpy as np
import chess

from .config import NetworkConfig

_PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def _board_planes(board: chess.Board) -> np.ndarray:
    """
    12 planes (6 piece types x 2 colors) of shape (12, 8, 8), always
    expressed from the perspective of `board.turn` (mirrored if black).
    Plane order: [mine_pawn, mine_knight, ..., mine_king,
                  theirs_pawn, ..., theirs_king]
    """
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    mover = board.turn
    for square, piece in board.piece_map().items():
        rank, file = chess.square_rank(square), chess.square_file(square)
        if mover == chess.BLACK:
            rank = 7 - rank  # vertical mirror
        type_idx = _PIECE_TYPES.index(piece.piece_type)
        color_offset = 0 if piece.color == mover else 6
        planes[color_offset + type_idx, rank, file] = 1.0
    return planes


def _repetition_planes(board: chess.Board) -> np.ndarray:
    """2 planes flagging whether this exact position has occurred >=1 / >=2 times before."""
    planes = np.zeros((2, 8, 8), dtype=np.float32)
    try:
        if board.is_repetition(2):
            planes[0, :, :] = 1.0
        if board.is_repetition(3):
            planes[1, :, :] = 1.0
    except Exception:
        # is_repetition can be expensive/fragile on boards built without
        # full move-stack history; fail safe to "no repetition known".
        pass
    return planes


def _meta_planes(board: chess.Board) -> np.ndarray:
    """7 constant-valued planes describing global game state."""
    planes = np.zeros((7, 8, 8), dtype=np.float32)
    mover = board.turn

    planes[0, :, :] = 1.0 if mover == chess.WHITE else 0.0
    planes[1, :, :] = min(board.fullmove_number, 200) / 200.0

    # Castling rights, always listed as [mine_kingside, mine_queenside,
    # theirs_kingside, theirs_queenside] regardless of color, since the
    # board is already mirrored to the mover's perspective.
    planes[2, :, :] = 1.0 if board.has_kingside_castling_rights(mover) else 0.0
    planes[3, :, :] = 1.0 if board.has_queenside_castling_rights(mover) else 0.0
    planes[4, :, :] = 1.0 if board.has_kingside_castling_rights(not mover) else 0.0
    planes[5, :, :] = 1.0 if board.has_queenside_castling_rights(not mover) else 0.0

    planes[6, :, :] = min(board.halfmove_clock, 100) / 100.0
    return planes


class HistoryEncoder:
    """
    Stateful helper that a self-play / UCI game loop keeps around to
    avoid re-deriving the full history every ply. Internally just
    keeps a deque of the last T `_board_planes()` + `_repetition_planes`
    snapshots (24 + 2 = 14 planes each) and concatenates them with the
    current meta-planes on every `encode()` call.

    Usage:
        enc = HistoryEncoder(cfg)
        enc.push(board)              # call after every move, including ply 0 (initial board)
        tensor = enc.encode(board)   # shape (C, 8, 8) float32, ready for the network
    """

    def __init__(self, cfg: NetworkConfig):
        self.cfg = cfg
        self._history: Deque[np.ndarray] = deque(maxlen=cfg.history_length)

    def reset(self):
        self._history.clear()

    def clone(self) -> "HistoryEncoder":
        """
        Returns an independent copy whose internal deque can be pushed
        to during a single MCTS rollout (to correctly reflect the
        positions visited along that simulated path) without mutating
        this encoder's "real game" history.
        """
        new = HistoryEncoder(self.cfg)
        new._history = deque(self._history, maxlen=self.cfg.history_length)
        return new

    def push(self, board: chess.Board):
        """Record the per-position planes (piece + repetition) for `board`."""
        planes = np.concatenate(
            [_board_planes(board), _repetition_planes(board)], axis=0
        )  # (14, 8, 8)
        self._history.append(planes)

    def encode(self, board: chess.Board) -> np.ndarray:
        """
        Returns the full (C, 8, 8) input tensor. Assumes `push(board)`
        has already been called for the current position (and ideally
        for prior positions too, to populate real history). Missing
        history slots (e.g. at the very start of a game) are zero-padded,
        matching AlphaZero's own handling of "no history available yet".
        """
        frames = list(self._history)
        pad_count = self.cfg.history_length - len(frames)
        if pad_count > 0:
            pad = [np.zeros((self.cfg.planes_per_position, 8, 8), dtype=np.float32)] * pad_count
            frames = pad + frames
        else:
            frames = frames[-self.cfg.history_length:]

        history_tensor = np.concatenate(frames, axis=0)  # (T*14, 8, 8)
        meta = _meta_planes(board)                        # (7, 8, 8)
        return np.concatenate([history_tensor, meta], axis=0)  # (C, 8, 8)


def encode_single(board: chess.Board, cfg: NetworkConfig) -> np.ndarray:
    """
    Convenience one-shot encoder with NO real history (all history
    planes zeroed except the current position). Handy for quick
    evaluation / tablebase-adjacent leaf scoring where reconstructing
    full game history isn't available or needed.
    """
    enc = HistoryEncoder(cfg)
    enc.push(board)
    return enc.encode(board)
