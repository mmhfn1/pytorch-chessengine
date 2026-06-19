"""
tablebase.py
=============
Thin, defensive wrapper around python-chess's built-in Syzygy support
(`chess.syzygy`). Used in two places:

  1. Inside MCTS (mcts.py): once a search reaches a position with at
     most `tablebase_max_pieces` pieces on the board, we probe the
     tablebase for an exact win/draw/loss result instead of trusting
     the network's (necessarily approximate) evaluation. This directly
     addresses the user's concern that "raw RL networks struggle with
     exact 1-ply tactical endgames".

  2. Inside the UCI loop (uci.py): if a DTZ-optimal move exists in a
     probed position, we play it directly rather than spending search
     time, guaranteeing technically correct endgame conversion.

If no tablebase files are configured/found, every method degrades
gracefully to returning `None`, so the rest of the engine works
identically with or without Syzygy files present.
"""

from __future__ import annotations
import os
from typing import Optional

import chess
import chess.syzygy


class Tablebase:
    def __init__(self, path: Optional[str]):
        self.path = path
        self._tb: Optional[chess.syzygy.Tablebase] = None
        if path and os.path.isdir(path):
            try:
                self._tb = chess.syzygy.open_tablebase(path)
            except Exception:
                self._tb = None

    @property
    def available(self) -> bool:
        return self._tb is not None

    def close(self):
        if self._tb is not None:
            self._tb.close()

    def probe_value(self, board: chess.Board) -> Optional[float]:
        """
        Returns an exact value in {-1, 0, 1} from the perspective of
        the side to move, or None if the tablebase is unavailable or
        the position can't be probed (e.g. too many pieces, or
        en-passant/castling edge cases the WDL tables don't cover).
        """
        if self._tb is None:
            return None
        try:
            wdl = self._tb.probe_wdl(board)
        except (chess.syzygy.MissingTableError, KeyError, ValueError):
            return None
        if wdl > 0:
            return 1.0
        if wdl < 0:
            return -1.0
        return 0.0

    def probe_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Returns the DTZ-optimal move (fastest route to the proven
        result) for the side to move, or None if unavailable. Used by
        the UCI loop to play technically-perfect endgame moves once
        few enough pieces remain.

        Strategy: among all legal moves, keep only those that achieve
        the best possible resulting WDL for us (i.e. don't throw away
        a win for a draw, or a draw for a loss), then among those,
        pick the one with the smallest |DTZ| (converts fastest if
        winning, or survives longest if merely drawing/losing).
        """
        if self._tb is None:
            return None

        candidates = []  # (our_result_after, abs_dtz, move)
        for move in board.legal_moves:
            board.push(move)
            try:
                opp_wdl = self._tb.probe_wdl(board)
                dtz = self._tb.probe_dtz(board)
            except (chess.syzygy.MissingTableError, KeyError, ValueError):
                board.pop()
                continue
            board.pop()
            our_result_after = -opp_wdl  # flip perspective: opponent's WDL -> ours
            candidates.append((our_result_after, abs(dtz), move))

        if not candidates:
            return None

        best_result = max(c[0] for c in candidates)
        best_among_best = [c for c in candidates if c[0] == best_result]
        best_among_best.sort(key=lambda c: c[1])  # smallest |DTZ| first
        return best_among_best[0][2]
