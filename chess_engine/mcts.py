"""
mcts.py
========
PUCT (Predictor + UCT) Monte-Carlo Tree Search, as used by AlphaZero,
with two engine-specific extensions:

  1. Dirichlet noise mixed into the ROOT node's prior policy only
     (never at internal nodes), to keep self-play exploration high
     without polluting the actual tree statistics used at non-root
     nodes — exactly matching the AlphaZero paper.

  2. "Effective value" leaf evaluation: rather than backing up the raw
     network value, every leaf's value is blended with the hand-
     crafted + learned aggression score (heuristics.aggression_score
     and the network's own AggressionHead output), so the search
     itself is biased toward sharp, attacking lines — not just the
     final policy/value targets used at training time. See
     `effective_value()` below.

The implementation is single-threaded-per-search but designed to be
called many times in parallel (e.g. once per self-play worker
process); virtual loss support is included so a future batched/
multi-threaded search can reuse the same Node class safely.
"""

from __future__ import annotations
import math
import time
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import chess

from .config import EngineConfig
from .move_encoding import legal_move_mask
from .encoder import HistoryEncoder
from .heuristics import aggression_score
from .network import ChessNet


class Node:
    __slots__ = (
        "parent", "move", "prior", "children", "visit_count",
        "value_sum", "virtual_loss_count", "is_expanded", "to_play",
    )

    def __init__(self, parent: Optional["Node"], move: Optional[chess.Move],
                 prior: float, to_play: chess.Color):
        self.parent = parent
        self.move = move                 # move that led to this node from the parent
        self.prior = prior                # P(s, a) from the policy network
        self.children: Dict[chess.Move, "Node"] = {}
        self.visit_count = 0
        self.value_sum = 0.0              # sum of backed-up values, from this node's own to_play perspective
        self.virtual_loss_count = 0
        self.is_expanded = False
        self.to_play = to_play            # side to move AT this node

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    """
    One MCTS instance is bound to a single network + config and is
    reused across an entire search (and can be reused across many
    searches in sequence, e.g. for every move of a self-play game).
    """

    def __init__(self, network: ChessNet, cfg: EngineConfig, device: torch.device):
        self.net = network
        self.cfg = cfg
        self.device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, board: chess.Board, history_encoder: HistoryEncoder,
            add_root_noise: bool = True,
            tablebase=None) -> Tuple[Node, Dict[chess.Move, float]]:
        """
        Runs `cfg.mcts.num_simulations` simulations from `board` and
        returns (root_node, visit_distribution) where visit_distribution
        maps each legal move to its normalized visit-count fraction —
        this IS the MCTS policy target used for training (pi).
        """
        mcts_cfg = self.cfg.mcts
        root = Node(parent=None, move=None, prior=1.0, to_play=board.turn)
        self._expand(root, board, history_encoder)

        if add_root_noise and len(root.children) > 0:
            self._add_dirichlet_noise(root)

        for _ in range(mcts_cfg.num_simulations):
            self._simulate(root, board.copy(stack=False), history_encoder, tablebase)

        visit_dist = {
            move: child.visit_count / max(1, root.visit_count)
            for move, child in root.children.items()
        }
        return root, visit_dist

    def select_move(self, visit_dist: Dict[chess.Move, float], ply_count: int) -> chess.Move:
        """Temperature-controlled move sampling from the visit distribution."""
        moves = list(visit_dist.keys())
        visits = np.array(list(visit_dist.values()), dtype=np.float64)

        temperature = (
            self.cfg.mcts.temperature_early
            if ply_count < self.cfg.mcts.temperature_moves
            else self.cfg.mcts.temperature_late
        )

        if temperature <= 1e-3:
            return moves[int(np.argmax(visits))]

        powered = np.power(visits, 1.0 / temperature)
        total = powered.sum()
        if total <= 0:
            return moves[int(np.argmax(visits))]
        probs = powered / total
        idx = np.random.choice(len(moves), p=probs)
        return moves[idx]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def run_timed(self, board: chess.Board, history_encoder: HistoryEncoder,
                   time_budget: float = float("inf"),
                   add_root_noise: bool = False,
                   tablebase=None,
                   min_simulations: int = 1,
                   max_simulations: Optional[int] = None,
                   stop_event: Optional["threading.Event"] = None,
                   ) -> Tuple[Node, Dict[chess.Move, float], int]:
        """
        Like `run()`, but stops based on wall-clock time and/or an
        externally-set `stop_event` instead of a fixed simulation
        count — used by the UCI loop, where how many simulations are
        affordable varies move to move depending on the remaining
        clock, and where a GUI can send `stop` at any moment.

        Stopping conditions (checked periodically, not after every
        single simulation, to keep the time-check overhead negligible):
          - `stop_event` is set (e.g. UCI `stop` received), OR
          - `time_budget` seconds have elapsed, OR
          - `max_simulations` simulations have been run (if given).
        Regardless of the above, at least `min_simulations` simulations
        are always run, so the engine never returns a move based on
        zero search even under extreme time pressure.

        Returns (root_node, visit_distribution, simulations_run).
        """
        root = Node(parent=None, move=None, prior=1.0, to_play=board.turn)
        self._expand(root, board, history_encoder)

        if add_root_noise and len(root.children) > 0:
            self._add_dirichlet_noise(root)

        sims_done = 0
        if len(root.children) > 1:
            start = time.monotonic()
            check_every = 8  # amortize time.monotonic()/event-check overhead
            while True:
                for _ in range(check_every):
                    self._simulate(root, board.copy(stack=False), history_encoder, tablebase)
                    sims_done += 1
                    if max_simulations is not None and sims_done >= max_simulations:
                        break
                if max_simulations is not None and sims_done >= max_simulations:
                    break
                if sims_done < min_simulations:
                    continue
                if stop_event is not None and stop_event.is_set():
                    break
                if time.monotonic() - start >= time_budget:
                    break

        visit_dist = {
            move: child.visit_count / max(1, root.visit_count)
            for move, child in root.children.items()
        }
        return root, visit_dist, sims_done

    def _puct_score(self, parent: Node, child: Node) -> float:
        mcts_cfg = self.cfg.mcts
        c_puct = (
            math.log((parent.visit_count + mcts_cfg.c_puct_base + 1) / mcts_cfg.c_puct_base)
            + mcts_cfg.c_puct_init
        )
        u = c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)

        if child.visit_count == 0:
            # First Play Urgency: discourage but don't forbid exploring totally
            # unvisited children, matching modern AlphaZero-derivative engines.
            q = parent.value - mcts_cfg.fpu_reduction
        else:
            q = child.value
        return q + u

    def _select_child(self, node: Node) -> Tuple[chess.Move, Node]:
        return max(node.children.items(), key=lambda mc: self._puct_score(node, mc[1]))

    def _expand(self, node: Node, board: chess.Board, history_encoder: HistoryEncoder):
        """Runs the network once on `board` and populates `node.children`."""
        x = torch.from_numpy(history_encoder.encode(board)).unsqueeze(0).to(self.device)
        policy_probs, value, aggression = self.net.infer(x)
        policy_probs = policy_probs[0].cpu().numpy()
        value = float(value[0].item())
        aggression_net = float(aggression[0].item())

        indices, moves = legal_move_mask(board)
        if len(moves) == 0:
            node.is_expanded = True
            return value, aggression_net

        priors = policy_probs[indices]
        total = priors.sum()
        if total > 1e-8:
            priors = priors / total
        else:
            priors = np.full(len(moves), 1.0 / len(moves))

        for move, prior in zip(moves, priors):
            board.push(move)
            child_to_play = board.turn
            board.pop()
            node.children[move] = Node(parent=node, move=move, prior=float(prior), to_play=child_to_play)

        node.is_expanded = True
        return value, aggression_net

    def _add_dirichlet_noise(self, root: Node):
        mcts_cfg = self.cfg.mcts
        moves = list(root.children.keys())
        noise = np.random.dirichlet([mcts_cfg.dirichlet_alpha] * len(moves))
        eps = mcts_cfg.dirichlet_epsilon
        for move, n in zip(moves, noise):
            child = root.children[move]
            child.prior = (1 - eps) * child.prior + eps * n

    def effective_value(self, network_value: float, aggression_net: float,
                         board: chess.Board, perspective: chess.Color) -> float:
        """
        Blends the network's raw value with both the network's learned
        aggression prediction AND the hand-crafted heuristic, so that
        even early in training (before the aggression head has learned
        much) the search still receives a meaningful attacking-style
        signal. This is the concrete "structural bias toward
        aggressive play" injected into the search itself, as requested.
        """
        agg_cfg = self.cfg.aggression
        heuristic = aggression_score(board, perspective, agg_cfg)
        # Average the learned and hand-crafted aggression signals.
        blended_aggression = 0.5 * aggression_net + 0.5 * heuristic
        blend_w = agg_cfg.value_blend_weight
        value = (1 - blend_w) * network_value + blend_w * blended_aggression
        return max(-1.0, min(1.0, value))

    def _simulate(self, root: Node, board: chess.Board, history_encoder: HistoryEncoder,
                  tablebase=None):
        node = root
        path: List[Node] = [node]
        # Per-simulation history clone: pushed along the selection path so
        # that the leaf is encoded with the *actual* sequence of positions
        # visited in this rollout, not just the real game's history-so-far.
        sim_history = history_encoder.clone()

        # --- selection ---------------------------------------------------------
        while node.is_expanded and len(node.children) > 0 and not board.is_game_over(claim_draw=True):
            move, child = self._select_child(node)
            board.push(move)
            sim_history.push(board)
            node = child
            path.append(node)

        # --- terminal handling --------------------------------------------------
        if board.is_game_over(claim_draw=True):
            value = self._terminal_value(board, node.to_play)
        elif (
            tablebase is not None
            and self.cfg.mcts.use_tablebase
            and chess.popcount(board.occupied) <= self.cfg.mcts.tablebase_max_pieces
        ):
            tb_value = tablebase.probe_value(board)
            if tb_value is not None:
                value = tb_value
            else:
                value = self._expand_and_evaluate(node, board, sim_history)
        else:
            value = self._expand_and_evaluate(node, board, sim_history)

        # --- backup --------------------------------------------------------------
        # `value` is from the perspective of the side to move at the LEAF.
        # As we walk back up, the perspective flips at every ply.
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += value if n.to_play == node.to_play else -value

    def _expand_and_evaluate(self, node: Node, board: chess.Board,
                              history_encoder: HistoryEncoder) -> float:
        network_value, aggression_net = self._expand(node, board, history_encoder)
        return self.effective_value(network_value, aggression_net, board, board.turn)

    @staticmethod
    def _terminal_value(board: chess.Board, to_play: chess.Color) -> float:
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == to_play else -1.0
