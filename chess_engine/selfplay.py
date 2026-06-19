"""
selfplay.py
============
Generates training data by having the current network play full games
against itself, guided by MCTS. Each ply records:

    (encoded board state, MCTS visit distribution, side-to-move)

and once the game concludes, every recorded position is back-filled
with the actual outcome `z` (+1 / 0 / -1, from that position's own
side-to-move perspective) and the hand-crafted aggression score,
producing ready-to-train `Example`s (replay_buffer.py).

Parallelization: `run_selfplay_parallel` spins up
`cfg.selfplay.num_workers` separate OS processes (via
`multiprocessing`), each independently generating full games and
flushing finished shards straight to disk, so the trainer process
never blocks on game generation and can simply stream whatever
shards exist in the replay-buffer directory.

Resignation: optionally allows a game to resign early once the
network is confidently and consistently losing, to avoid wasting
compute simulating hopeless, already-decided positions to checkmate —
matching standard AlphaZero-style self-play practice. A fraction of
games disable resignation entirely so the resign threshold's accuracy
can be continually verified (false resignations would otherwise
silently corrupt the value head's training signal).
"""

from __future__ import annotations
import multiprocessing as mp
import os
import random
import time
from typing import List, Optional

import numpy as np
import torch
import chess

from .config import EngineConfig
from .encoder import HistoryEncoder
from .heuristics import aggression_score
from .mcts import MCTS
from .move_encoding import POLICY_SIZE, move_to_index
from .network import ChessNet
from .replay_buffer import Example, ReplayBuffer
from .tablebase import Tablebase


class GameRecord:
    """Accumulates raw per-ply data for a single game before outcome backfill."""

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.policies: List[np.ndarray] = []
        self.to_plays: List[chess.Color] = []
        self.aggressions: List[float] = []

    def add_ply(self, state: np.ndarray, policy: np.ndarray, to_play: chess.Color, aggression: float):
        self.states.append(state)
        self.policies.append(policy)
        self.to_plays.append(to_play)
        self.aggressions.append(aggression)

    def finalize(self, winner: Optional[chess.Color]) -> List[Example]:
        examples = []
        for state, policy, to_play, agg in zip(self.states, self.policies, self.to_plays, self.aggressions):
            if winner is None:
                z = 0.0
            else:
                z = 1.0 if winner == to_play else -1.0
            examples.append(Example(state=state, policy=policy, value=z, aggression=agg))
        return examples


def play_one_game(net: ChessNet, cfg: EngineConfig, device: torch.device,
                   tablebase: Optional[Tablebase] = None) -> List[Example]:
    """Plays a single self-play game to completion and returns its training examples."""
    board = chess.Board()
    history = HistoryEncoder(cfg.network)
    history.push(board)

    mcts = MCTS(net, cfg, device)
    record = GameRecord()

    resign_disabled = random.random() < cfg.selfplay.resign_disable_fraction
    consecutive_losing_plies = 0
    ply = 0
    winner: Optional[chess.Color] = None

    while not board.is_game_over(claim_draw=True) and ply < cfg.selfplay.max_game_plies:
        root, visit_dist = mcts.run(board, history, add_root_noise=True, tablebase=tablebase)

        # Build the full 4672-length policy target (zeros for moves never visited).
        policy_target = np.zeros(POLICY_SIZE, dtype=np.float32)
        for move, frac in visit_dist.items():
            idx = move_to_index(move, board)
            if idx is not None:
                policy_target[idx] = frac

        state_tensor = history.encode(board)
        agg = aggression_score(board, board.turn, cfg.aggression)
        record.add_ply(state_tensor, policy_target, board.turn, agg)

        # --- resignation check (based on the root's own value estimate) -------
        if cfg.selfplay.resign_threshold is not None and not resign_disabled:
            if root.value < cfg.selfplay.resign_threshold:
                consecutive_losing_plies += 1
            else:
                consecutive_losing_plies = 0
            if consecutive_losing_plies >= cfg.selfplay.resign_consecutive:
                winner = not board.turn  # the side about to move is resigning
                break

        move = mcts.select_move(visit_dist, ply_count=ply)
        board.push(move)
        history.push(board)
        ply += 1

    if winner is None and board.is_game_over(claim_draw=True):
        outcome = board.outcome(claim_draw=True)
        winner = outcome.winner if outcome is not None else None

    return record.finalize(winner)


def _worker_loop(weights_path: str, cfg: EngineConfig, device_str: str,
                  output_dir: str, games_to_play: int, worker_id: int,
                  shard_flush_every: int, progress_queue: Optional["mp.Queue"] = None):
    """Entry point run inside each self-play worker process."""
    torch.manual_seed(worker_id * 7919 + int(time.time()) % 1000)
    random.seed(worker_id * 104729 + int(time.time()) % 1000)

    device = torch.device(device_str)
    net = ChessNet(cfg.network).to(device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()

    tablebase = Tablebase(cfg.syzygy_path) if cfg.syzygy_path else None
    buffer = ReplayBuffer(max_positions=10 ** 9)  # effectively unbounded local staging area

    games_since_flush = 0
    for _ in range(games_to_play):
        examples = play_one_game(net, cfg, device, tablebase)
        buffer.add_game(examples)
        games_since_flush += 1
        if progress_queue is not None:
            # One message per finished game, so the parent process can
            # print a live "N/total games complete" progress line.
            progress_queue.put((worker_id, len(examples)))
        if games_since_flush >= shard_flush_every:
            buffer.flush_to_disk(output_dir)
            games_since_flush = 0

    buffer.flush_to_disk(output_dir)
    if tablebase is not None:
        tablebase.close()


def run_selfplay_parallel(weights_path: str, cfg: EngineConfig,
                           output_dir: str, total_games: Optional[int] = None,
                           shard_flush_every: int = 10, show_progress: bool = True):
    """
    Spawns `cfg.selfplay.num_workers` worker processes, each playing
    roughly `total_games / num_workers` games (defaulting to
    `cfg.selfplay.games_per_iteration` if `total_games` is None), and
    writes finished shards directly to `output_dir`.

    `weights_path` must be a path to a saved `state_dict` (see
    train.py's checkpointing) so each worker process can independently
    load its own copy of the network — required because CUDA contexts
    and PyTorch modules generally don't survive a `fork()` cleanly.

    If `show_progress` is True (default), prints a live "games
    complete / rate / ETA" line to stdout as workers finish games, so
    self-play progress can be monitored during what is often the
    longest-running stage of a training iteration.
    """
    total_games = total_games or cfg.selfplay.games_per_iteration
    num_workers = max(1, cfg.selfplay.num_workers)
    games_per_worker = [total_games // num_workers] * num_workers
    for i in range(total_games % num_workers):
        games_per_worker[i] += 1
    scheduled_games = sum(g for g in games_per_worker if g > 0)

    device_str = cfg.device if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    ctx = mp.get_context("spawn")  # "spawn" is required for safe CUDA use in workers
    progress_queue = ctx.Queue() if show_progress else None
    processes = []
    for worker_id, n_games in enumerate(games_per_worker):
        if n_games == 0:
            continue
        p = ctx.Process(
            target=_worker_loop,
            args=(weights_path, cfg, device_str, output_dir, n_games, worker_id,
                  shard_flush_every, progress_queue),
        )
        p.start()
        processes.append(p)

    if show_progress and scheduled_games > 0:
        print(f"[selfplay] starting {scheduled_games} games across {len(processes)} worker(s)...", flush=True)
        start_time = time.time()
        games_done = 0
        total_plies = 0
        while games_done < scheduled_games:
            _worker_id, num_plies = progress_queue.get()
            games_done += 1
            total_plies += num_plies
            elapsed = time.time() - start_time
            rate = games_done / elapsed if elapsed > 0 else 0.0
            remaining = scheduled_games - games_done
            eta_s = remaining / rate if rate > 0 else float("inf")
            eta_str = f"{eta_s:.0f}s" if eta_s < float("inf") else "?"
            pct = 100.0 * games_done / scheduled_games
            avg_plies = total_plies / games_done
            print(
                f"[selfplay] {games_done}/{scheduled_games} games complete "
                f"({pct:.0f}%) - {rate * 60:.1f} games/min - avg {avg_plies:.0f} plies/game "
                f"- elapsed {elapsed:.0f}s - ETA {eta_str}",
                flush=True,
            )

    for p in processes:
        p.join()
