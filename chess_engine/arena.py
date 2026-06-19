"""
arena.py
=========
Candidate-vs-champion gating, exactly matching AlphaZero's promotion
rule: before a freshly-trained network is allowed to replace the
network self-play actually uses to generate data, it must first beat
the current champion in a head-to-head match by at least
`cfg.arena.win_rate_to_promote` (draws count as half a win). This
guards against a single noisy training iteration silently regressing
the engine's playing strength.

Matches are played with a smaller, fixed MCTS simulation budget
(`cfg.arena.mcts_simulations`) and with Dirichlet root noise disabled,
since the goal here is an accurate strength comparison, not training
data diversity. Colors are alternated game-to-game so neither network
is unfairly favored by always playing White.
"""

from __future__ import annotations
import copy
import multiprocessing as mp
import os
import random
import shutil
import time
from dataclasses import dataclass
from typing import List, Optional

import torch
import chess

from .config import EngineConfig
from .encoder import HistoryEncoder
from .mcts import MCTS
from .network import ChessNet
from .tablebase import Tablebase


@dataclass
class ArenaResult:
    candidate_wins: int
    draws: int
    champion_wins: int
    total_games: int
    win_rate: float          # candidate's score fraction; draws count as 0.5
    promoted: bool


def _arena_cfg(cfg: EngineConfig) -> EngineConfig:
    """
    Returns a copy of `cfg` with MCTS reconfigured for arena play:
    fewer simulations (cheaper, since we need many games) and no
    tablebase-piece-count change. We deep-copy so this never mutates
    the caller's real training/self-play config.
    """
    arena_cfg = copy.deepcopy(cfg)
    arena_cfg.mcts.num_simulations = cfg.arena.mcts_simulations
    return arena_cfg


def play_one_arena_game(candidate: ChessNet, champion: ChessNet, cfg: EngineConfig,
                         device: torch.device, candidate_is_white: bool,
                         tablebase: Optional[Tablebase] = None) -> float:
    """
    Plays one game between `candidate` and `champion` and returns the
    result from the candidate's perspective: 1.0 (win), 0.5 (draw), or
    0.0 (loss). Both sides search with the same (arena-sized) MCTS
    budget and no Dirichlet noise, so the outcome reflects genuine
    relative strength rather than exploration randomness.
    """
    board = chess.Board()
    history = HistoryEncoder(cfg.network)
    history.push(board)

    mcts_candidate = MCTS(candidate, cfg, device)
    mcts_champion = MCTS(champion, cfg, device)

    ply = 0
    while not board.is_game_over(claim_draw=True) and ply < cfg.arena.max_plies:
        white_to_move = board.turn == chess.WHITE
        candidate_to_move = (white_to_move == candidate_is_white)
        mcts = mcts_candidate if candidate_to_move else mcts_champion

        root, visit_dist = mcts.run(board, history, add_root_noise=False, tablebase=tablebase)
        if not visit_dist:
            break  # no legal moves; is_game_over should already catch this
        # ply_count is set far beyond temperature_moves so move selection
        # always uses temperature_late: i.e. essentially greedy by visit
        # count, but with the small residual randomness needed so 400
        # arena games don't all collapse into a handful of duplicate lines.
        move = mcts.select_move(visit_dist, ply_count=10 ** 9)
        board.push(move)
        history.push(board)
        ply += 1

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.5
    candidate_won = (outcome.winner == chess.WHITE) == candidate_is_white
    return 1.0 if candidate_won else 0.0


def _load_net(weights_path: str, cfg: EngineConfig, device: torch.device) -> ChessNet:
    net = ChessNet(cfg.network).to(device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    # Accept either a plain state_dict or a full Trainer checkpoint dict.
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    net.load_state_dict(state_dict)
    net.eval()
    return net


def _arena_worker(candidate_path: str, champion_path: str, cfg: EngineConfig,
                   device_str: str, games_to_play: int, worker_id: int,
                   candidate_starts_white: bool, result_queue: "mp.Queue"):
    """Entry point run inside each arena worker process."""
    torch.manual_seed(worker_id * 7919 + int(time.time()) % 1000)
    random.seed(worker_id * 104729 + int(time.time()) % 1000)

    device = torch.device(device_str)
    candidate = _load_net(candidate_path, cfg, device)
    champion = _load_net(champion_path, cfg, device)
    tablebase = Tablebase(cfg.syzygy_path) if cfg.syzygy_path else None
    arena_cfg = _arena_cfg(cfg)

    candidate_white = candidate_starts_white
    for _ in range(games_to_play):
        score = play_one_arena_game(candidate, champion, arena_cfg, device, candidate_white, tablebase)
        # Reported one game at a time (rather than batched at the end)
        # so the parent process can print live match progress.
        result_queue.put(score)
        candidate_white = not candidate_white  # alternate colors every game

    if tablebase is not None:
        tablebase.close()


def run_arena_match(candidate_path: str, champion_path: str, cfg: EngineConfig,
                     num_games: Optional[int] = None, show_progress: bool = True) -> ArenaResult:
    """
    Plays `num_games` (default: cfg.arena.num_games) games between the
    candidate and champion checkpoints, split across worker processes
    (reusing cfg.selfplay.num_workers as the parallelism factor, since
    arena evaluation is just as embarrassingly parallel as self-play),
    and returns an `ArenaResult` with the aggregate score and whether
    the candidate reached the promotion threshold.

    Both `candidate_path` and `champion_path` must point to saved
    `state_dict`s (or full Trainer checkpoints) compatible with
    `cfg.network`.

    If `show_progress` is True (default), prints a live "games
    complete / running score" line to stdout as each arena game
    finishes.
    """
    num_games = num_games or cfg.arena.num_games
    num_workers = max(1, min(cfg.selfplay.num_workers, num_games))
    games_per_worker = [num_games // num_workers] * num_workers
    for i in range(num_games % num_workers):
        games_per_worker[i] += 1
    scheduled_games = sum(g for g in games_per_worker if g > 0)

    device_str = cfg.device if torch.cuda.is_available() else "cpu"

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []
    candidate_white = True
    for worker_id, n_games in enumerate(games_per_worker):
        if n_games == 0:
            continue
        p = ctx.Process(
            target=_arena_worker,
            args=(candidate_path, champion_path, cfg, device_str, n_games,
                  worker_id, candidate_white, result_queue),
        )
        p.start()
        processes.append(p)
        # Stagger which color each worker's first game starts with so
        # colors stay balanced overall even though each worker
        # alternates independently.
        candidate_white = not candidate_white

    all_scores: List[float] = []
    if scheduled_games > 0:
        if show_progress:
            print(f"[arena] starting {scheduled_games}-game match across {len(processes)} worker(s)...", flush=True)
        start_time = time.time()
        for _ in range(scheduled_games):
            score = result_queue.get()
            all_scores.append(score)
            if show_progress:
                games_done = len(all_scores)
                wins = sum(1 for s in all_scores if s == 1.0)
                draws = sum(1 for s in all_scores if s == 0.5)
                losses = sum(1 for s in all_scores if s == 0.0)
                win_rate_so_far = sum(all_scores) / games_done
                elapsed = time.time() - start_time
                rate = games_done / elapsed if elapsed > 0 else 0.0
                remaining = scheduled_games - games_done
                eta_s = remaining / rate if rate > 0 else float("inf")
                eta_str = f"{eta_s:.0f}s" if eta_s < float("inf") else "?"
                pct = 100.0 * games_done / scheduled_games
                print(
                    f"[arena] {games_done}/{scheduled_games} games complete ({pct:.0f}%) - "
                    f"candidate {wins}W-{draws}D-{losses}L (win_rate={win_rate_so_far:.3f}) - "
                    f"elapsed {elapsed:.0f}s - ETA {eta_str}",
                    flush=True,
                )
    for p in processes:
        p.join()

    candidate_wins = sum(1 for s in all_scores if s == 1.0)
    draws = sum(1 for s in all_scores if s == 0.5)
    champion_wins = sum(1 for s in all_scores if s == 0.0)
    total = len(all_scores)
    win_rate = (sum(all_scores) / total) if total > 0 else 0.0
    promoted = total > 0 and win_rate >= cfg.arena.win_rate_to_promote

    return ArenaResult(
        candidate_wins=candidate_wins,
        draws=draws,
        champion_wins=champion_wins,
        total_games=total,
        win_rate=win_rate,
        promoted=promoted,
    )


def promote_if_better(candidate_path: str, champion_path: str, cfg: EngineConfig,
                       num_games: Optional[int] = None) -> ArenaResult:
    """
    Runs the gating match and, if the candidate wins by enough of a
    margin, overwrites `champion_path` with the candidate's weights
    (i.e. the candidate becomes the new champion that future self-play
    workers and the UCI engine will load). Returns the `ArenaResult`
    either way so the caller (main.py's training loop) can log it.
    """
    result = run_arena_match(candidate_path, champion_path, cfg, num_games=num_games)
    if result.promoted:
        os.makedirs(os.path.dirname(champion_path) or ".", exist_ok=True)
        shutil.copyfile(candidate_path, champion_path)
    return result
