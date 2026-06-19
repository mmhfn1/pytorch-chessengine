"""
main.py
========
Top-level orchestration of the full AlphaZero-style training loop:

    repeat:
        1. SELF-PLAY  -- the current champion network plays games
           against itself (run_selfplay_parallel), guided by MCTS,
           producing fresh training data. No human game data and no
           opening book are ever used: the very first game of the
           very first iteration is already 100% self-generated.
        2. TRAIN        -- a candidate network (a copy of the champion)
           is trained on the most recent self-play data. This step
           always runs on GPU (see train.py's `select_training_device`)
           and never silently falls back to CPU.
        3. ARENA GATE  -- the candidate plays a head-to-head match
           against the current champion (arena.py). Only if it wins by
           at least `cfg.arena.win_rate_to_promote` does it replace the
           champion that future self-play/training/UCI play will use.

This file is the thing you actually run to train the engine from
scratch (or resume an existing run); `uci.py` / `uci_main.py` is the
separate, lightweight entry point used to actually *play* with
whatever champion this loop has produced so far.
"""

from __future__ import annotations
import argparse
import os
import shutil
import time

import torch

from .arena import run_arena_match
from .config import EngineConfig
from .network import ChessNet
from .selfplay import run_selfplay_parallel
from .train import Trainer, select_training_device

CHAMPION_FILENAME = "champion.pt"
CANDIDATE_FILENAME = "candidate.pt"


def _bootstrap_champion(cfg: EngineConfig, champion_path: str):
    """
    Creates the very first champion checkpoint: a freshly, randomly
    initialized network. There is deliberately no pretraining on
    human games and no opening book anywhere in this pipeline — every
    bit of strength the engine ever gains comes from its own self-play
    starting from this random network, exactly matching AlphaZero's
    "tabula rasa" approach.
    """
    os.makedirs(os.path.dirname(champion_path) or ".", exist_ok=True)
    net = ChessNet(cfg.network)
    torch.save(net.state_dict(), champion_path)
    print(f"[main] bootstrapped a fresh, randomly-initialized champion at {champion_path}", flush=True)


def run_training_pipeline(cfg: EngineConfig, output_dir: str, num_iterations: int,
                            train_steps_per_iteration: int = None,
                            games_per_iteration: int = None,
                            max_shards: int = None,
                            start_iteration: int = 0):
    """
    Runs `num_iterations` full self-play -> train -> arena-gate cycles.

    `output_dir` layout:
        champion.pt              - current best network (plain state_dict)
        candidate.pt              - scratch space for the network being trained/evaluated
        replay_buffer/            - self-play data shards (grows over time)
        checkpoints/trainer_iter_N.pt  - full optimizer-inclusive checkpoints, for resuming
    """
    os.makedirs(output_dir, exist_ok=True)
    champion_path = os.path.join(output_dir, CHAMPION_FILENAME)
    candidate_path = os.path.join(output_dir, CANDIDATE_FILENAME)
    replay_dir = os.path.join(output_dir, "replay_buffer")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if not os.path.exists(champion_path):
        _bootstrap_champion(cfg, champion_path)

    # Training itself always runs on GPU -- raises immediately and
    # clearly if no CUDA device is available, rather than silently
    # training on CPU.
    train_device = select_training_device(cfg)
    print(f"[main] training device: {train_device}", flush=True)
    print(f"[main] starting run: {num_iterations} iteration(s), output_dir={output_dir}", flush=True)

    pipeline_start = time.time()
    for loop_idx, it in enumerate(range(start_iteration, start_iteration + num_iterations)):
        iter_start = time.time()
        iters_done_so_far = loop_idx  # completed iterations before this one
        if iters_done_so_far > 0:
            avg_iter_time = (iter_start - pipeline_start) / iters_done_so_far
            remaining_iters = num_iterations - iters_done_so_far
            eta_s = avg_iter_time * remaining_iters
            eta_str = f"{eta_s / 60:.1f} min"
        else:
            eta_str = "?"
        pct = 100.0 * loop_idx / num_iterations
        print(
            f"\n===== Iteration {it} ({loop_idx + 1}/{num_iterations}, {pct:.0f}% of this run) "
            f"- run elapsed {(iter_start - pipeline_start) / 60:.1f} min - ETA {eta_str} =====",
            flush=True,
        )

        # ---- 1. self-play (inference device auto-selects GPU if present, else CPU) ----
        t0 = time.time()
        run_selfplay_parallel(
            weights_path=champion_path,
            cfg=cfg,
            output_dir=replay_dir,
            total_games=games_per_iteration,
        )
        print(f"[main] self-play finished in {time.time() - t0:.1f}s", flush=True)

        # ---- 2. train a candidate, starting from the current champion's weights ----
        candidate_net = ChessNet(cfg.network).to(train_device)
        champion_state = torch.load(champion_path, map_location=train_device, weights_only=True)
        candidate_net.load_state_dict(champion_state)

        trainer = Trainer(candidate_net, cfg, train_device)
        trainer_checkpoint_path = os.path.join(checkpoint_dir, f"trainer_iter_{it}.pt")
        prev_trainer_checkpoint = os.path.join(checkpoint_dir, f"trainer_iter_{it - 1}.pt")
        if os.path.exists(prev_trainer_checkpoint):
            # Resume optimizer/scheduler state across iterations rather
            # than restarting Adam's moment estimates from scratch
            # every single iteration.
            trainer.load_checkpoint(prev_trainer_checkpoint)

        t0 = time.time()
        losses = trainer.train_iteration(
            replay_dir,
            num_steps=train_steps_per_iteration,
            max_shards=max_shards,
        )
        print(f"[main] training finished in {time.time() - t0:.1f}s, final losses: {losses}", flush=True)
        trainer.save_checkpoint(trainer_checkpoint_path)
        torch.save(candidate_net.state_dict(), candidate_path)

        # ---- 3. arena gate: candidate must beat the champion to be promoted ----
        t0 = time.time()
        result = run_arena_match(candidate_path, champion_path, cfg)
        print(
            f"[main] arena result: {result.candidate_wins}W-{result.draws}D-"
            f"{result.champion_wins}L (win_rate={result.win_rate:.3f}, "
            f"threshold={cfg.arena.win_rate_to_promote:.3f}) in {time.time() - t0:.1f}s",
            flush=True,
        )
        if result.promoted:
            shutil.copyfile(candidate_path, champion_path)
            print(f"[main] candidate PROMOTED -> new champion at {champion_path}", flush=True)
        else:
            print("[main] candidate did not reach the promotion threshold; champion unchanged", flush=True)

        print(f"[main] iteration {it} finished in {(time.time() - iter_start) / 60:.1f} min", flush=True)

    total_min = (time.time() - pipeline_start) / 60
    print(f"\n[main] run complete: {num_iterations} iteration(s) in {total_min:.1f} min total", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Self-play / train / arena-gate training loop.")
    parser.add_argument("--output-dir", type=str, default="./run", help="Directory for champion/candidate weights, replay buffer, and checkpoints.")
    parser.add_argument("--iterations", type=int, default=100, help="Number of self-play/train/gate cycles to run.")
    parser.add_argument("--train-steps", type=int, default=None, help="Override cfg.train.train_steps_per_iteration.")
    parser.add_argument("--games-per-iteration", type=int, default=None, help="Override cfg.selfplay.games_per_iteration.")
    parser.add_argument("--max-shards", type=int, default=None, help="Train on only the most recent N replay-buffer shards (sliding window).")
    parser.add_argument("--start-iteration", type=int, default=0, help="Iteration counter to start from (for resuming, affects checkpoint filenames only).")
    args = parser.parse_args()

    cfg = EngineConfig()
    run_training_pipeline(
        cfg,
        output_dir=args.output_dir,
        num_iterations=args.iterations,
        train_steps_per_iteration=args.train_steps,
        games_per_iteration=args.games_per_iteration,
        max_shards=args.max_shards,
        start_iteration=args.start_iteration,
    )


if __name__ == "__main__":
    main()
