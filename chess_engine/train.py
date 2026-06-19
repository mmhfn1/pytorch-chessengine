"""
train.py
=========
Supervised training over self-play data. The loss is a 3-term
multi-task objective:

    L = policy_loss_weight   * CrossEntropy(policy_logits, pi)
      + value_loss_weight     * ValueLoss(value_head_out, z)
      + aux_loss_weight        * MSE(aggression_head_out, heuristic_score)

  - Policy loss: cross-entropy between the predicted move distribution
    and the MCTS visit distribution pi (soft targets, standard
    AlphaZero policy loss).
  - Value loss: if `use_wdl_head`, cross-entropy against a one-hot/
    soft (win, draw, loss) target derived from z; otherwise plain MSE
    against the scalar outcome z.
  - Aggression loss: MSE against the hand-crafted heuristic score
    (heuristics.aggression_score), implementing the "custom evaluation
    bias" as an auxiliary learning signal rather than a fixed
    hand-tuned constant — over time the network learns to recognize
    attacking patterns on its own.

L2 weight decay is applied via the optimizer (`weight_decay`), exactly
matching AlphaZero's training recipe (Adam/SGD + L2 regularization,
no separate explicit L2 term needed in the loss function itself).
"""

from __future__ import annotations
import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import EngineConfig
from .network import ChessNet
from .replay_buffer import ShardedReplayDataset


def select_training_device(cfg: EngineConfig) -> torch.device:
    """
    Training is required to run entirely on GPU — no silent CPU
    fallback. Self-play and inference (selfplay.py / uci.py) are
    allowed to degrade gracefully to CPU when no GPU is present,
    since that only affects search speed. Training is different:
    running gradient updates on CPU would be both impractically slow
    for a network this size and would silently hide a misconfigured
    GPU/driver/CUDA build that should be fixed before any real
    training compute is spent. So we fail loudly instead.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Training requires a CUDA-capable GPU, and none was detected "
            "(torch.cuda.is_available() is False). Training intentionally "
            "does not fall back to CPU. Check that a GPU is attached and "
            "that this environment's PyTorch build has CUDA support."
        )
    # Let cuDNN pick the fastest convolution algorithms for our fixed
    # (B, 119, 8, 8) input shape — meaningful speedup for a tower this
    # deep, with negligible one-time autotune cost.
    torch.backends.cudnn.benchmark = True
    device_str = cfg.device if cfg.device.startswith("cuda") else "cuda"
    return torch.device(device_str)


def value_target_to_wdl(z: torch.Tensor) -> torch.Tensor:
    """
    Converts a scalar outcome z in {-1, 0, 1} into a soft (win, draw,
    loss) target distribution suitable for cross-entropy against the
    WDL head's logits. z=1 -> [1,0,0], z=0 -> [0,1,0], z=-1 -> [0,0,1].
    """
    win = torch.clamp(z, min=0.0)
    loss = torch.clamp(-z, min=0.0)
    draw = 1.0 - win - loss
    return torch.stack([win, draw, loss], dim=-1)


class Trainer:
    def __init__(self, net: ChessNet, cfg: EngineConfig, device: torch.device,
                 require_cuda: bool = True):
        if require_cuda and device.type != "cuda":
            raise RuntimeError(
                f"Trainer was constructed with device={device}, but training "
                "is configured to run exclusively on GPU. Use "
                "select_training_device(cfg) to obtain a valid device, or "
                "pass require_cuda=False explicitly if you really intend to "
                "train on CPU (e.g. for a quick correctness test)."
            )
        self.net = net.to(device)
        self.cfg = cfg
        self.device = device
        self.train_cfg = cfg.train

        if self.train_cfg.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                net.parameters(),
                lr=self.train_cfg.learning_rate,
                momentum=self.train_cfg.momentum,
                weight_decay=self.train_cfg.weight_decay,
            )
        else:
            self.optimizer = torch.optim.Adam(
                net.parameters(),
                lr=self.train_cfg.learning_rate,
                weight_decay=self.train_cfg.weight_decay,
            )

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=list(self.train_cfg.lr_milestones),
            gamma=self.train_cfg.lr_decay,
        )

        self.global_step = 0

    def loss_fn(self, batch) -> dict:
        states, policy_targets, value_targets, aggression_targets = batch
        non_blocking = self.device.type == "cuda"
        states = states.to(self.device, non_blocking=non_blocking)
        policy_targets = policy_targets.to(self.device, non_blocking=non_blocking)
        value_targets = value_targets.to(self.device, non_blocking=non_blocking)
        aggression_targets = aggression_targets.to(self.device, non_blocking=non_blocking)

        out = self.net(states)

        # --- policy loss: soft cross-entropy against MCTS visit distribution ---
        log_probs = F.log_softmax(out["policy_logits"], dim=-1)
        policy_loss = -(policy_targets * log_probs).sum(dim=-1).mean()

        # --- value loss ----------------------------------------------------------
        if self.cfg.network.use_wdl_head:
            wdl_target = value_target_to_wdl(value_targets)
            log_wdl = F.log_softmax(out["value"], dim=-1)
            value_loss = -(wdl_target * log_wdl).sum(dim=-1).mean()
        else:
            value_pred = out["value"].squeeze(-1)
            value_loss = F.mse_loss(value_pred, value_targets)

        total = (
            self.train_cfg.policy_loss_weight * policy_loss
            + self.train_cfg.value_loss_weight * value_loss
        )

        losses = {"policy_loss": policy_loss, "value_loss": value_loss}

        # --- auxiliary aggression loss -------------------------------------------
        if self.cfg.network.use_aggression_head and "aggression" in out:
            agg_loss = F.mse_loss(out["aggression"], aggression_targets)
            total = total + self.cfg.aggression.aux_loss_weight * agg_loss
            losses["aggression_loss"] = agg_loss

        losses["total_loss"] = total
        return losses

    def train_step(self, batch) -> dict:
        self.net.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses = self.loss_fn(batch)
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.train_cfg.grad_clip_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1
        return {k: float(v.item()) for k, v in losses.items()}

    def train_iteration(self, replay_dir: str, num_steps: Optional[int] = None,
                          log_every: int = 50, max_shards: Optional[int] = None) -> dict:
        """
        Runs `num_steps` (default: cfg.train.train_steps_per_iteration)
        gradient steps, sampling minibatches via a DataLoader over the
        on-disk replay buffer shards. Returns a dict of the final
        smoothed loss values for logging/monitoring.
        """
        num_steps = num_steps or self.train_cfg.train_steps_per_iteration
        dataset = ShardedReplayDataset(replay_dir, max_shards=max_shards)
        if len(dataset) == 0:
            raise RuntimeError(f"No training data found in {replay_dir}")
        if len(dataset) < self.train_cfg.batch_size:
            raise RuntimeError(
                f"Replay dataset only has {len(dataset)} positions, which is "
                f"fewer than batch_size={self.train_cfg.batch_size}. With "
                "drop_last=True this would yield zero batches per epoch and "
                "loop forever with no progress. Either lower cfg.train."
                "batch_size or generate more self-play data before training."
            )

        loader = DataLoader(
            dataset,
            batch_size=self.train_cfg.batch_size,
            shuffle=True,
            num_workers=min(4, os.cpu_count() or 1),
            drop_last=True,
            persistent_workers=False,
            pin_memory=(self.device.type == "cuda"),
        )

        print(
            f"[train] starting {num_steps} steps on {len(dataset)} positions "
            f"(batch_size={self.train_cfg.batch_size}, device={self.device})...",
            flush=True,
        )

        running = {}
        step = 0
        t0 = time.time()
        while step < num_steps:
            for batch in loader:
                if step >= num_steps:
                    break
                losses = self.train_step(batch)
                for k, v in losses.items():
                    running[k] = 0.98 * running.get(k, v) + 0.02 * v
                step += 1
                if step % log_every == 0 or step == num_steps:
                    elapsed = time.time() - t0
                    rate = step / elapsed if elapsed > 0 else 0.0
                    remaining = num_steps - step
                    eta_s = remaining / rate if rate > 0 else 0.0
                    pct = 100.0 * step / num_steps
                    print(
                        f"[train] step {step}/{num_steps} ({pct:.0f}%) "
                        f"- {rate:.1f} steps/s - elapsed {elapsed:.1f}s - ETA {eta_s:.0f}s - " +
                        " ".join(f"{k}={v:.4f}" for k, v in running.items()),
                        flush=True,
                    )
        return running

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt["global_step"]
