"""
replay_buffer.py
==================
Storage for self-play training examples. Each example is a tuple:

    (state_tensor, policy_target, value_target, aggression_target)

  state_tensor      : (C, 8, 8) float32 — the encoded board (encoder.py)
  policy_target     : (4672,) float32   — the MCTS visit distribution (pi)
  value_target      : float             — final game outcome z in {-1, 0, 1}
                        from the perspective of the player to move in
                        state_tensor
  aggression_target : float             — heuristics.aggression_score(...)
                        computed at that position, used to train the
                        auxiliary AggressionHead

The buffer supports:
  - in-memory ring-buffer behaviour (bounded by `max_positions`)
  - sharded disk persistence (`.npz` files) so multiple self-play
    worker processes can each flush independently and a single
    trainer process can stream from disk without holding everything
    in RAM at once.
"""

from __future__ import annotations
import os
import glob
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import TrainConfig


@dataclass
class Example:
    state: np.ndarray          # (C, 8, 8) float32
    policy: np.ndarray          # (4672,) float32
    value: float
    aggression: float


class ReplayBuffer:
    """In-memory ring buffer; used directly inside a single process, or
    as the staging area before `flush_to_disk()` for multiprocess setups."""

    def __init__(self, max_positions: int):
        self.max_positions = max_positions
        self._states: List[np.ndarray] = []
        self._policies: List[np.ndarray] = []
        self._values: List[float] = []
        self._aggressions: List[float] = []

    def __len__(self) -> int:
        return len(self._states)

    def add_game(self, examples: List[Example]):
        for ex in examples:
            self._states.append(ex.state)
            self._policies.append(ex.policy)
            self._values.append(ex.value)
            self._aggressions.append(ex.aggression)
        self._truncate()

    def _truncate(self):
        overflow = len(self._states) - self.max_positions
        if overflow > 0:
            self._states = self._states[overflow:]
            self._policies = self._policies[overflow:]
            self._values = self._values[overflow:]
            self._aggressions = self._aggressions[overflow:]

    def flush_to_disk(self, directory: str) -> str:
        """Writes the current contents as one compressed .npz shard and clears the in-memory buffer."""
        os.makedirs(directory, exist_ok=True)
        if len(self._states) == 0:
            return ""
        shard_name = f"shard_{int(time.time())}_{uuid.uuid4().hex[:8]}.npz"
        path = os.path.join(directory, shard_name)
        np.savez_compressed(
            path,
            states=np.stack(self._states).astype(np.float32),
            policies=np.stack(self._policies).astype(np.float32),
            values=np.array(self._values, dtype=np.float32),
            aggressions=np.array(self._aggressions, dtype=np.float32),
        )
        self._states.clear()
        self._policies.clear()
        self._values.clear()
        self._aggressions.clear()
        return path


class ShardedReplayDataset(Dataset):
    """
    A torch Dataset that lazily loads positions across all `.npz`
    shards found in `directory`, suitable for wrapping in a
    `torch.utils.data.DataLoader` with multiple worker processes for
    fast, parallel batch construction during training.

    Pass `max_shards` to only train on the most recent N shards (a
    simple way to implement a sliding "recent self-play data" window
    without re-reading everything from the start of training).
    """

    def __init__(self, directory: str, max_shards: Optional[int] = None):
        self.directory = directory
        shard_paths = sorted(
            glob.glob(os.path.join(directory, "shard_*.npz")),
            key=os.path.getmtime,
        )
        if max_shards is not None:
            shard_paths = shard_paths[-max_shards:]
        self.shard_paths = shard_paths

        self._shard_sizes: List[int] = []
        self._cumulative: List[int] = [0]
        for p in self.shard_paths:
            with np.load(p) as data:
                n = data["values"].shape[0]
            self._shard_sizes.append(n)
            self._cumulative.append(self._cumulative[-1] + n)

        self._cache_idx: Optional[int] = None
        self._cache_data = None

    def __len__(self) -> int:
        return self._cumulative[-1] if self._cumulative else 0

    def _locate(self, global_idx: int) -> Tuple[int, int]:
        # binary search over cumulative sizes
        lo, hi = 0, len(self._shard_sizes) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._cumulative[mid + 1] <= global_idx:
                lo = mid + 1
            else:
                hi = mid
        local_idx = global_idx - self._cumulative[lo]
        return lo, local_idx

    def __getitem__(self, idx: int):
        shard_idx, local_idx = self._locate(idx)
        if shard_idx != self._cache_idx:
            self._cache_data = np.load(self.shard_paths[shard_idx])
            self._cache_idx = shard_idx
        data = self._cache_data
        state = torch.from_numpy(data["states"][local_idx].copy())
        policy = torch.from_numpy(data["policies"][local_idx].copy())
        value = torch.tensor(float(data["values"][local_idx]), dtype=torch.float32)
        aggression = torch.tensor(float(data["aggressions"][local_idx]), dtype=torch.float32)
        return state, policy, value, aggression
