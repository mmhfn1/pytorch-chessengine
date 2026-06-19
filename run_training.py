#!/usr/bin/env python3
"""
run_training.py
=================
The actual executable entry point for training the engine from
scratch (or resuming a previous run), e.g.:

    python run_training.py --output-dir ./run --iterations 200

This is a thin wrapper around `chess_engine.main.main()` — see that
module's docstring for the full self-play -> train -> arena-gate loop
this kicks off. Training always runs on GPU and will raise immediately
with a clear error if no CUDA device is available (see
`chess_engine/train.py: select_training_device`); self-play game
generation gracefully uses GPU if present and otherwise falls back to
CPU, since that only affects search speed, not correctness.
"""
from chess_engine.main import main

if __name__ == "__main__":
    main()
