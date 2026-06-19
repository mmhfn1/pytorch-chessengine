"""
chess_engine
============
A self-learning, AlphaZero-style chess engine tuned for aggressive,
attacking play, implemented with PyTorch + python-chess, and exposed
to the outside world (TCEC, GUIs, CLI play) via the UCI protocol.

By design the engine consults no opening book of human reference
moves: every move, from the very first ply, is chosen by its own
self-learned MCTS search over its own trained network. The only
non-learned component is the optional Syzygy endgame tablebase, which
supplies mathematically exact (not "book") play once an endgame is
small enough to be fully solved, exactly as top engines (including
self-learning ones) do once a position is no longer in dispute.

Sub-modules:
    config          - all tunable hyperparameters in one place
    move_encoding   - bijection between python-chess Move objects and
                       the flat 4672-way policy vector used by the net
    encoder         - board/history -> tensor encoding (the "planes")
    heuristics      - hand-crafted positional features used to build
                       the "aggression" auxiliary training signal
    network         - the ResNet (policy head + value head + aux head)
    mcts            - PUCT Monte-Carlo Tree Search
    replay_buffer   - on-disk / in-memory storage for self-play data
    selfplay        - self-play game generation (single + multiprocess)
    train            - supervised training step over self-play data,
                       always performed on GPU (see train.py)
    arena           - candidate-vs-champion gating match
    tablebase       - Syzygy endgame tablebase probing
    uci             - the UCI protocol engine loop
"""

__version__ = "1.0.0"
