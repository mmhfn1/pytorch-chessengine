"""
config.py
=========
Single source of truth for every hyperparameter used across the
engine. Keeping configuration centralized makes it trivial to spin up
different training runs (e.g. a "balanced" vs. an "aggressive"
profile) without touching algorithmic code.

All dataclasses are plain, picklable, and safe to pass to worker
processes during multiprocessed self-play.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NetworkConfig:
    # --- input encoding -----------------------------------------------
    history_length: int = 8          # T: number of past board positions stacked
    planes_per_position: int = 14    # 6 piece types x 2 colors + 2 repetition planes
    aux_planes: int = 7              # side-to-move, move count, 4x castling, no-progress
    input_channels: int = field(init=False)

    # --- backbone --------------------------------------------------------
    num_filters: int = 256
    num_residual_blocks: int = 19    # AlphaZero used 19 (small) / 39 (large)
    se_blocks: bool = True           # squeeze-and-excitation channel attention

    # --- policy head -----------------------------------------------------
    policy_size: int = 4672          # 64 squares x 73 move-planes

    # --- value head --------------------------------------------------------
    value_hidden_size: int = 256
    use_wdl_head: bool = True        # predict (win, draw, loss) instead of scalar [-1, 1]

    # --- auxiliary "aggression" head --------------------------------------
    # A small extra head trained (with its own loss term) to predict a
    # hand-crafted attacking/aggression score for the position. Its
    # output is blended into the value used by MCTS so that the search
    # is nudged toward sharp, attacking continuations rather than only
    # the win/draw/loss expectation.
    use_aggression_head: bool = True
    aggression_hidden_size: int = 128

    def __post_init__(self):
        self.input_channels = (
            self.history_length * self.planes_per_position + self.aux_planes
        )


@dataclass
class AggressionConfig:
    """
    Weights for the hand-crafted "attacking style" bias. These combine
    into a single scalar heuristic score in [-1, 1] (heuristics.py)
    which is (a) used as an auxiliary training target and (b) blended
    into the value estimate used inside MCTS via `value_blend_weight`.
    """
    material_activity_weight: float = 0.20   # reward active piece placement
    king_safety_weight: float = 0.35          # reward exposing the *enemy* king
    sac_potential_weight: float = 0.25        # reward sound sacrifices for initiative
    mobility_weight: float = 0.20             # reward higher piece mobility

    # How much the heuristic aggression score perturbs the MCTS value
    # of a leaf, on top of the network's own value prediction:
    #   v_eff = (1 - blend) * v_network + blend * aggression_score
    value_blend_weight: float = 0.12

    # Training loss weight for the auxiliary aggression head.
    aux_loss_weight: float = 0.15


@dataclass
class MCTSConfig:
    num_simulations: int = 800
    c_puct_base: float = 19652.0      # AlphaZero's c_puct schedule constants
    c_puct_init: float = 1.25
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25   # fraction of noise mixed into root priors
    fpu_reduction: float = 0.25       # "first play urgency" reduction for unvisited children
    virtual_loss: float = 1.0         # used during multi-threaded / batched search
    temperature_moves: int = 30        # ply count after which temperature -> ~0 (greedy)
    temperature_early: float = 1.0
    temperature_late: float = 0.1
    use_tablebase: bool = True
    tablebase_max_pieces: int = 7


@dataclass
class SelfPlayConfig:
    games_per_iteration: int = 5000
    max_game_plies: int = 512
    num_workers: int = 8
    resign_threshold: Optional[float] = -0.95  # resign if value < threshold for resign_consecutive plies
    resign_consecutive: int = 4
    resign_disable_fraction: float = 0.1       # disable resignation in 10% of games (to keep resign-threshold honest)


@dataclass
class TrainConfig:
    batch_size: int = 1024
    learning_rate: float = 2e-2
    lr_milestones: tuple = (100_000, 300_000, 500_000)  # training-step milestones
    lr_decay: float = 0.1
    momentum: float = 0.9             # used if optimizer == "sgd"
    optimizer: str = "adam"           # "adam" or "sgd"
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    train_steps_per_iteration: int = 1000
    checkpoint_dir: str = "./checkpoints"
    replay_buffer_dir: str = "./replay_buffer"
    replay_buffer_max_positions: int = 2_000_000
    grad_clip_norm: float = 5.0


@dataclass
class ArenaConfig:
    num_games: int = 400
    win_rate_to_promote: float = 0.55   # candidate must reach this (draws count half)
    mcts_simulations: int = 400         # cheaper search for faster gating matches
    max_plies: int = 400


@dataclass
class EngineConfig:
    """Top level aggregate config, what most modules import."""
    network: NetworkConfig = field(default_factory=NetworkConfig)
    aggression: AggressionConfig = field(default_factory=AggressionConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    arena: ArenaConfig = field(default_factory=ArenaConfig)

    # NOTE: self-play, MCTS search, and inference may fall back to CPU
    # automatically if no GPU is present (see selfplay.py / uci.py).
    # Training never does this — see train.py's `select_training_device`,
    # which raises rather than silently training on CPU.
    device: str = "cuda"
    syzygy_path: Optional[str] = None          # directory containing .rtbw/.rtbz files
    # Intentionally no `opening_book_path`: the engine is designed to be
    # completely self-learning and never consults human opening-book
    # moves. Every move, including the first, comes from its own MCTS
    # search over its own trained network.


DEFAULT_CONFIG = EngineConfig()
