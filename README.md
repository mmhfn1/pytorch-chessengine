# Self-Learning Chess Engine

An AlphaZero-style chess engine, implemented from scratch in PyTorch + python-chess,
tuned for aggressive/attacking play, exposed to GUIs and tournament managers (including
TCEC) via the UCI protocol.

The engine is **completely self-learning**: it consults no opening book and no human
game data anywhere in the pipeline. Every move it has ever played — including the
opening move of the very first self-play game — was chosen by its own MCTS search over
its own neural network. The only non-learned component is an *optional* Syzygy endgame
tablebase, which supplies mathematically exact play once a position is small enough to
be fully solved — that's arithmetic, not "book knowledge."

## Architecture

```
chess_engine/
    config.py          all tunable hyperparameters (one dataclass tree)
    move_encoding.py   chess.Move <-> flat 4672-way policy index bijection
    encoder.py         board/history -> (119, 8, 8) input tensor
    heuristics.py       hand-crafted "aggression" scoring (material activity,
                        king safety differential, sacrifice potential, mobility)
    network.py          ResNet trunk (SE blocks) + policy/value/aggression heads
    mcts.py             PUCT search, with a fixed-simulation mode (run, used by
                        self-play) and a time-budgeted/stoppable mode (run_timed,
                        used by the UCI loop)
    replay_buffer.py     in-memory + on-disk (sharded .npz) self-play data storage
    selfplay.py          single-process and multiprocess self-play game generation
    train.py             supervised training step (policy + value + aggression
                        multi-task loss), GPU-only (see below)
    arena.py             candidate-vs-champion gating match before promotion
    tablebase.py         Syzygy endgame tablebase probing (optional)
    uci.py              the UCI protocol engine loop
    main.py             self-play -> train -> arena-gate orchestration loop

uci_main.py            top-level executable: what a GUI/TCEC actually launches
run_training.py        top-level executable: what you run to train the engine
requirements.txt
README.md
```

### How the "aggressive style" bias is implemented

Three reinforcing layers, all driven by `heuristics.aggression_score` (material
activity, king-safety differential favoring exposing the *enemy* king, sound-sacrifice
detection, and mobility):

1. **Auxiliary training signal** — `network.AggressionHead` is trained (via an MSE loss
   term in `train.py`) to predict the hand-crafted heuristic directly from the trunk's
   features, forcing the network to represent attacking-relevant concepts explicitly
   rather than leaving them implicit in the value head alone.
2. **Search-time value blending** — `mcts.MCTS.effective_value` blends the network's own
   (learned) aggression prediction and the hand-crafted heuristic into the value MCTS
   actually backs up at every leaf (`cfg.aggression.value_blend_weight`, default 0.12),
   so the search itself is nudged toward sharper, more attacking lines — not just the
   final training targets.
3. **Runtime tuning** — the UCI `Aggression` option (0-200, default 100) scales this
   blend weight live, letting an operator dial the style up or down without retraining.

### Why no opening book

This was a deliberate design choice: the engine plays its openings the same way it
plays everything else, by searching with its own trained network. This keeps the
entire system "tabula rasa" in the AlphaZero sense — its strength is a direct measure
of what the self-play/training loop has actually learned, with no human opening theory
papering over gaps in that learning.

### Training is GPU-only, by design

`train.py:select_training_device` raises immediately and explicitly if no CUDA device
is available — training never silently falls back to CPU. This is intentional: training
this network on CPU would be impractically slow, and a silent fallback would mask a
misconfigured GPU/driver/CUDA environment that should be fixed before any real training
compute is spent. `Trainer.__init__` itself also defaults to enforcing this
(`require_cuda=True`); pass `require_cuda=False` explicitly only if you really want a
CPU correctness test (e.g. in a sandbox with no GPU).

Self-play game generation and UCI search/inference are different: they automatically
use a GPU if present and otherwise fall back to CPU, since that only affects search
speed, not correctness — an engine that refused to run at all on GPU-less hardware
would be needlessly restrictive.

## Setup

```bash
pip install -r requirements.txt
```

Optional: download [Syzygy tablebases](https://syzygy-tables.info/) if you want exact
endgame play, and point `--syzygy /path/to/tables` at them (or the UCI `SyzygyPath`
option) when running the engine.

## Training

```bash
python run_training.py --output-dir ./run --iterations 200
```

Requires a CUDA GPU (see above). Each iteration: self-play games are generated with the
current champion network, a candidate is trained on the most recent self-play data, and
the candidate must beat the champion in a gating match
(`cfg.arena.win_rate_to_promote`, default 55%, draws count half) to be promoted. Useful
flags:

- `--games-per-iteration N` — override `cfg.selfplay.games_per_iteration`
- `--train-steps N` — override `cfg.train.train_steps_per_iteration`
- `--max-shards N` — train on only the most recent N replay-buffer shards (sliding
  window over self-play history, rather than every game ever played)

`./run/champion.pt` is the network the UCI engine should be pointed at; it's a plain
`state_dict`, updated in place every time a candidate is promoted.

## Playing / TCEC integration

```bash
python uci_main.py --weights ./run/champion.pt --syzygy /path/to/syzygy
```

Point any UCI-speaking GUI or tournament manager (TCEC, cutechess-cli, Arena, etc.) at
`uci_main.py` with these arguments as the engine command. Supports `uci`, `isready`,
`ucinewgame`, `position` (`startpos` or `fen`, with `moves`), `go` (with `wtime`/`btime`/
`winc`/`binc`/`movestogo`, `movetime`, `nodes`, `infinite`, `ponder`/`ponderhit`,
`searchmoves`), `stop`, `setoption`, and `quit`. `go` runs in a background thread so the
engine keeps responding to `isready`/`stop` while searching, as real-time play requires.

Known, documented approximations: `go depth N` is translated into a simulation-count
budget (MCTS has no native notion of ply-depth), and the `score cp` reported in `info`
lines is a smooth monotonic transform of the network's win-probability-style value, not
a literal material-based centipawn count — both are clearly commented in `uci.py`.

## Testing components individually

Every module was built and unit/smoke-tested independently during development (move
encoding round-trips, network shape/parameter checks, MCTS simulation counts, self-play
game generation, training loss convergence, arena gating with identical networks
producing draws as expected, and a full subprocess-piped UCI session). There is no
separate test suite file; re-run the same checks by importing any module and exercising
its public functions/classes directly, as shown throughout the module docstrings.
