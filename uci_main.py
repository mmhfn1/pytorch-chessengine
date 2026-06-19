#!/usr/bin/env python3
"""
uci_main.py
============
The actual executable entry point a UCI GUI or TCEC invokes to play
against this engine, e.g.:

    python uci_main.py --weights /path/to/champion.pt --syzygy /path/to/syzygy

(or, in a TCEC engine definition file, point the "command" field at
this script with the appropriate arguments).

This is a thin wrapper around `chess_engine.uci.main()`. It exists as
a separate top-level script — rather than asking users to remember
`python -m chess_engine.uci` — because `chess_engine/uci.py` uses
relative imports (as part of the `chess_engine` package) and so
cannot be executed directly as a standalone script.

No opening book is ever consulted by the engine launched here: every
move comes from the network/MCTS search loaded from `--weights`,
which in turn was produced entirely by self-play (see main.py /
run_training.py) with no human game data anywhere in the pipeline.
"""
from chess_engine.uci import main

if __name__ == "__main__":
    main()
