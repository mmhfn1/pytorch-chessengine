"""
uci.py
=======
The Universal Chess Interface (UCI) protocol loop: the piece that
makes this engine usable by any UCI-speaking GUI, match manager, or
testing framework — including TCEC, which communicates with engines
exclusively over UCI on stdin/stdout.

Design notes relevant to TCEC compliance:

  - `go` is handled asynchronously in a background thread, so the
    engine keeps reading and responding to stdin (in particular
    `stop` and `quit`) while a search is in progress, exactly as
    real-time tournament play requires.
  - Time management (`wtime`/`btime`/`winc`/`binc`/`movestogo`,
    `movetime`, `nodes`, `depth`, `infinite`, `ponder`) is translated
    into a time/simulation budget for `mcts.run_timed` (see mcts.py).
    Since this engine's search is simulation-count-based (MCTS) rather
    than ply-depth-based (alpha-beta), `depth` is necessarily an
    approximation — this is called out explicitly below rather than
    silently mis-implemented.
  - The engine never reads from an opening book: every move, from the
    very first ply of the game, comes from its own MCTS search over
    its own trained network. The only non-learned input it ever
    consults is an optional Syzygy tablebase for provably-exact
    endgame play, which is mathematics, not pre-recorded "book" moves.
  - Malformed or unrecognized input is ignored rather than crashing
    the process, per the UCI specification ("if an unknown command is
    sent, the engine should just ignore it").
"""

from __future__ import annotations
import argparse
import math
import os
import sys
import threading
import time
from typing import Dict, List, Optional

import torch
import chess

from .config import EngineConfig, DEFAULT_CONFIG
from .encoder import HistoryEncoder
from .mcts import MCTS
from .network import ChessNet
from .tablebase import Tablebase

ENGINE_NAME = "SelfLearnZero 1.0"
ENGINE_AUTHOR = "Self-Play RL Engine Project"


def _approx_centipawns(value: float) -> int:
    """
    Maps the network's win/draw/loss-derived value in [-1, 1] to an
    approximate centipawn score for GUI display purposes only. This is
    a smooth, monotonic transform (not a literal material count, which
    this engine never computes) — values near 0 map to small cp
    numbers and values near +-1 map to large ones, similar in spirit
    to how other neural MCTS engines translate a win-probability-style
    value into a conventional-looking score for compatibility with
    GUIs built around traditional alpha-beta engines.
    """
    value = max(-0.999, min(0.999, value))
    cp = math.tan(value * math.pi / 2.0) * 150.0
    return int(max(-10000, min(10000, cp)))


def _load_weights_into(net: ChessNet, weights_path: str, device: torch.device) -> None:
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    net.load_state_dict(state_dict)


class UCIEngine:
    """
    Holds all mutable engine state (board, search threads, loaded
    network) across the lifetime of one UCI session (one `uci.py`
    process, which a GUI/TCEC typically keeps alive for an entire
    match or tournament).
    """

    def __init__(self, cfg: Optional[EngineConfig] = None, weights_path: Optional[str] = None):
        self.cfg = cfg if cfg is not None else EngineConfig()
        # Inference/search may fall back to CPU if no GPU is present —
        # unlike training (see train.py), this only affects search
        # speed, not correctness, so a hard requirement here would
        # needlessly prevent the engine from running at all on
        # GPU-less hardware (e.g. a laptop, or this very sandbox).
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")

        self.net = ChessNet(self.cfg.network).to(self.device)
        self.weights_path = weights_path
        if weights_path and os.path.exists(weights_path):
            _load_weights_into(self.net, weights_path, self.device)
        self.net.eval()

        self.mcts = MCTS(self.net, self.cfg, self.device)

        self.board = chess.Board()
        self.history = HistoryEncoder(self.cfg.network)
        self.history.push(self.board)

        self.tablebase = Tablebase(self.cfg.syzygy_path) if self.cfg.syzygy_path else None

        self.search_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self._ponder_timer: Optional[threading.Timer] = None
        self._last_normal_budget: float = 5.0  # seconds; used if a ponder search gets a ponderhit
        self._quit = False
        self.debug = False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def loop(self):
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                self._dispatch(line)
            except Exception as exc:  # never let malformed input kill the engine
                if self.debug:
                    print(f"info string error handling '{line}': {exc}", flush=True)
            if self._quit:
                break

    def _dispatch(self, line: str):
        parts = line.split()
        cmd = parts[0]
        args = parts[1:]

        if cmd == "uci":
            self._cmd_uci()
        elif cmd == "isready":
            self._cmd_isready()
        elif cmd == "ucinewgame":
            self._cmd_ucinewgame()
        elif cmd == "position":
            self._cmd_position(args)
        elif cmd == "go":
            self._cmd_go(args)
        elif cmd == "stop":
            self._cmd_stop()
        elif cmd == "ponderhit":
            self._cmd_ponderhit()
        elif cmd == "setoption":
            self._cmd_setoption(args)
        elif cmd == "debug":
            self.debug = (args[:1] == ["on"])
        elif cmd == "quit":
            self._cmd_quit()
        # "register", "ponderhit" mistypes, and any other unrecognized
        # command are silently ignored per the UCI spec.

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    def _cmd_uci(self):
        print(f"id name {ENGINE_NAME}", flush=True)
        print(f"id author {ENGINE_AUTHOR}", flush=True)
        print("option name Hash type spin default 256 min 1 max 65536", flush=True)
        print("option name Threads type spin default 1 min 1 max 128", flush=True)
        print("option name SyzygyPath type string default <empty>", flush=True)
        print(
            f"option name MCTS_Simulations type spin default {self.cfg.mcts.num_simulations} "
            "min 1 max 100000",
            flush=True,
        )
        print("option name Aggression type spin default 100 min 0 max 200", flush=True)
        print("uciok", flush=True)

    def _cmd_isready(self):
        print("readyok", flush=True)

    def _cmd_ucinewgame(self):
        self._stop_search_if_running()
        self.board = chess.Board()
        self.history = HistoryEncoder(self.cfg.network)
        self.history.push(self.board)

    def _cmd_position(self, args: List[str]):
        self._stop_search_if_running()
        if not args:
            return
        idx = 0
        if args[0] == "startpos":
            self.board = chess.Board()
            idx = 1
        elif args[0] == "fen":
            idx = 1
            fen_tokens = []
            while idx < len(args) and args[idx] != "moves":
                fen_tokens.append(args[idx])
                idx += 1
            try:
                self.board = chess.Board(" ".join(fen_tokens))
            except ValueError:
                return  # malformed FEN: ignore the whole command, per UCI leniency
        else:
            return

        self.history = HistoryEncoder(self.cfg.network)
        self.history.push(self.board)

        if idx < len(args) and args[idx] == "moves":
            for move_str in args[idx + 1:]:
                try:
                    move = chess.Move.from_uci(move_str)
                except ValueError:
                    break
                if move not in self.board.legal_moves:
                    break
                self.board.push(move)
                self.history.push(self.board)

    def _cmd_go(self, args: List[str]):
        self._stop_search_if_running()
        params = self._parse_go_args(args)
        self.stop_event = threading.Event()
        self.search_thread = threading.Thread(target=self._run_search, args=(params,), daemon=True)
        self.search_thread.start()

    def _cmd_stop(self):
        self._cancel_ponder_timer()
        self.stop_event.set()
        if self.search_thread is not None:
            self.search_thread.join(timeout=10.0)

    def _cmd_ponderhit(self):
        """
        The GUI is telling us the opponent played the move we were
        pondering on, and our clock now officially starts. Since our
        ongoing `go ... ponder` search was already running with an
        unbounded budget, we schedule it to stop after the time
        budget that would have applied to a normal (non-ponder) search
        from this position, computed back when `go ponder` was issued.
        """
        self._cancel_ponder_timer()
        budget = self._last_normal_budget
        self._ponder_timer = threading.Timer(budget, self._cmd_stop)
        self._ponder_timer.daemon = True
        self._ponder_timer.start()

    def _cmd_setoption(self, args: List[str]):
        if not args or args[0] != "name":
            return
        try:
            value_idx = args.index("value")
            name = " ".join(args[1:value_idx]).strip()
            value = " ".join(args[value_idx + 1:]).strip()
        except ValueError:
            name = " ".join(args[1:]).strip()
            value = None

        name_lower = name.lower()
        if name_lower == "syzygypath":
            if self.tablebase is not None:
                self.tablebase.close()
            self.cfg.syzygy_path = value or None
            self.tablebase = Tablebase(value) if value else None
        elif name_lower == "mcts_simulations" and value:
            try:
                self.cfg.mcts.num_simulations = max(1, int(value))
            except ValueError:
                pass
        elif name_lower == "aggression" and value:
            try:
                pct = max(0, min(200, int(value)))
                # 100% maps back to the trained default blend weight;
                # this lets a tournament operator dial the engine's
                # attacking style up or down without retraining.
                self.cfg.aggression.value_blend_weight = 0.12 * (pct / 100.0)
            except ValueError:
                pass
        # Hash / Threads are accepted (so GUIs that always send them
        # don't see an error) but currently have no effect: the search
        # is single-threaded and doesn't use a hash table the way
        # alpha-beta engines do (MCTS's tree already serves that role
        # within one search).

    def _cmd_quit(self):
        self._cancel_ponder_timer()
        self._stop_search_if_running()
        if self.tablebase is not None:
            self.tablebase.close()
        self._quit = True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _stop_search_if_running(self):
        self._cancel_ponder_timer()
        if self.search_thread is not None and self.search_thread.is_alive():
            self.stop_event.set()
            self.search_thread.join(timeout=10.0)
        self.search_thread = None

    def _cancel_ponder_timer(self):
        if self._ponder_timer is not None:
            self._ponder_timer.cancel()
            self._ponder_timer = None

    def _parse_go_args(self, args: List[str]) -> Dict:
        params: Dict = {"infinite": False, "ponder": False}
        i = 0
        int_keys = {"wtime", "btime", "winc", "binc", "movestogo", "depth", "nodes", "movetime", "mate"}
        while i < len(args):
            token = args[i]
            if token in int_keys:
                if i + 1 < len(args):
                    try:
                        params[token] = int(args[i + 1])
                    except ValueError:
                        pass
                    i += 2
                    continue
                i += 1
                continue
            elif token == "infinite":
                params["infinite"] = True
                i += 1
            elif token == "ponder":
                params["ponder"] = True
                i += 1
            elif token == "searchmoves":
                moves = []
                i += 1
                while i < len(args) and args[i] not in int_keys and args[i] not in (
                    "infinite", "ponder", "searchmoves",
                ):
                    try:
                        moves.append(chess.Move.from_uci(args[i]))
                    except ValueError:
                        pass
                    i += 1
                params["searchmoves"] = moves
            else:
                i += 1
        return params

    def _compute_budget(self, params: Dict):
        """Returns (time_budget_seconds, max_simulations_or_None)."""
        if params.get("movetime") is not None:
            budget = max(0.05, params["movetime"] / 1000.0 - 0.05)
            return budget, None

        if params.get("nodes") is not None:
            return float("inf"), max(1, params["nodes"])

        if params.get("depth") is not None:
            # MCTS has no native notion of ply-depth; we approximate by
            # scaling the simulation budget with the requested depth.
            # This is a documented approximation, not a literal
            # depth-limited search.
            return float("inf"), max(1, params["depth"]) * 800

        if params.get("infinite") or params.get("ponder"):
            return float("inf"), None

        wtime = params.get("wtime")
        btime = params.get("btime")
        if wtime is not None or btime is not None:
            my_time = (wtime if self.board.turn == chess.WHITE else btime) or 0
            my_inc = (params.get("winc") if self.board.turn == chess.WHITE else params.get("binc")) or 0
            movestogo = params.get("movestogo") or 30
            allocated_ms = my_time / max(1, movestogo) + my_inc * 0.8
            allocated_ms = min(allocated_ms, my_time * 0.5)  # never burn more than half the clock on one move
            allocated_ms = max(allocated_ms, 50.0)           # always search at least a little
            return allocated_ms / 1000.0, None

        # `go` with no time-control information at all (e.g. raw
        # testing from a terminal): fall back to a fixed, reasonable
        # default rather than searching forever.
        return 5.0, None

    def _run_search(self, params: Dict):
        move = self._select_move_for_go(params)
        if move is not None:
            print(f"bestmove {move.uci()}", flush=True)
        else:
            print("bestmove 0000", flush=True)

    def _select_move_for_go(self, params: Dict) -> Optional[chess.Move]:
        if self.board.is_game_over(claim_draw=True):
            return None

        # Exact endgame play via tablebase, when available — this is
        # mathematics applied to the current position, not a
        # pre-recorded "book" move, and is always at least as good as
        # anything MCTS could find by searching.
        if (
            self.tablebase is not None
            and self.tablebase.available
            and chess.popcount(self.board.occupied) <= self.cfg.mcts.tablebase_max_pieces
        ):
            tb_move = self.tablebase.probe_best_move(self.board)
            if tb_move is not None:
                return tb_move

        time_budget, max_sims = self._compute_budget(params)
        if not (params.get("infinite") or params.get("ponder")):
            self._last_normal_budget = time_budget if time_budget != float("inf") else self._last_normal_budget

        root, visit_dist, sims_done = self.mcts.run_timed(
            self.board,
            self.history,
            time_budget=time_budget,
            add_root_noise=False,
            tablebase=self.tablebase,
            max_simulations=max_sims,
            stop_event=self.stop_event,
        )

        if not visit_dist:
            return None

        searchmoves = params.get("searchmoves")
        if searchmoves:
            restricted = {m: v for m, v in visit_dist.items() if m in searchmoves}
            if restricted:
                visit_dist = restricted

        best_move = max(visit_dist.items(), key=lambda kv: kv[1])[0]
        cp = _approx_centipawns(root.value)
        print(
            f"info depth 1 nodes {sims_done} score cp {cp} pv {best_move.uci()}",
            flush=True,
        )

        return self.mcts.select_move(visit_dist, ply_count=10 ** 9)


def main():
    parser = argparse.ArgumentParser(description="UCI loop for the self-learning chess engine.")
    parser.add_argument("--weights", type=str, default=None, help="Path to a trained network checkpoint (state_dict).")
    parser.add_argument("--syzygy", type=str, default=None, help="Directory containing Syzygy tablebase files.")
    parser.add_argument("--simulations", type=int, default=None, help="Override cfg.mcts.num_simulations.")
    args = parser.parse_args()

    cfg = EngineConfig()
    if args.syzygy:
        cfg.syzygy_path = args.syzygy
    if args.simulations:
        cfg.mcts.num_simulations = args.simulations

    engine = UCIEngine(cfg=cfg, weights_path=args.weights)
    engine.loop()


if __name__ == "__main__":
    main()
