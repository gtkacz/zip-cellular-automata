"""Top-level public API for the Zip cellular-automata solver."""

from .actions import build_allowed_mask
from .diffusion import (
    ALPHA,
    C0,
    DELTA,
    build_mutual_open,
    build_open_mutual,
    build_sources,
    diffuse_tick,
    init_chems,
)
from .direction import Direction
from .engine import BETA, ETA_0, TAU, EngineState
from .geometry import Cell, Edge, canonical_edge
from .noise import inject_noise
from .path_shapes import path_to_shapes
from .puzzle import Puzzle, PuzzleValidationError, Waypoint
from .puzzle_io import (
    dump_puzzle,
    load_puzzle,
    parse_puzzle,
    parse_puzzle_obj,
    to_json_obj,
)
from .quiescence import QuiescenceDetector
from .scoring import score_shapes
from .shapes import (
    ENDPOINT_SHAPES,
    NUM_SHAPES,
    THROUGH_SHAPES,
    Shape,
    open_ports,
    shape_for_ports,
)
from .solver import R_MAX, T_MAX, RunStats, SolveResult, solve
from .trace import TraceResult, trace_path
from .viz import render_chem_layer, render_path_layer

__all__ = [
    "ALPHA",
    "BETA",
    "C0",
    "DELTA",
    "ENDPOINT_SHAPES",
    "ETA_0",
    "NUM_SHAPES",
    "R_MAX",
    "TAU",
    "THROUGH_SHAPES",
    "T_MAX",
    "Cell",
    "Direction",
    "Edge",
    "EngineState",
    "Puzzle",
    "PuzzleValidationError",
    "QuiescenceDetector",
    "RunStats",
    "Shape",
    "SolveResult",
    "TraceResult",
    "Waypoint",
    "build_allowed_mask",
    "build_mutual_open",
    "build_open_mutual",
    "build_sources",
    "canonical_edge",
    "diffuse_tick",
    "dump_puzzle",
    "init_chems",
    "inject_noise",
    "load_puzzle",
    "open_ports",
    "parse_puzzle",
    "parse_puzzle_obj",
    "path_to_shapes",
    "render_chem_layer",
    "render_path_layer",
    "score_shapes",
    "shape_for_ports",
    "solve",
    "to_json_obj",
    "trace_path",
]
