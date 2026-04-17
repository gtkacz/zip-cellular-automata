"""Top-level public API for the Zip cellular-automata solver."""

from .actions import build_allowed_mask
from .direction import Direction
from .geometry import Cell, Edge, canonical_edge
from .path_shapes import path_to_shapes
from .puzzle import Puzzle, PuzzleValidationError, Waypoint
from .puzzle_io import (
    dump_puzzle,
    load_puzzle,
    parse_puzzle,
    parse_puzzle_obj,
    to_json_obj,
)
from .shapes import (
    ENDPOINT_SHAPES,
    NUM_SHAPES,
    THROUGH_SHAPES,
    Shape,
    open_ports,
    shape_for_ports,
)

__all__ = [
    "ENDPOINT_SHAPES",
    "NUM_SHAPES",
    "THROUGH_SHAPES",
    "Cell",
    "Direction",
    "Edge",
    "Puzzle",
    "PuzzleValidationError",
    "Shape",
    "Waypoint",
    "build_allowed_mask",
    "canonical_edge",
    "dump_puzzle",
    "load_puzzle",
    "open_ports",
    "parse_puzzle",
    "parse_puzzle_obj",
    "path_to_shapes",
    "shape_for_ports",
    "to_json_obj",
]
