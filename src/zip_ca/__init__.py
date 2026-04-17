"""Top-level public API for the Zip cellular-automata solver."""

from .direction import Direction
from .geometry import Cell, Edge, canonical_edge
from .puzzle import Puzzle, PuzzleValidationError, Waypoint
from .puzzle_io import (
    dump_puzzle,
    load_puzzle,
    parse_puzzle,
    parse_puzzle_obj,
    to_json_obj,
)

__all__ = [
    "Cell",
    "Direction",
    "Edge",
    "Puzzle",
    "PuzzleValidationError",
    "Waypoint",
    "canonical_edge",
    "dump_puzzle",
    "load_puzzle",
    "parse_puzzle",
    "parse_puzzle_obj",
    "to_json_obj",
]
