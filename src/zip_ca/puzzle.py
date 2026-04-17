"""Immutable puzzle artifact.

The ``Puzzle`` dataclass is the sole in-memory representation of a Zip puzzle.
It is constructed only via the parsers in :mod:`zip_ca.puzzle_io`; downstream
solver code consumes it as a frozen, shareable backreference.
"""

from dataclasses import dataclass

from .geometry import Edge


@dataclass(frozen=True, slots=True)
class Waypoint:
    """A numbered waypoint cell in the puzzle.

    Waypoint numbers across a puzzle form exactly the contiguous set
    ``{1, 2, …, K}`` for some ``K ≥ 2`` (validated at parse time).

    Attributes:
        row: Zero-indexed row, ``0 ≤ row < size``.
        col: Zero-indexed column, ``0 ≤ col < size``.
        number: 1-indexed waypoint ordinal.
    """

    row: int
    col: int
    number: int


@dataclass(frozen=True, slots=True)
class Puzzle:
    """A fully-validated Zip puzzle.

    Walls are stored as canonical undirected edges, never as per-cell port
    blockers, so the in-memory representation has exactly one form per wall.

    Attributes:
        size: Side length of the square grid.
        waypoints: Waypoints sorted ascending by ``.number``.
        walled_edges: Canonical edges blocked by walls.
        name: Optional human-readable label.
        source: Optional provenance string.
    """

    size: int
    waypoints: tuple[Waypoint, ...]
    walled_edges: frozenset[Edge]
    name: str | None = None
    source: str | None = None


class PuzzleValidationError(ValueError):
    """Raised when a JSON puzzle violates a data-model invariant.

    Subclasses :class:`ValueError` so callers handling user-supplied input
    can catch the broader category, while genuine bugs continue to surface
    as their own exception types.
    """
