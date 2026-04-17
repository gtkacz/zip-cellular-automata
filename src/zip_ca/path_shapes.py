"""Path-to-shape derivation for Layer-1 fixtures and visualisation.

Given a Hamiltonian path through a puzzle, :func:`path_to_shapes`
mechanically derives the :class:`~zip_ca.shapes.Shape` at each cell
by accumulating open-port directions from consecutive path
transitions. This module is a **fixture/visualisation utility**, not
a solver: it consumes a hand-supplied path and converts it into the
per-cell shape array the Panel-A renderer expects.

The function doubles as a language-level validator for
``*.solution.json`` files: any path that fails to satisfy the
Hamiltonian, adjacency, wall-respecting, or waypoint-ordering
invariants raises :class:`~zip_ca.puzzle.PuzzleValidationError` with
a precise message.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .actions import build_allowed_mask
from .direction import Direction
from .geometry import Cell, canonical_edge
from .puzzle import Puzzle, PuzzleValidationError
from .shapes import shape_for_ports

# Inverse of Direction.delta; lets us classify the (dr, dc) of a
# path transition directly as a port Direction without a linear scan.
_DELTA_TO_DIRECTION: dict[tuple[int, int], Direction] = {d.delta: d for d in Direction}


def _direction_between(src: Cell, dst: Cell) -> Direction:
    """Return the :class:`Direction` from ``src`` to its 4-adjacent neighbour ``dst``.

    Raises:
        PuzzleValidationError: If ``src`` and ``dst`` are not 4-adjacent.
    """
    delta = (dst.x - src.x, dst.y - src.y)
    direction = _DELTA_TO_DIRECTION.get(delta)
    if direction is None:
        msg = f"Cells {tuple(src)} and {tuple(dst)} are not 4-adjacent (delta={delta})"
        raise PuzzleValidationError(msg)
    return direction


def _validate_path(path: Sequence[Cell], puzzle: Puzzle) -> None:
    """Enforce the eight path-level invariants from the Phase-3 plan §8."""
    n = puzzle.size
    expected_len = n * n
    if len(path) != expected_len:
        msg = f"Path length {len(path)} does not match N² = {expected_len} (non-Hamiltonian)"
        raise PuzzleValidationError(msg)

    seen: dict[Cell, int] = {}
    for idx, cell in enumerate(path):
        if not (0 <= cell.x < n and 0 <= cell.y < n):
            msg = f"Path cell {tuple(cell)} at index {idx} is out of bounds for size {n}"
            raise PuzzleValidationError(msg)
        if cell in seen:
            msg = f"Cell {tuple(cell)} re-used at path indices {seen[cell]} and {idx}"
            raise PuzzleValidationError(msg)
        seen[cell] = idx

    for idx in range(len(path) - 1):
        src, dst = path[idx], path[idx + 1]
        # Raises PuzzleValidationError if not 4-adjacent; keeps the
        # adjacency check co-located with its natural error site.
        _direction_between(src, dst)
        if canonical_edge(src, dst) in puzzle.walled_edges:
            msg = f"Path crosses walled edge between {tuple(src)} and {tuple(dst)} at index {idx}"
            raise PuzzleValidationError(msg)

    # Waypoint-order traversal: the subsequence of waypoint cells visited,
    # in the order the path visits them, must equal waypoints sorted by number.
    waypoint_cells = {Cell(wp.row, wp.col): wp.number for wp in puzzle.waypoints}
    visited_numbers = [waypoint_cells[cell] for cell in path if cell in waypoint_cells]
    expected_numbers = list(range(1, len(puzzle.waypoints) + 1))
    if visited_numbers != expected_numbers:
        msg = (
            f"Path visits waypoints in order {visited_numbers}; "
            f"expected ascending {expected_numbers}"
        )
        raise PuzzleValidationError(msg)


def path_to_shapes(path: Sequence[Cell], puzzle: Puzzle) -> NDArray[np.int8]:
    """Derive the per-cell :class:`Shape` index array for a Hamiltonian ``path``.

    Args:
        path: Ordered sequence of cells forming a Hamiltonian path
            through ``puzzle``. First cell must be waypoint #1; last
            cell must be waypoint #K.
        puzzle: The puzzle whose geometry the path traverses.

    Returns:
        ``NDArray[np.int8]`` of shape ``(puzzle.size, puzzle.size)``
        whose entries are :class:`Shape` indices (``0..9``). Cells
        not on the path receive ``-1`` (unreachable for a Hamiltonian
        path but kept as the sentinel convention for Phase-5's
        partial-state visualisation).

    Raises:
        PuzzleValidationError: If the path violates any of the
            invariants from §8 of the Phase-3 plan, or if any derived
            shape disagrees with :func:`build_allowed_mask` (which
            would indicate a fixture-vs-puzzle mismatch).
    """
    _validate_path(path, puzzle)

    n = puzzle.size
    ports_by_cell: dict[Cell, set[Direction]] = {cell: set() for cell in path}
    for idx in range(len(path) - 1):
        src, dst = path[idx], path[idx + 1]
        port_out = _direction_between(src, dst)
        ports_by_cell[src].add(port_out)
        ports_by_cell[dst].add(port_out.opposite())

    shapes = np.full((n, n), -1, dtype=np.int8)
    allowed = build_allowed_mask(puzzle)
    for cell, ports in ports_by_cell.items():
        try:
            shape = shape_for_ports(frozenset(ports))
        except KeyError as exc:
            msg = (
                f"Cell {tuple(cell)} has port set {sorted(p.value for p in ports)} "
                f"which does not correspond to any Shape"
            )
            raise PuzzleValidationError(msg) from exc
        if not bool(allowed[cell.x, cell.y, shape]):
            msg = (
                f"Derived shape {shape.name} at {tuple(cell)} is not allowed by "
                "the static mask (fixture disagrees with puzzle geometry)"
            )
            raise PuzzleValidationError(msg)
        shapes[cell.x, cell.y] = shape

    return shapes
