"""Allowed-action mask for Layer 1 of the dual-layer CA.

Implements ``docs/design.md`` §4.2: the per-cell bitmask over
:class:`~zip_ca.shapes.Shape` indices recording which shapes are
structurally admissible *before* any dynamics. Three independent
filters combine with logical AND:

1. **Boundary.** A shape whose open ports would point off-grid is
   rejected. Corner and edge cells lose the shapes that open
   outward.
2. **Wall.** A shape whose open port crosses a member of
   ``puzzle.walled_edges`` is rejected.
3. **Waypoint role.** Endpoints (waypoints #1 and #K) admit only
   endpoint shapes; every other cell admits only through-shapes.

The returned array has ``dtype=np.bool_`` and shape
``(N, N, 10)`` so it can be combined elementwise with Phase-5
probability tensors via plain multiplication or boolean masking.
"""

from enum import Enum

import numpy as np
from numpy.typing import NDArray

from .geometry import Cell, canonical_edge
from .puzzle import Puzzle, PuzzleValidationError
from .shapes import ENDPOINT_SHAPES, NUM_SHAPES, Shape, open_ports


class _WaypointRole(Enum):
    """Per-cell role governing which shape class is admissible.

    ``FIRST`` and ``LAST`` correspond to the two terminal waypoints
    of the Hamiltonian path; ``INTERMEDIATE`` is any other numbered
    waypoint; ``NONE`` is any cell not listed in
    :attr:`~zip_ca.puzzle.Puzzle.waypoints`.
    """

    NONE = "NONE"
    FIRST = "FIRST"
    INTERMEDIATE = "INTERMEDIATE"
    LAST = "LAST"


def _roles_by_cell(puzzle: Puzzle) -> dict[Cell, _WaypointRole]:
    """Index waypoint roles by cell for O(1) lookup inside the mask loop."""
    last_number = len(puzzle.waypoints)
    roles: dict[Cell, _WaypointRole] = {}
    for wp in puzzle.waypoints:
        cell = Cell(wp.row, wp.col)
        if wp.number == 1:
            roles[cell] = _WaypointRole.FIRST
        elif wp.number == last_number:
            roles[cell] = _WaypointRole.LAST
        else:
            roles[cell] = _WaypointRole.INTERMEDIATE
    return roles


def _shape_is_allowed(
    puzzle: Puzzle,
    cell: Cell,
    shape: Shape,
    role: _WaypointRole,
) -> bool:
    """Return ``True`` iff ``shape`` passes all three §4.2 filters at ``cell``."""
    shape_is_endpoint = shape in ENDPOINT_SHAPES
    cell_is_terminal = role in {_WaypointRole.FIRST, _WaypointRole.LAST}
    if shape_is_endpoint != cell_is_terminal:
        return False

    n = puzzle.size
    for port in open_ports(shape):
        dr, dc = port.delta
        nr, nc = cell.x + dr, cell.y + dc
        if not (0 <= nr < n and 0 <= nc < n):
            return False
        if canonical_edge(cell, Cell(nr, nc)) in puzzle.walled_edges:
            return False
    return True


def build_allowed_mask(puzzle: Puzzle) -> NDArray[np.bool_]:
    """Build the per-cell allowed-shape bitmask for ``puzzle``.

    Args:
        puzzle: The fully-validated puzzle to analyse.

    Returns:
        Boolean array of shape ``(puzzle.size, puzzle.size, 10)``.
        ``mask[i, j, s]`` is ``True`` iff shape index ``s`` is
        structurally admissible at cell ``(i, j)`` under the three
        filters from design §4.2.

    Raises:
        PuzzleValidationError: If any cell ends up with zero
            admissible shapes — the puzzle is unsolvable from the
            static structure alone.
    """
    n = puzzle.size
    roles = _roles_by_cell(puzzle)
    mask = np.zeros((n, n, NUM_SHAPES), dtype=np.bool_)

    for i in range(n):
        for j in range(n):
            cell = Cell(i, j)
            role = roles.get(cell, _WaypointRole.NONE)
            for shape in Shape:
                if _shape_is_allowed(puzzle, cell, shape, role):
                    mask[i, j, shape] = True

    # Surface unsolvable puzzles here rather than deep inside a Phase-5
    # solver loop, where the silent failure mode would be "probabilities
    # collapse to zero".
    counts = mask.any(axis=-1)
    if not bool(counts.all()):
        dead_cells = [(int(i), int(j)) for i, j in np.argwhere(~counts)]
        msg = f"Puzzle has cells with zero admissible shapes: {dead_cells}"
        raise PuzzleValidationError(msg)

    return mask
