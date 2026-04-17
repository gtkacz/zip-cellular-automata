"""External path validator for the Layer-1 shape grid.

Implements ``docs/design.md`` §9.2: walk the shape array from
waypoint #1, following the unique non-incoming open port at each
cell, and verify the result is a Hamiltonian path that visits all
waypoints in numeric order, terminates at waypoint #K, and never
mismatches a port across an edge.

The validator returns a :class:`TraceResult` carrying a diagnostic
reason and the longest-valid-prefix path. *Never* raises on a failed
trace: "no valid path yet" is an expected state every tick before
convergence, and wrapping it in exceptions would make the dev loop
noisy. Bona fide bugs (e.g., out-of-range shape indices) still
surface as ``ValueError`` from :class:`~zip_ca.shapes.Shape`.

Separation of concerns: Phase 3's ``path_to_shapes`` *derives* a
shape grid from a known path; ``trace_path`` *validates* a shape
grid against a puzzle's waypoint spec. The two functions are
independent implementations of inverse operations, which is exactly
the property that makes
``trace_path(path_to_shapes(solution, puzzle), puzzle).ok`` a
meaningful self-test of both modules.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .direction import Direction
from .geometry import Cell
from .puzzle import Puzzle, Waypoint
from .shapes import NUM_SHAPES, Shape, open_ports


@dataclass(frozen=True, slots=True)
class TraceResult:
    """Outcome of a :func:`trace_path` invocation.

    Attributes:
        ok: ``True`` iff the shape grid encodes a full valid
            Hamiltonian path from W_1 to W_K touching waypoints in
            order.
        reason: ``"valid"`` on success; a short diagnostic string
            describing the first failure point otherwise.
        path: The longest valid prefix, as a tuple of cells in walk
            order. On success equals the full Hamiltonian path.
    """

    ok: bool
    reason: str
    path: tuple[Cell, ...]


def trace_path(shapes: NDArray[np.int8], puzzle: Puzzle) -> TraceResult:
    """Validate ``shapes`` as a Hamiltonian path through ``puzzle``.

    Starts at waypoint #1 and follows the unique non-incoming open
    port at each cell. Checks, in order:

    1. Start cell equals waypoint #1's cell and has an endpoint shape.
    2. Every through-cell has exactly one exit other than its
       incoming port.
    3. Every step stays on-grid (defensive; :func:`build_allowed_mask`
       should forbid boundary-opening shapes at edge cells).
    4. The destination cell's opposite port is open (reciprocity).
    5. No cell is revisited (Hamiltonian → simple path).
    6. Waypoints encountered appear in strictly increasing
       ``.number`` order.
    7. The walk terminates at waypoint #K with an endpoint shape
       and has visited every cell exactly once.

    Args:
        shapes: ``(N, N)`` int8 shape-index array.
        puzzle: The puzzle whose waypoint ordering and size define
            the success condition.

    Returns:
        A :class:`TraceResult`.
    """
    waypoints_by_cell: dict[Cell, Waypoint] = {
        Cell(wp.row, wp.col): wp for wp in puzzle.waypoints
    }
    try:
        w1 = next(wp for wp in puzzle.waypoints if wp.number == 1)
    except StopIteration:
        return TraceResult(False, "puzzle has no waypoint numbered 1", ())

    current = Cell(w1.row, w1.col)
    path: list[Cell] = [current]
    visited: set[Cell] = {current}
    incoming: Direction | None = None
    expected_number = 2
    last_number = len(puzzle.waypoints)

    while True:
        shape_idx = int(shapes[current.x, current.y])
        if not (0 <= shape_idx < NUM_SHAPES):
            return TraceResult(
                False, f"invalid shape index {shape_idx} at {current}", tuple(path),
            )
        shape = Shape(shape_idx)
        ports = open_ports(shape)

        outgoing = _pick_outgoing(ports, incoming, is_start=(len(path) == 1))
        if outgoing is None:
            return TraceResult(
                False, f"no valid exit at {current} (shape={shape.name})", tuple(path),
            )

        dr, dc = outgoing.delta
        nxt = Cell(current.x + dr, current.y + dc)
        if not (0 <= nxt.x < puzzle.size and 0 <= nxt.y < puzzle.size):
            return TraceResult(
                False, f"path exits grid at {current} via {outgoing.name}", tuple(path),
            )

        nxt_shape_idx = int(shapes[nxt.x, nxt.y])
        if not (0 <= nxt_shape_idx < NUM_SHAPES):
            return TraceResult(
                False, f"invalid shape index {nxt_shape_idx} at {nxt}", tuple(path),
            )
        nxt_shape = Shape(nxt_shape_idx)
        if outgoing.opposite() not in open_ports(nxt_shape):
            return TraceResult(
                False, f"port mismatch {current} -> {nxt}", tuple(path),
            )
        if nxt in visited:
            return TraceResult(False, f"revisits cell {nxt}", tuple(path))

        visited.add(nxt)
        path.append(nxt)

        wp = waypoints_by_cell.get(nxt)
        if wp is not None:
            if wp.number != expected_number:
                return TraceResult(
                    False,
                    f"waypoint order violated at {nxt}: got {wp.number}, expected {expected_number}",
                    tuple(path),
                )
            expected_number += 1
            if wp.number == last_number:
                if len(open_ports(nxt_shape)) != 1:
                    return TraceResult(
                        False, "final waypoint has non-endpoint shape", tuple(path),
                    )
                if len(visited) != puzzle.size * puzzle.size:
                    return TraceResult(
                        False,
                        f"early termination: {len(visited)}/{puzzle.size ** 2} cells visited",
                        tuple(path),
                    )
                return TraceResult(True, "valid", tuple(path))

        incoming = outgoing.opposite()
        current = nxt


def _pick_outgoing(
    ports: frozenset[Direction],
    incoming: Direction | None,
    *,
    is_start: bool,
) -> Direction | None:
    """Select the outgoing port for the current cell.

    At the starting cell there is no incoming port; the cell must be
    an endpoint shape (exactly one open port), which is then the
    outgoing. For any other cell, the outgoing port is the unique
    open port that is not ``incoming``. Returns ``None`` if the cell
    violates these invariants (wrong number of ports for its role,
    or incoming port not open).
    """
    if is_start:
        if len(ports) != 1:
            return None
        (outgoing,) = ports
        return outgoing
    if incoming is None or incoming not in ports:
        return None
    outgoing_candidates = ports - {incoming}
    if len(outgoing_candidates) != 1:
        return None
    (outgoing,) = outgoing_candidates
    return outgoing
