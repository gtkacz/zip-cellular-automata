"""Chemical-diffusion substrate for Layer 2 of the dual-layer CA.

Implements ``docs/design.md`` §5 as a standalone, pure-function API:
the per-cell segment-chemical tensor, the precomputed mutual-open
adjacency derived from a fixed shape grid, the Dirichlet waypoint
source specification, and a single vectorised tick of the
shape-gated 5-point diffusion stencil.

Phase 4 deliberately omits the Layer-1 probability tensor and any
reward coupling — the shape array is a fixed input derived once
from a hand-supplied solution path. Phase 5 will fold these four
functions into :class:`EngineState.tick` (see ``docs/design.md``
§11.2) and switch to in-place updates once profiling warrants.

Indexing convention (binding):

* Design's 1-indexed segment ``k`` (``1 <= k <= K-1``) maps to the
  numpy third-axis index ``k - 1``. All array operations use the
  0-indexed form; only docstrings and error messages mention the
  1-indexed form.
* Axis-2 of :data:`mutual_open` orders directions ``(N, E, S, W)``
  per the module-local :data:`_PORT_ORDER`. This ordering is
  internal; the public API never exposes raw axis-2 indices.
"""

from typing import Final

import numpy as np
from numpy.typing import NDArray

from .direction import Direction
from .geometry import Cell, canonical_edge
from .puzzle import Puzzle, PuzzleValidationError
from .shapes import NUM_SHAPES, Shape, open_ports

# Phase 5 will lift these to EngineState constructor args; Final here
# signals "compile-time constant, do not rebind".
C0: Final[float] = 1.0
ALPHA: Final[float] = 0.25
DELTA: Final[float] = 0.01

# Internal ordering for mutual_open's axis-2. Must not leak into the
# public API — Shape enum ordering is the public convention.
_PORT_ORDER: Final[tuple[Direction, ...]] = (
    Direction.N,
    Direction.E,
    Direction.S,
    Direction.W,
)

# A K-waypoint puzzle has K-1 segment chemicals; the design invariant
# §11.1.4 guarantees K ≥ 2 at parse time, but we double-check here so
# the failure mode inside a solver loop is never "silent shape (N, N, 0)".
_MIN_SEGMENTS: Final[int] = 1


def init_chems(puzzle: Puzzle) -> NDArray[np.float32]:
    """Allocate the zero-initialised chemical tensor for ``puzzle``.

    Args:
        puzzle: A fully-validated puzzle with ``K ≥ 2`` waypoints.

    Returns:
        ``NDArray[np.float32]`` of shape ``(N, N, K-1)`` filled with
        zeros. Waypoint source clamps are *not* applied here — they
        are re-asserted every tick by :func:`diffuse_tick`, which
        matches design §8 Algorithm 1 Step 2.

    Raises:
        PuzzleValidationError: If the puzzle has fewer than two
            waypoints (no segments → no chemicals).
    """
    k_segments = len(puzzle.waypoints) - 1
    if k_segments < _MIN_SEGMENTS:
        msg = (
            f"Puzzle has {len(puzzle.waypoints)} waypoint(s); need ≥ 2 "
            "to define at least one segment chemical"
        )
        raise PuzzleValidationError(msg)
    return np.zeros((puzzle.size, puzzle.size, k_segments), dtype=np.float32)


def build_mutual_open(
    shapes: NDArray[np.int8],
    puzzle: Puzzle,
) -> NDArray[np.bool_]:
    """Precompute the mutual-open neighbour mask from a fixed shape grid.

    ``mutual_open[i, j, d]`` is ``True`` iff chemicals may flow from
    cell ``(i, j)`` to its neighbour in direction ``_PORT_ORDER[d]``,
    which requires *all* of:

    1. Port ``d`` is open in ``shapes[i, j]``.
    2. The neighbour at direction ``d`` is in-bounds.
    3. Port ``d.opposite()`` is open in the neighbour's shape.
    4. The edge between them is not in ``puzzle.walled_edges``.

    The wall check is defensive: a shape array produced by
    :func:`~zip_ca.path_shapes.path_to_shapes` from a validated path
    already respects walls, but Phase 5 will re-run this function
    every tick on shapes chosen by probability argmax, where the wall
    intersection is load-bearing.

    Args:
        shapes: ``NDArray[np.int8]`` of shape ``(N, N)`` whose entries
            are :class:`~zip_ca.shapes.Shape` indices in ``0..9`` or
            ``-1`` for "no shape at this cell". The ``-1`` sentinel
            is treated as having no open ports.
        puzzle: The puzzle whose size and walls gate the adjacency.

    Returns:
        ``NDArray[np.bool_]`` of shape ``(N, N, 4)`` with axis-2
        ordering ``_PORT_ORDER``.

    Raises:
        ValueError: If ``shapes`` has the wrong shape or an entry
            outside ``{-1, 0, …, 9}``.
    """
    n = puzzle.size
    if shapes.shape != (n, n):
        msg = f"shapes must have shape ({n}, {n}); got {shapes.shape}"
        raise ValueError(msg)

    opens = _compute_open_ports(shapes, n)
    return _compute_mutual(opens, puzzle, n)


def _compute_open_ports(shapes: NDArray[np.int8], n: int) -> NDArray[np.bool_]:
    """Expand a shape grid into a ``(N, N, 4)`` per-port open-mask.

    Iterating per-cell with the Shape enum is O(N²) and trivially
    readable; vectorising would require a (10, 4) lookup array, not
    worth the obfuscation for grids up to 8x8.
    """
    opens = np.zeros((n, n, len(_PORT_ORDER)), dtype=np.bool_)
    for i in range(n):
        for j in range(n):
            idx = int(shapes[i, j])
            if idx == -1:
                continue
            if not (0 <= idx < NUM_SHAPES):
                msg = f"Invalid shape index {idx} at cell ({i}, {j})"
                raise ValueError(msg)
            cell_ports = open_ports(Shape(idx))
            for d_idx, direction in enumerate(_PORT_ORDER):
                if direction in cell_ports:
                    opens[i, j, d_idx] = True
    return opens


def _compute_mutual(
    opens: NDArray[np.bool_],
    puzzle: Puzzle,
    n: int,
) -> NDArray[np.bool_]:
    """Intersect ``opens`` with neighbour reciprocity and the wall set."""
    mutual = np.zeros_like(opens)
    for i in range(n):
        for j in range(n):
            for d_idx, direction in enumerate(_PORT_ORDER):
                if not opens[i, j, d_idx]:
                    continue
                dr, dc = direction.delta
                ni, nj = i + dr, j + dc
                if not (0 <= ni < n and 0 <= nj < n):
                    continue
                opp_idx = _PORT_ORDER.index(direction.opposite())
                if not opens[ni, nj, opp_idx]:
                    continue
                if canonical_edge(Cell(i, j), Cell(ni, nj)) in puzzle.walled_edges:
                    continue
                mutual[i, j, d_idx] = True
    return mutual


def build_sources(puzzle: Puzzle) -> NDArray[np.int64]:
    """Build the Dirichlet waypoint source clamp specification.

    Per design §5.2, each waypoint re-clamps the chemicals of its
    adjacent segment(s) to :data:`C0` every tick:

    * Waypoint ``W_1`` clamps segment 1 only (numpy index ``0``).
    * Waypoint ``W_K`` clamps segment ``K-1`` only (numpy index ``K-2``).
    * Intermediate ``W_k`` (``1 < k < K``) clamps segments ``k-1`` and
      ``k`` (numpy indices ``k-2`` and ``k-1``).

    The resulting array is sized for advanced indexing in one call:
    ``u[rows, cols, segs] = C0``.

    Args:
        puzzle: The puzzle whose waypoints drive the clamps.

    Returns:
        ``NDArray[np.int64]`` of shape ``(2K-2, 3)``. Each row is
        ``[row, col, numpy_segment_index]``. Ordering follows the
        waypoint's ``.number`` and, for intermediate waypoints, their
        incoming segment first then outgoing.
    """
    triples: list[tuple[int, int, int]] = []
    last_number = len(puzzle.waypoints)
    for wp in puzzle.waypoints:
        if wp.number > 1:
            triples.append((wp.row, wp.col, wp.number - 2))
        if wp.number < last_number:
            triples.append((wp.row, wp.col, wp.number - 1))
    return np.array(triples, dtype=np.int64)


def diffuse_tick(
    chems: NDArray[np.float32],
    mutual_open: NDArray[np.bool_],
    sources: NDArray[np.int64],
) -> NDArray[np.float32]:
    r"""Advance the chemical field by one tick of shape-gated diffusion.

    Implements design §5.3 plus the Dirichlet source re-assertion of
    §5.2, in the order prescribed by §8 Algorithm 1 (diffuse → decay
    → clamp):

    .. math::

        u^{t+1}_{i,j}[k] = (1 - \\delta) \\cdot
            \\begin{cases}
                (1 - \\alpha) u^t_{i,j}[k] + \\alpha \\,
                    \\overline{u^t_{\\Omega}}[k] & |\\Omega| > 0 \\\\
                u^t_{i,j}[k] & |\\Omega| = 0
            \\end{cases}

    Waypoint source entries are subsequently overwritten with
    :data:`C0`, regardless of the pre-clamp decay result.

    Args:
        chems: Current ``(N, N, K-1)`` float32 concentration tensor.
            Not mutated.
        mutual_open: Precomputed ``(N, N, 4)`` bool mask from
            :func:`build_mutual_open`. Axis-2 order is
            :data:`_PORT_ORDER`.
        sources: ``(M, 3)`` int64 array from :func:`build_sources`.

    Returns:
        A new ``(N, N, K-1)`` float32 array; input ``chems`` is
        unchanged. Identical inputs produce arrays equal under
        :func:`numpy.array_equal` — the function is referentially
        transparent.
    """
    # Boundary wrap from np.roll is masked to zero by mutual_open,
    # which has False for any out-of-bounds neighbour, so the wrapped
    # contributions vanish arithmetically without an explicit slice.
    u_n = np.roll(chems, shift=+1, axis=0)
    u_e = np.roll(chems, shift=-1, axis=1)
    u_s = np.roll(chems, shift=-1, axis=0)
    u_w = np.roll(chems, shift=+1, axis=1)

    gate_n = mutual_open[..., 0:1].astype(np.float32)
    gate_e = mutual_open[..., 1:2].astype(np.float32)
    gate_s = mutual_open[..., 2:3].astype(np.float32)
    gate_w = mutual_open[..., 3:4].astype(np.float32)

    contribs = u_n * gate_n + u_e * gate_e + u_s * gate_s + u_w * gate_w
    n_open = (gate_n + gate_e + gate_s + gate_w).astype(np.float32)

    has_neighbour = n_open > 0
    neighbour_mean = np.divide(
        contribs,
        n_open,
        out=np.zeros_like(chems),
        where=has_neighbour,
    )
    mix = np.where(
        has_neighbour,
        (np.float32(1.0) - np.float32(ALPHA)) * chems
        + np.float32(ALPHA) * neighbour_mean,
        chems,
    )
    u_new = ((np.float32(1.0) - np.float32(DELTA)) * mix).astype(np.float32)

    if sources.size > 0:
        rows, cols, segs = sources[:, 0], sources[:, 1], sources[:, 2]
        u_new[rows, cols, segs] = np.float32(C0)

    return u_new
