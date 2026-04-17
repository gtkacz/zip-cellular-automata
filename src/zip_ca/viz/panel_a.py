"""Panel A renderer: path-layer visualisation.

Implements ``docs/design.md`` §10 Panel A. Given a puzzle and a
per-cell :class:`~zip_ca.shapes.Shape` index array, draws the
grid, walls, chosen shape segments, and waypoint numbers on a
caller-supplied :class:`matplotlib.axes.Axes`.

The renderer is **pure**: it takes an ``Axes``, draws onto it, and
returns it. It never calls ``plt.show`` or ``savefig`` — the caller
owns the figure lifecycle so the same function is reusable from the
eventual animation harness (Phase 7) without modification.

Coordinate convention: cell ``(i, j)`` occupies the unit square
``[j, j+1] x [i, i+1]`` with matplotlib's default y-up axes. A call
to :meth:`Axes.invert_yaxis` at the end flips the view so row ``0``
appears at the top, matching the puzzle's row-major convention
without forcing every drawing call to do the subtraction.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

from zip_ca.direction import Direction
from zip_ca.geometry import Cell
from zip_ca.puzzle import Puzzle
from zip_ca.shapes import NUM_SHAPES, Shape, open_ports

_GRID_COLOR = "#cccccc"
_WALL_COLOR = "#000000"
_PATH_COLOR = "#1f77b4"
_WAYPOINT_COLOR = "#d62728"
_GRID_LW = 0.8
_WALL_LW = 4.0
_PATH_LW = 3.0


def _port_midpoint(cell: Cell, port: Direction) -> tuple[float, float]:
    """Return the ``(x, y)`` matplotlib coordinate of ``port`` on ``cell``."""
    i, j = cell.x, cell.y
    match port:
        case Direction.N:
            return (j + 0.5, float(i))
        case Direction.E:
            return (j + 1.0, i + 0.5)
        case Direction.S:
            return (j + 0.5, i + 1.0)
        case Direction.W:
            return (float(j), i + 0.5)


def _wall_segment(a: Cell, b: Cell) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the two endpoints of the wall segment shared by ``a`` and ``b``."""
    if a.x == b.x:
        # Horizontal neighbours; wall is the vertical edge between them.
        boundary_col = max(a.y, b.y)
        return ((float(boundary_col), float(a.x)), (float(boundary_col), float(a.x + 1)))
    # Vertical neighbours; wall is the horizontal edge between them.
    boundary_row = max(a.x, b.x)
    return ((float(a.y), float(boundary_row)), (float(a.y + 1), float(boundary_row)))


def _draw_confidence(ax: Axes, confidence: NDArray[np.float64], n: int) -> None:
    """Tint each cell by ``confidence`` using the ``Blues`` cmap."""
    cmap = plt.get_cmap("Blues")
    for i in range(n):
        for j in range(n):
            ax.add_patch(
                Rectangle(
                    (float(j), float(i)),
                    1.0,
                    1.0,
                    facecolor=cmap(float(confidence[i, j])),
                    edgecolor="none",
                    zorder=0,
                ),
            )


def _draw_grid(ax: Axes, n: int) -> None:
    """Draw the thin grid lines delimiting each cell."""
    for k in range(n + 1):
        ax.plot(
            [0.0, float(n)],
            [float(k), float(k)],
            color=_GRID_COLOR,
            linewidth=_GRID_LW,
            zorder=1,
        )
        ax.plot(
            [float(k), float(k)],
            [0.0, float(n)],
            color=_GRID_COLOR,
            linewidth=_GRID_LW,
            zorder=1,
        )


def _draw_walls(ax: Axes, puzzle: Puzzle) -> None:
    """Draw each wall as a thick black segment along the shared cell boundary."""
    for edge in puzzle.walled_edges:
        (x1, y1), (x2, y2) = _wall_segment(edge.a, edge.b)
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=_WALL_COLOR,
            linewidth=_WALL_LW,
            solid_capstyle="butt",
            zorder=3,
        )


def _draw_shapes(ax: Axes, shapes: NDArray[np.int8], n: int) -> None:
    """Draw the per-cell path segments for each non-sentinel shape entry."""
    for i in range(n):
        for j in range(n):
            idx = int(shapes[i, j])
            if idx == -1:
                continue
            if not (0 <= idx < NUM_SHAPES):
                msg = f"Invalid shape index {idx} at cell ({i}, {j})"
                raise ValueError(msg)
            cell = Cell(i, j)
            centre = (j + 0.5, i + 0.5)
            for port in open_ports(Shape(idx)):
                px, py = _port_midpoint(cell, port)
                ax.plot(
                    [centre[0], px],
                    [centre[1], py],
                    color=_PATH_COLOR,
                    linewidth=_PATH_LW,
                    solid_capstyle="round",
                    zorder=2,
                )


def _draw_waypoints(ax: Axes, puzzle: Puzzle) -> None:
    """Overlay waypoint numbers at each waypoint cell centre."""
    for wp in puzzle.waypoints:
        ax.text(
            wp.col + 0.5,
            wp.row + 0.5,
            str(wp.number),
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=_WAYPOINT_COLOR,
            zorder=4,
        )


def _finalize_axes(ax: Axes, n: int) -> None:
    """Lock aspect, flip y, hide ticks/spines — matches puzzle row-0-top convention."""
    ax.set_xlim(-0.05, n + 0.05)
    ax.set_ylim(-0.05, n + 0.05)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_path_layer(
    puzzle: Puzzle,
    shapes: NDArray[np.int8],
    *,
    confidence: NDArray[np.float64] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Draw Panel A: grid, walls, path shapes, and waypoint numbers.

    Args:
        puzzle: The puzzle whose geometry to draw.
        shapes: ``NDArray[np.int8]`` of shape
            ``(puzzle.size, puzzle.size)``. Entries are
            :class:`Shape` indices in ``0..9``; ``-1`` means
            "no shape at this cell" (nothing drawn there).
        confidence: Optional ``(N, N)`` float array in ``[0, 1]``
            tinting each cell's facecolor via the ``"Blues"`` cmap.
            Unset in Phase 3; wired up by Phase 5 once probability
            tensors exist.
        ax: Axes to draw onto. If ``None``, uses the current axes.

    Returns:
        The same :class:`Axes` with all artists added. Caller owns
        ``show`` / ``savefig``.

    Raises:
        ValueError: If ``shapes`` or ``confidence`` have the wrong
            dimensions.
    """
    n = puzzle.size
    if shapes.shape != (n, n):
        msg = f"shapes must have shape ({n}, {n}); got {shapes.shape}"
        raise ValueError(msg)
    if confidence is not None and confidence.shape != (n, n):
        msg = f"confidence must have shape ({n}, {n}); got {confidence.shape}"
        raise ValueError(msg)

    resolved_ax = ax if ax is not None else plt.gca()

    if confidence is not None:
        _draw_confidence(resolved_ax, confidence, n)
    _draw_grid(resolved_ax, n)
    _draw_walls(resolved_ax, puzzle)
    _draw_shapes(resolved_ax, shapes, n)
    _draw_waypoints(resolved_ax, puzzle)
    _finalize_axes(resolved_ax, n)

    return resolved_ax
