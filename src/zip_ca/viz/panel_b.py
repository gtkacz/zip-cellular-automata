"""Panel B renderer: chemical-layer visualisation.

Implements ``docs/design.md`` §10 Panel B in **multi-channel mode**:
a horizontal strip of ``K-1`` heatmaps, one per segment chemical,
sharing a fixed ``[0, C0]`` colour scale so the operator can
compare channels by eye. Composite mode (single heatmap with
hue = ``argmax_k u[k]``, value = ``max_k u[k]``) is deferred to
Phase 7 polish — multi-channel is the diagnostic view Phase 4
needs.

The renderer is **pure**: it receives a sequence of
:class:`~matplotlib.axes.Axes`, draws onto them, and returns the
same sequence. It never calls ``plt.show`` or ``savefig``. Caller
owns the figure lifecycle, matching the contract of
:func:`~zip_ca.viz.panel_a.render_path_layer`.

Coordinate convention: ``imshow`` is called with ``origin="upper"``
so row ``0`` of ``chems`` lands at the top of each subplot,
matching the row-0-top convention used by Panel A (which inverts
the y-axis to reach the same visual result).
"""

from collections.abc import Sequence
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from zip_ca.diffusion import C0
from zip_ca.puzzle import Puzzle

_WAYPOINT_COLOR = "#d62728"
_WAYPOINT_FONTSIZE = 12
_EXPECTED_NDIM = 3


def _draw_single_channel(
    ax: Axes,
    puzzle: Puzzle,
    field: NDArray[np.float32],
    segment_number: int,
    cmap: str,
) -> None:
    """Draw one segment's heatmap plus waypoint labels onto ``ax``."""
    ax.imshow(
        field,
        cmap=cmap,
        vmin=0.0,
        vmax=float(C0),
        origin="upper",
        interpolation="nearest",
    )
    for wp in puzzle.waypoints:
        ax.text(
            wp.col,
            wp.row,
            str(wp.number),
            ha="center",
            va="center",
            fontsize=_WAYPOINT_FONTSIZE,
            fontweight="bold",
            color=_WAYPOINT_COLOR,
        )
    ax.set_title(f"segment {segment_number} (W{segment_number}→W{segment_number + 1})")
    ax.set_xticks([])
    ax.set_yticks([])


def render_chem_layer(
    puzzle: Puzzle,
    chems: NDArray[np.float32],
    *,
    axes: Sequence[Axes] | None = None,
    cmap: str = "viridis",
) -> Sequence[Axes]:
    """Draw Panel B: multi-channel heatmap strip, one per segment.

    Args:
        puzzle: The puzzle whose waypoints label each heatmap.
        chems: ``NDArray[np.float32]`` of shape
            ``(puzzle.size, puzzle.size, K-1)`` — one 2-D
            concentration field per segment.
        axes: Pre-allocated sequence of length ``K-1``. If ``None``,
            a new figure with ``K-1`` horizontally-arranged subplots
            is created; the caller then has no reference to the
            figure, so pass ``axes`` explicitly when lifecycle
            control matters.
        cmap: Any matplotlib colormap name; ``"viridis"`` is the
            default because its perceptual uniformity makes gradient
            monotonicity visually obvious.

    Returns:
        The same sequence of axes, in the same order. Each axis has
        been mutated with the segment's heatmap and waypoint labels.

    Raises:
        ValueError: If ``chems`` has the wrong 2-D dimensions, or if
            ``axes`` (when supplied) has a length other than ``K-1``.
    """
    n = puzzle.size
    if chems.ndim != _EXPECTED_NDIM or chems.shape[:2] != (n, n):
        msg = (
            f"chems must have shape ({n}, {n}, K-1); got {chems.shape}"
        )
        raise ValueError(msg)

    k_segments = chems.shape[2]

    resolved_axes: Sequence[Axes]
    if axes is None:
        _fig, new_axes = plt.subplots(1, k_segments, figsize=(4 * k_segments, 4))
        # plt.subplots returns a bare Axes when k_segments == 1; wrap so
        # the rest of the function sees a uniform sequence.
        resolved_axes = (
            [new_axes]
            if k_segments == 1
            else [cast(Axes, ax) for ax in new_axes]  # type: ignore[union-attr]
        )
    else:
        if len(axes) != k_segments:
            msg = f"axes length {len(axes)} does not match K-1 = {k_segments}"
            raise ValueError(msg)
        resolved_axes = axes

    for k in range(k_segments):
        _draw_single_channel(
            resolved_axes[k],
            puzzle,
            chems[..., k],
            segment_number=k + 1,
            cmap=cmap,
        )

    return resolved_axes
