r"""Dev script: iterate Layer-2 diffusion over a fixed solution path.

Usage:
    uv run python scripts/render_diffusion.py \
        puzzles/tiny_3x3.json puzzles/tiny_3x3.solution.json \
        out/diffusion_tiny_3x3.png

Loads a puzzle + hand-supplied solution path, derives the shape
grid, runs ``diffuse_tick`` to quasi-steady state (L∞ delta below
:data:`_CONVERGENCE_EPS` or at most :data:`_MAX_TICKS`), and
renders Panel A + Panel B side-by-side to PNG.

Lives under ``scripts/`` rather than ``src/zip_ca/`` because
``docs/design.md`` §11.4.2 forbids any solver-code module from
importing a reader of ``*.solution.json``. Keeping the solution
loader in a top-level script preserves that boundary mechanically.

The ``np.roll`` wrap-around semantics inside ``diffuse_tick`` were
spot-checked at plan-review time on a 2x2 grid where every roll
either goes off-grid or wraps; the result equals a hand-computed
reference because ``mutual_open`` has ``False`` for every
out-of-bounds neighbour, zeroing the wrapped contributions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from zip_ca import (
    build_mutual_open,
    build_sources,
    diffuse_tick,
    init_chems,
    load_puzzle,
    path_to_shapes,
    render_chem_layer,
    render_path_layer,
)
from zip_ca.geometry import Cell

_CELL_PAIR_LENGTH = 2
_CONVERGENCE_EPS = 1e-5
_MAX_TICKS = 500
_SUBPLOT_SIZE_INCHES = 4.0


def _load_solution_path(path_file: Path) -> list[Cell]:
    """Load a ``*.solution.json`` and return its ``path`` as a list of cells."""
    with path_file.open() as f:
        raw: Any = json.load(f)
    if not isinstance(raw, dict) or "path" not in raw:
        msg = f"Solution file {path_file} missing 'path' key"
        raise ValueError(msg)
    raw_dict = cast(dict[str, Any], raw)
    raw_path = raw_dict["path"]
    if not isinstance(raw_path, list):
        msg = f"Solution 'path' in {path_file} must be a list"
        raise TypeError(msg)
    raw_path_list = cast(list[Any], raw_path)
    cells: list[Cell] = []
    for item in raw_path_list:
        if not (isinstance(item, list) and len(cast(list[Any], item)) == _CELL_PAIR_LENGTH):
            msg = f"Path entries must be [row, col] pairs; got {item!r}"
            raise ValueError(msg)
        pair = cast(list[Any], item)
        row, col = pair[0], pair[1]
        if not (isinstance(row, int) and isinstance(col, int)):
            msg = f"Path entries must be integer pairs; got {item!r}"
            raise TypeError(msg)
        cells.append(Cell(row, col))
    return cells


def _iterate_to_steady_state(
    chems: np.ndarray,  # type: ignore[type-arg]
    mutual_open: np.ndarray,  # type: ignore[type-arg]
    sources: np.ndarray,  # type: ignore[type-arg]
) -> tuple[np.ndarray, int, float]:  # type: ignore[type-arg]
    """Run ``diffuse_tick`` until L∞ delta falls below ``_CONVERGENCE_EPS``.

    Returns the final field, the tick count reached, and the last
    measured delta. The loop always completes at least one tick so
    the returned delta reflects a real measurement.
    """
    tick = 0
    delta = float("inf")
    current = chems
    while tick < _MAX_TICKS:
        tick += 1
        nxt = diffuse_tick(current, mutual_open, sources)
        delta = float(np.max(np.abs(nxt - current)))
        current = nxt
        if delta < _CONVERGENCE_EPS:
            break
    return current, tick, delta


def _report(chems: np.ndarray, tick: int, delta: float) -> None:  # type: ignore[type-arg]
    """Print per-segment diagnostic stats to stdout."""
    print(f"converged at tick {tick} (Δ_∞ = {delta:.2e})")
    for k in range(chems.shape[2]):
        field = chems[..., k]
        print(
            f"  segment {k + 1}: peak = {float(field.max()):.4f}, "
            f"floor = {float(field.min()):.4f}, mean = {float(field.mean()):.4f}"
        )


def main(argv: list[str] | None = None) -> int:
    """Entry point for the dev script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("puzzle", type=Path, help="Path to puzzle JSON")
    parser.add_argument("solution", type=Path, help="Path to solution JSON")
    parser.add_argument("output", type=Path, help="Output PNG path")
    args = parser.parse_args(argv)

    puzzle = load_puzzle(cast(Path, args.puzzle))
    path = _load_solution_path(cast(Path, args.solution))
    shapes = path_to_shapes(path, puzzle)

    chems = init_chems(puzzle)
    mutual = build_mutual_open(shapes, puzzle)
    sources = build_sources(puzzle)

    chems, tick, delta = _iterate_to_steady_state(chems, mutual, sources)
    _report(chems, tick, delta)

    k_segments = chems.shape[2]
    total_cols = 1 + k_segments
    fig, axes = plt.subplots(
        1,
        total_cols,
        figsize=(_SUBPLOT_SIZE_INCHES * total_cols, _SUBPLOT_SIZE_INCHES),
    )
    axes_list = [cast(Axes, ax) for ax in axes]  # type: ignore[union-attr]
    ax_panel_a = axes_list[0]
    axes_panel_b = axes_list[1:]

    render_path_layer(puzzle, shapes, ax=ax_panel_a)
    ax_panel_a.set_title("Panel A — path layer")
    render_chem_layer(puzzle, chems, axes=axes_panel_b)

    output_path = cast(Path, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return 0


if __name__ == "__main__":
    sys.exit(main())
