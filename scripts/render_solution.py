r"""Dev script: render the Panel-A view of a hand-supplied solution.

Usage:
    uv run python scripts/render_solution.py \
        puzzles/tiny_3x3.json puzzles/tiny_3x3.solution.json out/tiny_3x3.png

Lives under ``scripts/`` rather than ``src/zip_ca/`` because
``docs/design.md`` §11.4.2 forbids any solver-code module from
importing a reader of ``*.solution.json``. Keeping the solution
loader in a top-level script preserves that boundary mechanically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt

from zip_ca import load_puzzle
from zip_ca.geometry import Cell
from zip_ca.path_shapes import path_to_shapes
from zip_ca.viz import render_path_layer

# A solution JSON path entry is always a two-element [row, col] list;
# 2 is structural, not a tuning knob.
_CELL_PAIR_LENGTH = 2


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

    fig, ax = plt.subplots(figsize=(6, 6))
    render_path_layer(puzzle, shapes, ax=ax)
    output_path = cast(Path, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return 0


if __name__ == "__main__":
    sys.exit(main())
