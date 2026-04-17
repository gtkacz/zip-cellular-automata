r"""Dev driver: delegate to :func:`zip_ca.solve` and render the best run.

Usage:
    uv run python scripts/solve.py \
        puzzles/tiny_3x3.json out/solve_tiny_3x3.png

Phase 6 repurposes this script: the restart loop now lives inside the
library (:mod:`zip_ca.solver`), and this script is only a thin CLI
wrapper that:

1. loads the puzzle,
2. calls :func:`solve` with ``R_MAX`` runs and ``T_MAX`` ticks,
3. replays the best run up to its reported tick count to recover the
   final engine state for visualisation (the library intentionally
   does not carry the final :class:`EngineState` inside
   :class:`SolveResult` - that would bloat the return type for the
   library-internal consumers),
4. writes the Panel A + Panel B figure regardless of outcome.

Exit codes match the Phase 5 contract: 0 = SOLVED, 1 =
QUIESCENT-BUT-INVALID at the best run, 2 = all runs timed out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from zip_ca import (
    R_MAX,
    T_MAX,
    EngineState,
    Puzzle,
    SolveResult,
    load_puzzle,
    render_chem_layer,
    render_path_layer,
    solve,
)

_SUBPLOT_SIZE_INCHES: Final[float] = 4.0


def _replay_to(puzzle: Puzzle, run_id: int, tick_count: int) -> EngineState:
    """Reconstruct the engine state at the end of the given run.

    The solver discards every :class:`EngineState` after a run ends;
    the only reproducible way to recover the final field for
    visualisation is to re-seed with the same ``run_id`` and re-tick
    exactly ``tick_count`` times. The §11.3 reproducibility invariant
    guarantees byte-equal replay.
    """
    engine = EngineState.fresh(puzzle, run_id=run_id)
    for _ in range(tick_count):
        engine.tick()
    return engine


def _render(engine: EngineState, output_path: Path) -> None:
    """Render Panel A (path) + Panel B (chemicals) side-by-side to PNG."""
    k_segments = engine.chems.shape[2]
    total_cols = 1 + k_segments
    fig, axes = plt.subplots(
        1,
        total_cols,
        figsize=(_SUBPLOT_SIZE_INCHES * total_cols, _SUBPLOT_SIZE_INCHES),
    )
    axes_list = [cast(Axes, ax) for ax in axes]  # type: ignore[union-attr]
    ax_panel_a = axes_list[0]
    axes_panel_b = axes_list[1:]

    render_path_layer(engine.puzzle, engine.shapes, ax=ax_panel_a)
    ax_panel_a.set_title("Panel A - path layer")
    render_chem_layer(engine.puzzle, engine.chems, axes=axes_panel_b)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _report(result: SolveResult) -> None:
    """Print a one-line summary per run and a headline for the best run."""
    for run in result.runs:
        prefix_len = len(run.trace.path)
        print(
            f"  run={run.run_id:2d} "
            f"ticks={run.tick_count:4d} "
            f"outcome={run.outcome:18s} "
            f"prefix={prefix_len:2d} "
            f"stable={run.stable_ticks:3d}",
        )
    best = result.runs[result.best_run_id]
    if result.ok:
        print(f"SOLVED by run_id={result.best_run_id} at tick {best.tick_count}")
        print("path:", " -> ".join(f"({c.x},{c.y})" for c in result.best_trace.path))
    else:
        print(
            f"FAILED after {len(result.runs)} runs; "
            f"best run_id={result.best_run_id} "
            f"prefix={len(result.best_trace.path)} "
            f"reason={result.best_trace.reason!r}",
        )
        if result.best_trace.path:
            print(
                "longest valid prefix:",
                " -> ".join(f"({c.x},{c.y})" for c in result.best_trace.path),
            )


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on solve, 1 on quiescent-invalid best, 2 on all timeouts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("puzzle", type=Path, help="Path to puzzle JSON")
    parser.add_argument("output", type=Path, help="Output PNG path")
    parser.add_argument(
        "--max-runs",
        type=int,
        default=R_MAX,
        help=f"Max runs (default {R_MAX})",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=T_MAX,
        help=f"Max ticks per run (default {T_MAX})",
    )
    args = parser.parse_args(argv)

    puzzle = load_puzzle(cast(Path, args.puzzle))
    result = solve(puzzle, max_runs=int(args.max_runs), max_ticks=int(args.max_ticks))

    _report(result)

    best = result.runs[result.best_run_id]
    final_engine = _replay_to(puzzle, run_id=result.best_run_id, tick_count=best.tick_count)
    _render(final_engine, cast(Path, args.output))

    if result.ok:
        return 0
    if all(run.outcome == "timeout" for run in result.runs):
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
