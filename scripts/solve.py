r"""Dev driver: run :class:`~zip_ca.EngineState` to convergence on a puzzle.

Usage:
    uv run python scripts/solve.py \
        puzzles/tiny_3x3.json out/solve_tiny_3x3.png

Constructs a fresh engine, ticks until either (a) the shape grid is
quiescent and the trace validator accepts it (``SOLVED``), or (b)
``T_MAX`` ticks elapse without a valid solve (``TIMEOUT``). On exit
the final Panel A + Panel B figure is written to PNG regardless of
outcome so the operator can eyeball the failure mode.

This script is the Phase 5 acceptance harness; §12 row 4 says the
phase is complete when this returns 0 on ``tiny_3x3``. The script
lives under ``scripts/`` — not ``src/zip_ca/`` — because the
design's §11.4.2 layering forbids solver-code modules from anything
that might plausibly become aware of solution files, and scripts
sit above the library boundary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from zip_ca import (
    EngineState,
    QuiescenceDetector,
    TraceResult,
    load_puzzle,
    render_chem_layer,
    render_path_layer,
    trace_path,
)
from zip_ca.engine import T_WARM
from zip_ca.quiescence import T_STABLE

T_MAX: Final[int] = 5000
_SUBPLOT_SIZE_INCHES: Final[float] = 4.0


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
    ax_panel_a.set_title("Panel A — path layer")
    render_chem_layer(engine.puzzle, engine.chems, axes=axes_panel_b)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _run(
    engine: EngineState,
    qd: QuiescenceDetector,
    max_ticks: int,
    *,
    verbose: bool,
) -> TraceResult | None:
    """Tick the engine until solved, quiescent-but-invalid, or timeout.

    Returns the final :class:`TraceResult` if quiescence was reached
    (valid or not), ``None`` on timeout.
    """
    # Skip quiescence check during the warm-up ticks; `shapes` is
    # guaranteed unchanged there by the T_WARM invariant and reporting
    # "quiescent" would be a false positive.
    for _ in range(max_ticks):
        engine.tick()
        if engine.tick_count <= T_WARM:
            continue
        quiescent = qd.update(engine.shapes)
        if verbose and engine.tick_count % 50 == 0:
            print(
                f"  tick={engine.tick_count} "
                f"stable={qd.ticks_since_flip()} "
                f"probs.max={float(engine.probs.max()):.3f} "
                f"chems.max={float(engine.chems.max()):.3f}",
            )
        if quiescent:
            return trace_path(engine.shapes, engine.puzzle)
    return None


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on solve, 1 on quiescent-but-invalid, 2 on timeout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("puzzle", type=Path, help="Path to puzzle JSON")
    parser.add_argument("output", type=Path, help="Output PNG path")
    parser.add_argument("--run-id", type=int, default=0, help="Restart generation (Phase 5: always 0)")
    parser.add_argument("--max-ticks", type=int, default=T_MAX, help=f"Max ticks (default {T_MAX})")
    parser.add_argument("--verbose", action="store_true", help="Print per-tick diagnostics every 50 ticks")
    args = parser.parse_args(argv)

    puzzle = load_puzzle(cast(Path, args.puzzle))
    engine = EngineState.fresh(puzzle, run_id=int(args.run_id))
    qd = QuiescenceDetector(window=T_STABLE)

    result = _run(engine, qd, max_ticks=int(args.max_ticks), verbose=bool(args.verbose))
    _render(engine, cast(Path, args.output))

    if result is None:
        print(f"TIMEOUT after {engine.tick_count} ticks (max={args.max_ticks})")
        return 2
    if result.ok:
        print(f"SOLVED at tick {engine.tick_count}")
        print("path:", " -> ".join(f"({c.x},{c.y})" for c in result.path))
        return 0
    print(f"QUIESCENT BUT INVALID at tick {engine.tick_count}: {result.reason}")
    print("longest valid prefix:", " -> ".join(f"({c.x},{c.y})" for c in result.path))
    return 1


if __name__ == "__main__":
    sys.exit(main())
