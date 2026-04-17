"""Trace the per-cell argmax through a Phase 7 solve on tiny_3x3.

Phase 7's acceptance report is "prefix=1/9" — i.e., the derived path
extends only to the starting waypoint. The critique of the Phase 8
plan argued that "prefix=1/9" does not identify WHICH cells go wrong,
only that the reconstructed path breaks after one cell. This script
instruments a full engine run and prints the argmax grid at each
milestone tick, so we can see the actual failure mode.

Instruments:

* The first tick after warm-up (T_WARM = 20)
* Every 50 ticks thereafter until tick 500
* The quiescence tick (if any)

At each checkpoint we print the full shape grid, the softmax prob
row at each cell, and which cells deviate from the Hamiltonian.
"""

from __future__ import annotations

import numpy as np

from zip_ca import EngineState, Shape, load_puzzle
from zip_ca.engine import T_WARM

_CHECKPOINT_TICKS = (T_WARM, T_WARM + 1, T_WARM + 10, T_WARM + 50, T_WARM + 200, T_WARM + 500)
_HAMILTONIAN: dict[tuple[int, int], Shape] = {
    (0, 0): Shape.END_E, (0, 1): Shape.H,     (0, 2): Shape.SW,
    (1, 0): Shape.SE,    (1, 1): Shape.H,     (1, 2): Shape.NW,
    (2, 0): Shape.NE,    (2, 1): Shape.H,     (2, 2): Shape.END_W,
}


def _print_shape_grid(state: EngineState) -> None:
    n = state.puzzle.size
    for i in range(n):
        row_shapes = [Shape(int(state.shapes[i, j])).name for j in range(n)]
        print("  " + "  ".join(f"{s:6s}" for s in row_shapes))


def _print_prob_rows(state: EngineState) -> None:
    n = state.puzzle.size
    for i in range(n):
        for j in range(n):
            row = state.probs[i, j]
            entries = []
            for k, s in enumerate(Shape):
                if state.allowed[i, j, k]:
                    entries.append(f"{s.name}={row[k]:.3f}")
            top = int(np.argmax(row))
            marker = "*" if Shape(top) == _HAMILTONIAN.get((i, j)) else " "
            print(
                f"  ({i},{j})[{marker}] top={Shape(top).name:6s}  "
                + "  ".join(entries),
            )


def main() -> int:
    puzzle = load_puzzle("puzzles/tiny_3x3.json")
    state = EngineState.fresh(puzzle, run_id=0)
    max_tick = max(_CHECKPOINT_TICKS) + 1

    for tick_index in range(max_tick):
        state.tick()
        if state.tick_count in _CHECKPOINT_TICKS:
            print(f"\n{'=' * 68}")
            print(f"Tick {state.tick_count} (post-warm-up by {state.tick_count - T_WARM})")
            print("=" * 68)
            print("\nArgmax shape grid:")
            _print_shape_grid(state)

            mismatch = {
                (r, c): Shape(int(state.shapes[r, c]))
                for (r, c), expected in _HAMILTONIAN.items()
                if Shape(int(state.shapes[r, c])) != expected
            }
            print(f"\nMismatches with Hamiltonian: {len(mismatch)}/9")
            for (r, c), got in mismatch.items():
                expected = _HAMILTONIAN[(r, c)]
                print(f"  ({r},{c}) expected {expected.name:6s} got {got.name:6s}")

            print("\nProb rows (allowed shapes only; * = top matches Hamiltonian):")
            _print_prob_rows(state)

    print(f"\n{'=' * 68}")
    print("Final shape grid:")
    _print_shape_grid(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
