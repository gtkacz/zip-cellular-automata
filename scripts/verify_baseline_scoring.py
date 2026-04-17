"""Baseline: run the same argmax diagnostic on the CURRENT K-1 chemistry.

For a fair comparison against ``verify_phase8_option_h.py``, we use
the production ``score_shapes`` (which includes the Phase 5
consensus bonus) fed by the existing ``init_chems`` / ``build_sources``
/ ``build_open_mutual`` pipeline at steady state.
"""

from __future__ import annotations

import numpy as np

from zip_ca import (
    Shape,
    build_allowed_mask,
    build_open_mutual,
    build_sources,
    diffuse_tick,
    init_chems,
    load_puzzle,
    score_shapes,
)

_QUIESCENCE_TOL = 1e-6
_MAX_WARMUP_TICKS = 5000


def main() -> int:
    puzzle = load_puzzle("puzzles/tiny_3x3.json")
    n = puzzle.size

    chems = init_chems(puzzle)
    sources = build_sources(puzzle)
    mutual_open = build_open_mutual(puzzle)

    delta = float("inf")
    for tick in range(_MAX_WARMUP_TICKS):
        new_chems = diffuse_tick(chems, mutual_open, sources)
        delta = float(np.max(np.abs(new_chems - chems)))
        chems = new_chems
        if delta < _QUIESCENCE_TOL:
            print(f"Diffusion quiesced at tick {tick + 1} (delta={delta:.2e})")
            break
    else:
        print(f"Did not quiesce in {_MAX_WARMUP_TICKS} ticks (delta={delta:.2e})")

    print(f"\nchems shape = {chems.shape} (K-1 = {chems.shape[2]} segment chemicals)")
    print("\nchem (single segment, clamped at W_1 and W_2):")
    for i in range(n):
        print("  " + "  ".join(f"{chems[i, j, 0]:.4f}" for j in range(n)))

    allowed = build_allowed_mask(puzzle)
    scores = score_shapes(chems, puzzle, allowed)

    hamiltonian = {
        (0, 0): Shape.END_E, (0, 1): Shape.H,     (0, 2): Shape.SW,
        (1, 0): Shape.SE,    (1, 1): Shape.H,     (1, 2): Shape.NW,
        (2, 0): Shape.NE,    (2, 1): Shape.H,     (2, 2): Shape.END_W,
    }

    print("\n=== Baseline K-1 chemistry + score_shapes (with consensus bonus) ===")
    correct = 0
    for (r, c), expected in hamiltonian.items():
        cell_scores = {
            s: float(scores[r, c, int(s)])
            for s in Shape
            if bool(allowed[r, c, int(s)])
        }
        ranked = sorted(cell_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_shape = ranked[0][0]
        is_correct = top_shape == expected
        correct += int(is_correct)
        marker = "OK   " if is_correct else "WRONG"
        spread = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else float("inf")
        print(
            f"  ({r},{c}) expected {expected.name:6s} "
            f"top={top_shape.name:6s} {marker}  spread={spread:+.6f}",
        )
        for shape, score in ranked:
            print(f"         {shape.name:7s} = {score:+.4f}")

    print(f"\n=== Baseline summary: {correct}/9 cells argmax == Hamiltonian ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
