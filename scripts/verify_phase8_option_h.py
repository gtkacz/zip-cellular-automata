"""Falsify or confirm Phase 8 Option H's (1,1) score-separation claim.

The Phase 8 plan claims that replacing the K-1 two-endpoint segment
chemistry with K single-source chemistries will make ``H`` strictly
exceed every corner-shape at the central cell ``(1,1)`` on
``tiny_3x3`` by a margin >= 0.01. The critique argues this fails by
grid-Laplacian symmetry. This script settles the question empirically.

tiny_3x3 has a wall blocking (1,1)<->(2,1), so the allowed-shape set
at (1,1) excludes every shape that opens the S port. We filter the
score vector by ``build_allowed_mask`` before ranking so the
acceptance test sees only physically-realisable candidates.
"""

from __future__ import annotations

import numpy as np

from zip_ca import (
    Shape,
    build_allowed_mask,
    build_open_mutual,
    diffuse_tick,
    load_puzzle,
    open_ports,
)
from zip_ca.shapes import THROUGH_SHAPES

_PORT_ORDER = ("N", "E", "S", "W")
_PORT_TO_IDX = {p: i for i, p in enumerate(_PORT_ORDER)}
_QUIESCENCE_TOL = 1e-6
_MAX_WARMUP_TICKS = 5000
_TIE_TOL = 1e-5


def _build_gradients(chems: np.ndarray, n: int) -> np.ndarray:
    u_north = np.roll(chems, shift=+1, axis=0)
    u_east = np.roll(chems, shift=-1, axis=1)
    u_south = np.roll(chems, shift=-1, axis=0)
    u_west = np.roll(chems, shift=+1, axis=1)
    u_neigh = np.stack([u_north, u_east, u_south, u_west], axis=2)
    gradients = (u_neigh - chems[..., None, :]).astype(np.float32)
    gradients[0, :, 0, :] = 0.0
    gradients[:, n - 1, 1, :] = 0.0
    gradients[n - 1, :, 2, :] = 0.0
    gradients[:, 0, 3, :] = 0.0
    return gradients


def _paired_through_score(
    gradients: np.ndarray, a: int, b: int, k_sources: int,
) -> np.ndarray:
    g_a = gradients[..., a, :]
    g_b = gradients[..., b, :]
    if k_sources == 1:
        return (g_a + g_b).squeeze(-1).astype(np.float32)
    fwd = g_a[..., :-1] + g_b[..., 1:]
    bwd = g_a[..., 1:] + g_b[..., :-1]
    return np.maximum(fwd, bwd).max(axis=-1).astype(np.float32)


def _endpoint_score(gradients: np.ndarray, a: int) -> np.ndarray:
    return gradients[..., a, :].max(axis=-1).astype(np.float32)


def _compute_all_scores(
    gradients: np.ndarray, k_sources: int,
) -> dict[Shape, np.ndarray]:
    scores: dict[Shape, np.ndarray] = {}
    for shape in Shape:
        ports = open_ports(shape)
        port_indices = sorted(_PORT_TO_IDX[p.name] for p in ports)
        if shape in THROUGH_SHAPES:
            a, b = port_indices
            scores[shape] = _paired_through_score(gradients, a, b, k_sources)
        else:
            (a,) = port_indices
            scores[shape] = _endpoint_score(gradients, a)
    return scores


def _print_cell(
    row: int, col: int,
    scores: dict[Shape, np.ndarray],
    allowed: np.ndarray,
) -> tuple[Shape | None, float, float]:
    """Print ranked allowed scores at (row,col). Returns (top, spread, h_minus_best_other)."""
    cell_allowed = [s for s in Shape if bool(allowed[row, col, int(s)])]
    ranked = sorted(
        ((s, float(scores[s][row, col])) for s in cell_allowed),
        key=lambda kv: kv[1],
        reverse=True,
    )
    if not ranked:
        print(f"  ({row},{col}): NO ALLOWED SHAPES")
        return None, 0.0, 0.0

    print(f"  ({row},{col}):  allowed={[s.name for s in cell_allowed]}")
    for shape, score in ranked:
        marker = "  *" if shape == ranked[0][0] else "   "
        print(f"{marker} {shape.name:7s} = {score:+.6f}")

    top_shape, top_score = ranked[0]
    spread = top_score - ranked[1][1] if len(ranked) > 1 else float("inf")
    others = [v for s, v in ranked if s != Shape.H]
    if Shape.H in [s for s, _ in ranked] and others:
        h_score = scores[Shape.H][row, col]
        h_margin = float(h_score) - float(max(others))
    else:
        h_margin = float("nan")
    return top_shape, spread, h_margin


def main() -> int:
    puzzle = load_puzzle("puzzles/tiny_3x3.json")
    n = puzzle.size
    k_sources = len(puzzle.waypoints)

    chems = np.zeros((n, n, k_sources), dtype=np.float32)
    sources = np.array(
        [(wp.row, wp.col, wp.number - 1) for wp in puzzle.waypoints],
        dtype=np.int64,
    )
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
        print(f"Diffusion did not quiesce in {_MAX_WARMUP_TICKS} ticks (delta={delta:.2e})")

    print("\n=== Steady-state K=2 source fields ===")
    for k in range(k_sources):
        wp = puzzle.waypoints[k]
        print(f"\nchem_{k + 1} (clamped at W_{wp.number}=({wp.row},{wp.col})):")
        for i in range(n):
            print("  " + "  ".join(f"{chems[i, j, k]:.4f}" for j in range(n)))

    gradients = _build_gradients(chems, n)
    scores = _compute_all_scores(gradients, k_sources)
    allowed = build_allowed_mask(puzzle)

    print("\n=== Allowed-filtered shape scores, all cells ===")
    hamiltonian = {
        (0, 0): Shape.END_E, (0, 1): Shape.H,     (0, 2): Shape.SW,
        (1, 0): Shape.SE,    (1, 1): Shape.H,     (1, 2): Shape.NW,
        (2, 0): Shape.NE,    (2, 1): Shape.H,     (2, 2): Shape.END_W,
    }

    correct_at_top = 0
    total = 0
    for (r, c), expected in hamiltonian.items():
        print(f"\nCell ({r},{c}) expected {expected.name}:")
        top_shape, spread, h_margin = _print_cell(r, c, scores, allowed)
        total += 1
        if top_shape == expected:
            correct_at_top += 1
            verdict = "OK top matches Hamiltonian"
        else:
            verdict = f"WRONG top={top_shape.name if top_shape else 'None'}"
        print(
            f"  -> {verdict}  spread={spread:+.6f}  "
            f"h_minus_best_other={h_margin:+.6f}",
        )

    print(f"\n=== Summary: {correct_at_top}/{total} cells argmax == Hamiltonian ===")

    # Phase 8 specific acceptance: H > best corner at (1,1) by >= 0.01.
    h_at_center = float(scores[Shape.H][1, 1])
    corner_shapes = (Shape.NE, Shape.NW, Shape.SE, Shape.SW)
    allowed_corners_at_center = [
        s for s in corner_shapes if bool(allowed[1, 1, int(s)])
    ]
    best_corner = max(
        float(scores[s][1, 1]) for s in allowed_corners_at_center
    ) if allowed_corners_at_center else float("-inf")
    center_margin = h_at_center - best_corner
    print(
        f"\nAt (1,1): H = {h_at_center:+.6f}; "
        f"best allowed corner = {best_corner:+.6f} "
        f"({[s.name for s in allowed_corners_at_center]}); "
        f"H - best_corner = {center_margin:+.6f}",
    )
    print(
        f"Phase 8 acceptance (H - best_corner >= 0.01 at (1,1)): "
        f"{'SATISFIED' if center_margin >= 0.01 else 'FALSIFIED'}",
    )
    return 0 if correct_at_top == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
