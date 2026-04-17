"""Phase 8B margin diagnostic: quantify score separation at failing cells.

After Phase 8A (weight sweep, peak 5/9) and Phase 8B (symmetry-breakers,
peak 7/9) both failed their argmax gates, we need to decide between
Phase 8C (substrate redesign) and a scoring-formula amendment. Decision
criterion: the *magnitude* of ``score[expected] - score[argmax]`` at the
persistent failure cells.

* If the gap <= 0.3 under consensus, a cheap scoring-formula fix
  (through-shape prior, K-source normalisation) is likely sufficient.
* If the gap >= 1.0, the substrate cannot express the right signal and
  Phase 8C substrate redesign is justified.

This script dumps, for each (substrate in {baseline, Option H}) x
(weight in {0.0, 10.0}) x (failing cell in {(0,0), (0,1), (1,1), (2,2)}):

* full raw gradient score vector,
* full consensus-augmented score vector,
* expected-shape score,
* argmax-shape score,
* margin = argmax - expected (positive = expected is losing).
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from zip_ca import (
    Shape,
    build_allowed_mask,
    build_open_mutual,
    build_sources,
    diffuse_tick,
    init_chems,
    load_puzzle,
)
from zip_ca.puzzle import Puzzle
from zip_ca.scoring import (
    CONSENSUS_ITERATIONS,
    _build_gradients,
    _consensus_bonus,
    _endpoint_score,
)
from zip_ca.shapes import NUM_SHAPES, THROUGH_SHAPES, open_ports

_PORT_ORDER: Final[tuple[str, ...]] = ("N", "E", "S", "W")
_PORT_TO_IDX: Final[dict[str, int]] = {p: i for i, p in enumerate(_PORT_ORDER)}
_QUIESCENCE_TOL: Final[float] = 1e-6
_MAX_WARMUP_TICKS: Final[int] = 5000
_HAMILTONIAN: Final[dict[tuple[int, int], Shape]] = {
    (0, 0): Shape.END_E, (0, 1): Shape.H,     (0, 2): Shape.SW,
    (1, 0): Shape.SE,    (1, 1): Shape.H,     (1, 2): Shape.NW,
    (2, 0): Shape.NE,    (2, 1): Shape.H,     (2, 2): Shape.END_W,
}
_FAILURE_CELLS: Final[tuple[tuple[int, int], ...]] = (
    (0, 0), (0, 1), (1, 1), (2, 2),
)
_WEIGHTS: Final[tuple[float, ...]] = (0.0, 10.0)


def _quiesce(
    chems: NDArray[np.float32],
    mutual_open: NDArray[np.bool_],
    sources: NDArray[np.int64],
) -> NDArray[np.float32]:
    for _ in range(_MAX_WARMUP_TICKS):
        new_chems = diffuse_tick(chems, mutual_open, sources)
        if float(np.max(np.abs(new_chems - chems))) < _QUIESCENCE_TOL:
            return new_chems
        chems = new_chems
    return chems


def _through_score_baseline(
    gradients: NDArray[np.float32], a: int, b: int,
) -> NDArray[np.float32]:
    return (gradients[..., a, :] + gradients[..., b, :]).squeeze(-1).astype(np.float32)


def _through_score_optionh(
    gradients: NDArray[np.float32], a: int, b: int,
) -> NDArray[np.float32]:
    g_a = gradients[..., a, :]
    g_b = gradients[..., b, :]
    fwd = g_a[..., :-1] + g_b[..., 1:]
    bwd = g_a[..., 1:] + g_b[..., :-1]
    return np.maximum(fwd, bwd).max(axis=-1).astype(np.float32)


def _base_scores(
    chems: NDArray[np.float32], n: int, *, through_kind: str,
) -> NDArray[np.float32]:
    gradients = _build_gradients(chems, n)
    scores = np.full((n, n, NUM_SHAPES), -np.inf, dtype=np.float32)
    for shape in Shape:
        ports = open_ports(shape)
        port_indices = sorted(_PORT_TO_IDX[p.name] for p in ports)
        if shape in THROUGH_SHAPES:
            a, b = port_indices
            if through_kind == "baseline":
                scores[..., int(shape)] = _through_score_baseline(gradients, a, b)
            else:
                scores[..., int(shape)] = _through_score_optionh(gradients, a, b)
        else:
            (a,) = port_indices
            scores[..., int(shape)] = _endpoint_score(gradients, a)
    return scores


def _apply_consensus(
    base: NDArray[np.float32],
    allowed: NDArray[np.bool_],
    n: int,
    weight: float,
) -> NDArray[np.float32]:
    augmented = base
    for _ in range(CONSENSUS_ITERATIONS):
        bonus = _consensus_bonus(augmented, n)
        augmented = np.where(
            allowed, base + np.float32(weight) * bonus, np.float32(-np.inf),
        ).astype(np.float32)
    return augmented


def _baseline_substrate(puzzle: Puzzle) -> NDArray[np.float32]:
    chems = init_chems(puzzle)
    sources = build_sources(puzzle)
    mutual_open = build_open_mutual(puzzle)
    return _quiesce(chems, mutual_open, sources)


def _optionh_substrate(puzzle: Puzzle) -> NDArray[np.float32]:
    n = puzzle.size
    k_sources = len(puzzle.waypoints)
    chems = np.zeros((n, n, k_sources), dtype=np.float32)
    sources = np.array(
        [(wp.row, wp.col, wp.number - 1) for wp in puzzle.waypoints],
        dtype=np.int64,
    )
    mutual_open = build_open_mutual(puzzle)
    return _quiesce(chems, mutual_open, sources)


def _report_cell(
    cell: tuple[int, int],
    scores: NDArray[np.float32],
    allowed: NDArray[np.bool_],
) -> None:
    r, c = cell
    row = scores[r, c]
    mask = allowed[r, c]
    masked = np.where(mask, row, -np.inf)
    expected = _HAMILTONIAN[cell]
    top_idx = int(np.argmax(masked))
    argmax_shape = Shape(top_idx)
    expected_s = float(row[int(expected)])
    argmax_s = float(row[top_idx])
    margin = argmax_s - expected_s
    print(f"  cell ({r},{c})  expected={expected.name:<6} argmax={argmax_shape.name:<6}"
          f"  expected_score={expected_s:+.4f}  argmax_score={argmax_s:+.4f}"
          f"  margin={margin:+.4f}")
    allowed_shapes = [Shape(i) for i, allowed_i in enumerate(mask) if allowed_i]
    ranked = sorted(allowed_shapes, key=lambda s: -row[int(s)])[:5]
    top_line = "    top-5: " + ", ".join(
        f"{s.name}={row[int(s)]:+.4f}" for s in ranked
    )
    print(top_line)


def main() -> int:
    puzzle = load_puzzle("puzzles/tiny_3x3.json")
    allowed = build_allowed_mask(puzzle)
    n = puzzle.size

    print("Building steady-state substrates...")
    substrates: dict[str, tuple[NDArray[np.float32], str]] = {
        "baseline": (_baseline_substrate(puzzle), "baseline"),
        "optionH":  (_optionh_substrate(puzzle), "optionh"),
    }

    for substrate_name, (chems, through_kind) in substrates.items():
        base_raw = _base_scores(chems, n, through_kind=through_kind)
        base = np.where(allowed, base_raw, np.float32(-np.inf)).astype(np.float32)
        for weight in _WEIGHTS:
            scores = _apply_consensus(base, allowed, n, weight)
            print("\n" + "=" * 78)
            print(f"substrate={substrate_name}  CONSENSUS_WEIGHT={weight}")
            print("=" * 78)
            for cell in _FAILURE_CELLS:
                _report_cell(cell, scores, allowed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
