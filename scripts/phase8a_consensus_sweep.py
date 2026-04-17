"""Phase 8A gate: sweep CONSENSUS_WEIGHT across both substrate options.

The Phase 8 plan stipulates that before any substrate redesign or
score-level symmetry-breaker is committed, we first measure whether
the per-cell argmax failure on tiny_3x3 is actually dominated by the
raw chemistry gradients or by the consensus bonus. This script
answers that question by holding the chemistry at steady state and
sweeping CONSENSUS_WEIGHT in {0.0, 0.1, 1.0, 10.0} for:

* Substrate A — the production K-1 two-endpoint segment chemistry
  (build_sources clamps C0 at BOTH waypoints).
* Substrate B — Option H K-source chemistry (K independent fields,
  each clamped at one waypoint) with the paired-gradient through-
  shape score.

Gate outcome:

* If any (substrate, weight) combination yields 9/9 argmax-correct
  on tiny_3x3, commit to that combination and stop.
* Otherwise record the peak argmax-correct count per combination
  and proceed to Phase 8B.
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
_WEIGHTS: Final[tuple[float, ...]] = (0.0, 0.1, 1.0, 10.0)
_HAMILTONIAN: Final[dict[tuple[int, int], Shape]] = {
    (0, 0): Shape.END_E, (0, 1): Shape.H,     (0, 2): Shape.SW,
    (1, 0): Shape.SE,    (1, 1): Shape.H,     (1, 2): Shape.NW,
    (2, 0): Shape.NE,    (2, 1): Shape.H,     (2, 2): Shape.END_W,
}


def _quiesce(
    chems: NDArray[np.float32],
    mutual_open: NDArray[np.bool_],
    sources: NDArray[np.int64],
) -> tuple[NDArray[np.float32], int, float]:
    delta = float("inf")
    for tick in range(_MAX_WARMUP_TICKS):
        new_chems = diffuse_tick(chems, mutual_open, sources)
        delta = float(np.max(np.abs(new_chems - chems)))
        chems = new_chems
        if delta < _QUIESCENCE_TOL:
            return chems, tick + 1, delta
    return chems, _MAX_WARMUP_TICKS, delta


def _through_score_baseline(
    gradients: NDArray[np.float32], a: int, b: int,
) -> NDArray[np.float32]:
    """Same-segment sum for K-1 = 1 baseline substrate."""
    return (gradients[..., a, :] + gradients[..., b, :]).squeeze(-1).astype(np.float32)


def _through_score_optionh(
    gradients: NDArray[np.float32], a: int, b: int,
) -> NDArray[np.float32]:
    """Paired-gradient max over consecutive sources for Option H."""
    g_a = gradients[..., a, :]
    g_b = gradients[..., b, :]
    fwd = g_a[..., :-1] + g_b[..., 1:]
    bwd = g_a[..., 1:] + g_b[..., :-1]
    return np.maximum(fwd, bwd).max(axis=-1).astype(np.float32)


def _base_scores(
    chems: NDArray[np.float32],
    n: int,
    *,
    through_kind: str,
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
            elif through_kind == "optionh":
                scores[..., int(shape)] = _through_score_optionh(gradients, a, b)
            else:
                msg = f"unknown through_kind={through_kind}"
                raise ValueError(msg)
        else:
            (a,) = port_indices
            scores[..., int(shape)] = _endpoint_score(gradients, a)
    return scores


def _score_with_weight(
    chems: NDArray[np.float32],
    puzzle: Puzzle,
    allowed: NDArray[np.bool_],
    *,
    through_kind: str,
    weight: float,
) -> NDArray[np.float32]:
    """Replay score_shapes with a configurable consensus weight."""
    n = puzzle.size
    base_raw = _base_scores(chems, n, through_kind=through_kind)
    base = np.where(allowed, base_raw, np.float32(-np.inf)).astype(np.float32)
    augmented = base
    for _ in range(CONSENSUS_ITERATIONS):
        bonus = _consensus_bonus(augmented, n)
        augmented = np.where(
            allowed,
            base + np.float32(weight) * bonus,
            np.float32(-np.inf),
        ).astype(np.float32)
    return augmented


def _argmax_correct(
    scores: NDArray[np.float32],
    allowed: NDArray[np.bool_],
) -> tuple[int, dict[tuple[int, int], tuple[Shape, Shape]]]:
    correct = 0
    mismatches: dict[tuple[int, int], tuple[Shape, Shape]] = {}
    for (r, c), expected in _HAMILTONIAN.items():
        row_scores = scores[r, c]
        row_allowed = allowed[r, c]
        if not bool(row_allowed.any()):
            mismatches[(r, c)] = (expected, Shape(0))
            continue
        masked = np.where(row_allowed, row_scores, np.float32(-np.inf))
        top = int(np.argmax(masked))
        if Shape(top) == expected:
            correct += 1
        else:
            mismatches[(r, c)] = (expected, Shape(top))
    return correct, mismatches


def _baseline_substrate(
    puzzle: Puzzle,
) -> NDArray[np.float32]:
    chems = init_chems(puzzle)
    sources = build_sources(puzzle)
    mutual_open = build_open_mutual(puzzle)
    chems, tick, delta = _quiesce(chems, mutual_open, sources)
    print(f"    baseline K-1 substrate: quiesced at tick {tick} (delta={delta:.2e})")
    return chems


def _optionh_substrate(
    puzzle: Puzzle,
) -> NDArray[np.float32]:
    n = puzzle.size
    k_sources = len(puzzle.waypoints)
    chems = np.zeros((n, n, k_sources), dtype=np.float32)
    sources = np.array(
        [(wp.row, wp.col, wp.number - 1) for wp in puzzle.waypoints],
        dtype=np.int64,
    )
    mutual_open = build_open_mutual(puzzle)
    chems, tick, delta = _quiesce(chems, mutual_open, sources)
    print(f"    Option H K-source substrate: quiesced at tick {tick} (delta={delta:.2e})")
    return chems


def main() -> int:
    puzzle = load_puzzle("puzzles/tiny_3x3.json")
    allowed = build_allowed_mask(puzzle)

    print("Building steady-state substrates...")
    baseline_chems = _baseline_substrate(puzzle)
    optionh_chems = _optionh_substrate(puzzle)

    results: dict[tuple[str, float], int] = {}
    detail: dict[tuple[str, float], dict[tuple[int, int], tuple[Shape, Shape]]] = {}

    print("\n" + "=" * 72)
    print(f"{'substrate':<12} {'weight':>8}  {'argmax-correct':>16}  mismatches")
    print("=" * 72)
    for substrate_name, chems, through_kind in (
        ("baseline", baseline_chems, "baseline"),
        ("optionH", optionh_chems, "optionh"),
    ):
        for weight in _WEIGHTS:
            scores = _score_with_weight(
                chems, puzzle, allowed,
                through_kind=through_kind, weight=weight,
            )
            correct, mismatches = _argmax_correct(scores, allowed)
            results[(substrate_name, weight)] = correct
            detail[(substrate_name, weight)] = mismatches
            mm_str = ", ".join(
                f"({r},{c}):{got.name}(want {exp.name})"
                for (r, c), (exp, got) in mismatches.items()
            )
            print(f"{substrate_name:<12} {weight:>8.2f}  {correct:>5d} / 9          {mm_str}")

    print("\n" + "=" * 72)
    print("PHASE 8A GATE")
    print("=" * 72)

    peak_combo = max(results, key=results.get)
    peak_count = results[peak_combo]
    print(f"Peak: {peak_combo[0]} @ weight={peak_combo[1]} → {peak_count}/9 argmax-correct")

    if peak_count >= 9:
        print("\nGATE PASS: 9/9 achieved.")
        print(f"Commit substrate={peak_combo[0]}, CONSENSUS_WEIGHT={peak_combo[1]}.")
        return 0

    print("\nGATE FAIL: no (substrate, weight) combination reached 9/9.")
    print("Proceeding to Phase 8B (symmetry-breaker evaluation).")
    print("\nPeak-per-substrate:")
    for substrate_name in ("baseline", "optionH"):
        best_weight = max(
            _WEIGHTS, key=lambda w: results[(substrate_name, w)],
        )
        print(
            f"  {substrate_name}: best weight={best_weight}, "
            f"correct={results[(substrate_name, best_weight)]}/9",
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
