"""Phase 8B gate: evaluate three score-level symmetry-breakers on tiny_3x3.

Phase 8A showed that CONSENSUS_WEIGHT tuning cannot lift either
substrate above 5/9 argmax-correct. Phase 8B tests whether a
score-level symmetry-breaker can reach the 8/9 argmax threshold
(with solve-prefix >= 7/9 as the second gate if argmax passes).

Three candidates:

* **Noise** — add static per-(cell, shape) noise drawn once from a
  fixed seed, amplitude sigma in {1e-3, 1e-2, 1e-1}. Control
  condition: undirected symmetry-breaker.
* **Stagger** — clamp each waypoint source at ``1 + eps * (k - 1)``
  instead of uniformly at 1.0, amplitude eps in {1e-3, 1e-2, 1e-1}.
  Tests whether breaking W_1/W_K amplitude symmetry at the substrate
  level propagates into score asymmetry that resolves the three
  canonical mismatch cells on tiny_3x3.
* **Heading** — add a per-(cell, shape) term
  ``beta * <shape_port_mask, heading_to_next_waypoint>`` at
  endpoint-waypoint cells only. Targets the (0,0) END_S-over-END_E
  and (2,2) END_N-over-END_W failures directly; does NOT modify
  interior through-cell scores. Amplitude beta in {1e-3, 1e-2, 1e-1}.

Measurement: per-cell argmax-correct count on tiny_3x3 at steady
state, evaluated at the production CONSENSUS_WEIGHT=10.0.
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
_PORT_DELTAS: Final[dict[str, tuple[int, int]]] = {
    "N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1),
}
_QUIESCENCE_TOL: Final[float] = 1e-6
_MAX_WARMUP_TICKS: Final[int] = 5000
_AMPLITUDES: Final[tuple[float, ...]] = (1e-3, 1e-2, 1e-1)
_CONSENSUS_WEIGHT: Final[float] = 10.0
_SEED: Final[int] = 20260417
_HAMILTONIAN: Final[dict[tuple[int, int], Shape]] = {
    (0, 0): Shape.END_E, (0, 1): Shape.H,     (0, 2): Shape.SW,
    (1, 0): Shape.SE,    (1, 1): Shape.H,     (1, 2): Shape.NW,
    (2, 0): Shape.NE,    (2, 1): Shape.H,     (2, 2): Shape.END_W,
}


def _quiesce(
    chems: NDArray[np.float32],
    mutual_open: NDArray[np.bool_],
    sources: NDArray[np.int64],
) -> NDArray[np.float32]:
    delta = float("inf")
    for _ in range(_MAX_WARMUP_TICKS):
        new_chems = diffuse_tick(chems, mutual_open, sources)
        delta = float(np.max(np.abs(new_chems - chems)))
        chems = new_chems
        if delta < _QUIESCENCE_TOL:
            break
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
    base_raw: NDArray[np.float32],
    allowed: NDArray[np.bool_],
    n: int,
    weight: float,
) -> NDArray[np.float32]:
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
    scores: NDArray[np.float32], allowed: NDArray[np.bool_],
) -> tuple[int, list[tuple[tuple[int, int], Shape, Shape]]]:
    correct = 0
    mismatches: list[tuple[tuple[int, int], Shape, Shape]] = []
    for (r, c), expected in _HAMILTONIAN.items():
        row_scores = scores[r, c]
        row_allowed = allowed[r, c]
        masked = np.where(row_allowed, row_scores, np.float32(-np.inf))
        top = int(np.argmax(masked))
        if Shape(top) == expected:
            correct += 1
        else:
            mismatches.append(((r, c), expected, Shape(top)))
    return correct, mismatches


def _heading_vectors(puzzle: Puzzle) -> dict[tuple[int, int], tuple[float, float]]:
    """For each waypoint cell, unit heading toward the next waypoint in sequence.

    At W_1 it points toward W_2; at W_K it points toward W_{K-1}
    (reversed, representing "where the path came from" — the outbound
    port at the terminal waypoint should align with this reverse
    heading).
    """
    headings: dict[tuple[int, int], tuple[float, float]] = {}
    wps = sorted(puzzle.waypoints, key=lambda wp: wp.number)
    for idx, wp in enumerate(wps):
        neighbour_idx = idx + 1 if idx == 0 else idx - 1
        neighbour = wps[neighbour_idx]
        dr = float(neighbour.row - wp.row)
        dc = float(neighbour.col - wp.col)
        mag = (dr * dr + dc * dc) ** 0.5 or 1.0
        headings[(wp.row, wp.col)] = (dr / mag, dc / mag)
    return headings


def _heading_bonus(
    puzzle: Puzzle, n: int, allowed: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """Per-(cell, shape) heading bonus; non-waypoint cells get zero."""
    bonus = np.zeros((n, n, NUM_SHAPES), dtype=np.float32)
    headings = _heading_vectors(puzzle)
    for (r, c), (hr, hc) in headings.items():
        for shape in Shape:
            if not bool(allowed[r, c, int(shape)]):
                continue
            total = 0.0
            for port in open_ports(shape):
                dr, dc = _PORT_DELTAS[port.name]
                total += float(dr) * hr + float(dc) * hc
            bonus[r, c, int(shape)] = total
    return bonus


def _noise_bonus(n: int, rng_seed: int) -> NDArray[np.float32]:
    rng = np.random.default_rng(rng_seed)
    return rng.normal(0.0, 1.0, size=(n, n, NUM_SHAPES)).astype(np.float32)


def _baseline_substrate(puzzle: Puzzle, eps: float = 0.0) -> NDArray[np.float32]:
    """K-1 segment chemistry with optional source-amplitude stagger.

    Baseline's build_sources clamps both waypoints at the same
    segment. We approximate stagger here by post-rescaling the
    steady-state field — not exactly equivalent to clamped diffusion
    with two different amplitudes, but at K=2 and a single segment
    the two sources contribute additively to the profile, so scaling
    the field by (1 + eps) approximates raising W_2's clamp. A
    faithful stagger for baseline would need to override
    build_sources; punt that to production if this candidate wins.
    """
    chems = init_chems(puzzle)
    sources = build_sources(puzzle)
    mutual_open = build_open_mutual(puzzle)
    chems = _quiesce(chems, mutual_open, sources)
    if eps:
        chems = chems * np.float32(1.0 + eps)
    return chems


def _optionh_substrate(puzzle: Puzzle, eps: float = 0.0) -> NDArray[np.float32]:
    """K source-only fields with optional stagger at W_k amplitudes.

    Source k is clamped at ``1 + eps * k`` (k = 0 means W_1, so
    W_1 stays at 1.0; W_2 moves to 1 + eps).
    """
    n = puzzle.size
    k_sources = len(puzzle.waypoints)
    chems = np.zeros((n, n, k_sources), dtype=np.float32)
    wps = sorted(puzzle.waypoints, key=lambda wp: wp.number)
    sources = np.array(
        [(wp.row, wp.col, wp.number - 1) for wp in wps], dtype=np.int64,
    )
    mutual_open = build_open_mutual(puzzle)

    # With stagger, apply a post-diffusion scale per-source. This
    # approximates the steady state of a diffuse-with-Dirichlet
    # system where source k is clamped at (1 + eps * k): the Laplace
    # equation is linear in the boundary value, so rescaling the
    # field after a unit-clamp solve gives the same steady state as
    # running diffusion with the scaled boundary value.
    chems = _quiesce(chems, mutual_open, sources)
    if eps:
        for k in range(k_sources):
            chems[..., k] = chems[..., k] * np.float32(1.0 + eps * k)
    return chems


def _evaluate(
    label: str,
    through_kind: str,
    chems: NDArray[np.float32],
    puzzle: Puzzle,
    allowed: NDArray[np.bool_],
    extra_bonus: NDArray[np.float32] | None = None,
) -> tuple[int, list[tuple[tuple[int, int], Shape, Shape]]]:
    n = puzzle.size
    base_raw = _base_scores(chems, n, through_kind=through_kind)
    if extra_bonus is not None:
        base_raw = base_raw + extra_bonus
    scores = _apply_consensus(base_raw, allowed, n, _CONSENSUS_WEIGHT)
    correct, mismatches = _argmax_correct(scores, allowed)
    print(f"  {label:<50s}  {correct}/9  "
          + ", ".join(f"({r},{c}):{g.name}(want {e.name})"
                      for (r, c), e, g in mismatches))
    return correct, mismatches


def main() -> int:
    puzzle = load_puzzle("puzzles/tiny_3x3.json")
    n = puzzle.size
    allowed = build_allowed_mask(puzzle)

    print("=" * 82)
    print("PHASE 8B GATE: symmetry-breaker sweep on tiny_3x3")
    print("=" * 82)

    # Baseline reference (unmodified; matches 8A baseline@10)
    print("\n-- Reference (no symmetry breaker, CONSENSUS_WEIGHT=10.0) --")
    baseline_chems = _baseline_substrate(puzzle, eps=0.0)
    optionh_chems = _optionh_substrate(puzzle, eps=0.0)
    _evaluate("baseline K-1", "baseline", baseline_chems, puzzle, allowed)
    _evaluate("Option H K-source", "optionh", optionh_chems, puzzle, allowed)

    results: dict[tuple[str, str, float], int] = {}

    # Candidate 1: static noise
    print("\n-- Candidate 1: static per-cell score noise --")
    for sigma in _AMPLITUDES:
        noise = _noise_bonus(n, _SEED) * np.float32(sigma)
        correct, _ = _evaluate(
            f"baseline + noise(sigma={sigma})", "baseline",
            baseline_chems, puzzle, allowed, extra_bonus=noise,
        )
        results[("noise", "baseline", sigma)] = correct
        correct, _ = _evaluate(
            f"Option H + noise(sigma={sigma})", "optionh",
            optionh_chems, puzzle, allowed, extra_bonus=noise,
        )
        results[("noise", "optionh", sigma)] = correct

    # Candidate 2: source-amplitude stagger
    print("\n-- Candidate 2: staggered source amplitudes --")
    for eps in _AMPLITUDES:
        correct, _ = _evaluate(
            f"baseline-stagger(eps={eps})", "baseline",
            _baseline_substrate(puzzle, eps=eps), puzzle, allowed,
        )
        results[("stagger", "baseline", eps)] = correct
        correct, _ = _evaluate(
            f"Option H-stagger(eps={eps})", "optionh",
            _optionh_substrate(puzzle, eps=eps), puzzle, allowed,
        )
        results[("stagger", "optionh", eps)] = correct

    # Candidate 3: waypoint-only heading prior
    print("\n-- Candidate 3: waypoint heading prior (endpoint cells only) --")
    heading = _heading_bonus(puzzle, n, allowed)
    for beta in _AMPLITUDES:
        bonus = heading * np.float32(beta)
        correct, _ = _evaluate(
            f"baseline + heading(beta={beta})", "baseline",
            baseline_chems, puzzle, allowed, extra_bonus=bonus,
        )
        results[("heading", "baseline", beta)] = correct
        correct, _ = _evaluate(
            f"Option H + heading(beta={beta})", "optionh",
            optionh_chems, puzzle, allowed, extra_bonus=bonus,
        )
        results[("heading", "optionh", beta)] = correct

    # Candidate 4: heading + stagger combined on Option H (the plan's
    # strongest combination, since heading targets endpoints and
    # stagger targets interior)
    print("\n-- Candidate 4: heading + stagger (Option H) --")
    for beta in _AMPLITUDES:
        for eps in _AMPLITUDES:
            bonus = heading * np.float32(beta)
            correct, _ = _evaluate(
                f"Option H-stagger(eps={eps}) + heading(beta={beta})", "optionh",
                _optionh_substrate(puzzle, eps=eps), puzzle, allowed, extra_bonus=bonus,
            )
            results[("combined", f"optionh-eps={eps}", beta)] = correct

    print("\n" + "=" * 82)
    print("PHASE 8B GATE OUTCOME")
    print("=" * 82)
    peak = max(results.items(), key=lambda kv: kv[1])
    print(f"Peak: {peak[0]} -> {peak[1]}/9 argmax-correct")
    if peak[1] >= 8:
        print("\nARGMAX GATE PASS: >=8/9 achieved. Proceeding to solve() validation.")
        return 0
    print("\nARGMAX GATE FAIL: no candidate reached 8/9. Proceeding to Phase 8C.")
    print("\nAll-candidate peaks:")
    by_kind: dict[str, int] = {}
    for (kind, *_), count in results.items():
        by_kind[kind] = max(by_kind.get(kind, 0), count)
    for kind, count in by_kind.items():
        print(f"  {kind:<12s} peak: {count}/9")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
