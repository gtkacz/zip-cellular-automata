"""Fixed-point-elimination regression for the Phase 7 softmax reward.

The Phase 6 acceptance failure (0/20 on ``tiny_3x3``) root-caused to
§4.4's sign-band thresholded multiplicative update being a no-op
whenever all allowed-shape scores of a cell fall in the same sign band.
Both reward and penalty factors then apply uniformly across the row
and renormalisation restores the prior exactly — a fixed point.

Phase 7's plan is to replace that rule with a softmax-over-score-
differences convex-blended into probs. This file locks in the three
invariants that redesign must honour:

1. *Any* non-zero score gap between two allowed shapes must move probs
   away from uniform, even when both scores are positive (the exact
   case the sign-band rule mishandled).
2. Exactly-equal allowed scores must preserve a uniform distribution
   (no phantom bias; tiebreak is the argmax's problem, not the
   reward's).
3. Disallowed slots must stay at 0; rows must sum to 1.

The sign-band implementation fails (1); the softmax-blend
implementation passes all three. This test exists to hold the line.
"""

import numpy as np
from numpy.typing import NDArray

from zip_ca.engine import _apply_reward
from zip_ca.shapes import NUM_SHAPES


def _uniform_row(allowed_row: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Return a uniform-over-allowed probability row."""
    count = float(allowed_row.sum())
    return np.where(allowed_row, 1.0 / count, 0.0).astype(np.float64)


def _single_cell(
    scores_row: list[float],
    allowed_row: list[bool],
) -> tuple[NDArray[np.float64], NDArray[np.float32], NDArray[np.bool_]]:
    """Wrap a single (scores, allowed) row as 1x1x10 tensors."""
    if len(scores_row) != NUM_SHAPES or len(allowed_row) != NUM_SHAPES:
        msg = f"rows must have length {NUM_SHAPES}"
        raise ValueError(msg)
    allowed = np.array(allowed_row, dtype=np.bool_).reshape(1, 1, NUM_SHAPES)
    scores = np.array(scores_row, dtype=np.float32).reshape(1, 1, NUM_SHAPES)
    probs = _uniform_row(allowed[0, 0]).reshape(1, 1, NUM_SHAPES)
    return probs, scores, allowed


def test_all_positive_equal_scores_preserve_uniform() -> None:
    """All allowed scores equal -> probs stay uniform over allowed.

    This is the honest-uniform case: the reward has no evidence to
    separate shapes, so it must not invent any.
    """
    equal = 0.5
    probs, scores, allowed = _single_cell(
        scores_row=[
            equal,
            equal,
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            equal,
            -np.inf,
            -np.inf,
            -np.inf,
        ],
        allowed_row=[True, True, False, False, False, False, True, False, False, False],
    )
    out = _apply_reward(probs, scores, allowed)

    expected = _uniform_row(allowed[0, 0])
    assert np.allclose(out[0, 0], expected, atol=1e-12), (
        f"uniform-scores row should remain uniform, got {out[0, 0]}"
    )


def test_all_positive_unequal_scores_break_uniform() -> None:
    """All scores positive but unequal -> probs must become non-uniform.

    This is the test that the sign-band §4.4 rule *fails* on: every
    allowed score exceeds ``THETA_PLUS``, so every slot multiplies by
    ``R`` and renormalisation restores the prior. The softmax-blend
    rule must separate them monotonically in score.
    """
    probs, scores, allowed = _single_cell(
        scores_row=[0.2, 0.5, -np.inf, -np.inf, -np.inf, -np.inf, 0.8, -np.inf, -np.inf, -np.inf],
        allowed_row=[True, True, False, False, False, False, True, False, False, False],
    )
    out = _apply_reward(probs, scores, allowed)

    # Same sign, same magnitude band - sign-band update leaves probs uniform.
    # Softmax-blend update must produce strictly non-uniform probs.
    uniform = _uniform_row(allowed[0, 0])
    assert not np.allclose(out[0, 0], uniform, atol=1e-6), (
        f"unequal-scores row must leave uniform, got {out[0, 0]} (identical to uniform {uniform})"
    )

    # Monotonicity: shape 6 (highest score) must outrank shape 1, which must
    # outrank shape 0. Disallowed slots stay at 0.
    assert out[0, 0, 6] > out[0, 0, 1] > out[0, 0, 0] > 0.0, (
        f"probs must be strictly monotone in score, got {out[0, 0]}"
    )


def test_mixed_sign_scores_break_uniform() -> None:
    """Scores straddling zero must also produce non-uniform probs."""
    probs, scores, allowed = _single_cell(
        scores_row=[-0.3, 0.0, -np.inf, -np.inf, -np.inf, -np.inf, 0.4, -np.inf, -np.inf, -np.inf],
        allowed_row=[True, True, False, False, False, False, True, False, False, False],
    )
    out = _apply_reward(probs, scores, allowed)

    assert out[0, 0, 6] > out[0, 0, 1] > out[0, 0, 0], (
        f"mixed-sign row must be monotone in score, got {out[0, 0]}"
    )


def test_disallowed_slots_stay_zero() -> None:
    """Disallowed shape slots must remain exactly 0 post-update."""
    probs, scores, allowed = _single_cell(
        scores_row=[0.1, 0.5, -np.inf, -np.inf, -np.inf, -np.inf, 0.9, -np.inf, -np.inf, -np.inf],
        allowed_row=[True, True, False, False, False, False, True, False, False, False],
    )
    out = _apply_reward(probs, scores, allowed)

    disallowed = ~allowed[0, 0]
    assert np.all(out[0, 0, disallowed] == 0.0), (
        f"disallowed slots must be 0, got {out[0, 0, disallowed]}"
    )


def test_row_sums_to_one() -> None:
    """Every cell row must sum to 1 after the update."""
    probs, scores, allowed = _single_cell(
        scores_row=[0.1, 0.5, -np.inf, -np.inf, -np.inf, -np.inf, 0.9, -np.inf, -np.inf, -np.inf],
        allowed_row=[True, True, False, False, False, False, True, False, False, False],
    )
    out = _apply_reward(probs, scores, allowed)

    assert np.isclose(out[0, 0].sum(), 1.0, atol=1e-12), f"row must sum to 1, got {out[0, 0].sum()}"


def test_all_negative_unequal_scores_break_uniform() -> None:
    """All scores negative but unequal -> probs must still separate.

    Symmetric to the all-positive case: sign-band multiplies all by
    ``P`` and leaves the row uniform; softmax must separate them.
    """
    probs, scores, allowed = _single_cell(
        scores_row=[
            -0.8,
            -0.3,
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            -0.1,
            -np.inf,
            -np.inf,
            -np.inf,
        ],
        allowed_row=[True, True, False, False, False, False, True, False, False, False],
    )
    out = _apply_reward(probs, scores, allowed)

    # Shape 6 has the highest (least-negative) score -> highest prob.
    assert out[0, 0, 6] > out[0, 0, 1] > out[0, 0, 0], (
        f"all-negative row must be monotone in score, got {out[0, 0]}"
    )
