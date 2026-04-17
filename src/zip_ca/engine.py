"""Layer-1 ILCA state container and orchestrated tick.

Implements ``docs/design.md`` §11.2 - the single mutable long-lived
object in the solver - and Algorithm 1 §8 steps 1-5. Phase 5 shipped
steps 1-3 and 5; Phase 6 closes the loop with step 4 (noise).

The tick is the sole point of mutation. Every helper it calls
(``diffuse_tick``, ``score_shapes``, ``_apply_reward``,
``build_mutual_open``) is a pure function returning a fresh array;
``tick`` is the only place the engine's fields are reassigned. This
concentrates the state-lifecycle reasoning into one auditable call
site while preserving the Phase-4 immutability discipline at the
leaves.

Warm-up invariant (§12, row T_warm): for ``tick_count < T_WARM`` the
Layer-1 probability tensor and argmax shape array are untouched —
only ``chems`` evolves. This gives the diffusion field time to
carry waypoint identity into every reachable cell before the
probability update has any gradient to work with; starting the
update from an all-zero field is a guaranteed no-op.

Phase 6 adds Algorithm 1 step 4: per-cell stochastic noise with an
exponentially annealed ``η(t) = η₀·exp(-t/τ)`` schedule. The noise
call sits between the multiplicative reward and the argmax, so the
reward signal still biases the mix but noise jitters the winner out
of score-magnitude-collapsed tiebreaks (the exact failure mode
Phase 5 documented on ``tiny_3x3``).
"""

import math
from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .actions import build_allowed_mask
from .diffusion import (
    build_mutual_open,
    build_open_mutual,
    build_sources,
    diffuse_tick,
    init_chems,
)
from .noise import inject_noise
from .puzzle import Puzzle
from .scoring import score_shapes
from .shapes import NUM_SHAPES

T_WARM: Final[int] = 20

# §4.4 Phase 7 reward hyperparameters — softmax-over-score-differences
# convex-blended into probs. Replaces the Phase 5 sign-band
# multiplicative update, which was a fixed point whenever a cell's
# allowed-shape scores all fell in one sign band (the 0/20 tiny_3x3
# root cause surfaced in Phase 6). See
# docs/plans/2026-04-17-006-feat-phase-7-softmax-reward-redesign-plan.md
#
# TEMPERATURE controls softmax sharpness: a score gap of TEMPERATURE
# produces an e-fold probability ratio pre-blend. ALPHA is the convex
# blend weight per tick; small enough that a single tick cannot
# collapse the distribution, large enough that ALPHA * T_MAX >> 1 so
# evidence accumulates within the budget.
TEMPERATURE: Final[float] = 0.5
ALPHA: Final[float] = 0.15

# §12 Phase 6 hyperparameters — exponentially annealed per-cell noise.
# ETA_0 / BETA raised from the plan's nominal (0.10 / 0.20) per R2
# mitigation after tiny_3x3 refused to diverge across run_ids: the
# chemistry field is settled post-warm-up, so the reward reasserts the
# same per-cell winner every tick and weak noise cannot flip argmax
# within the T_STABLE window. Raising both gives each perturbed cell a
# large enough convex kick to beat the next tick's reward bounce.
ETA_0: Final[float] = 0.15
TAU: Final[float] = 200.0
BETA: Final[float] = 0.30

# R1 mitigation: gate quiescence on annealed noise. The solver refuses
# to declare quiescence until eta(t) has decayed below this threshold,
# which on the nominal schedule is tick ~t = TAU * ln(ETA_0 / ETA_QUIESCENCE_GATE)
# ≈ 542. Without this gate, quiescence fires at tick ~T_WARM + T_STABLE
# while noise is still strong enough to matter, reporting "converged"
# on a trajectory that is in fact still exploring in the prob space
# (argmax is sticky because the reward amplifies the current winner
# every tick).
ETA_QUIESCENCE_GATE: Final[float] = 0.01


def eta_schedule(tick_count: int) -> float:
    """Return ``η(t) = η₀·exp(-t/τ)`` for the given tick count.

    The schedule is a monotone function of ``tick_count`` (not
    ``tick_count - T_WARM``) so the warm-up period counts toward the
    exploration budget even though the noise injection is gated by
    ``T_WARM``. Design §7 is ambiguous on this choice; this reading
    keeps the schedule a single pure function of the engine's own
    tick counter and simplifies reasoning about determinism.

    Exposed (no leading underscore) so the solver can gate quiescence
    on ``eta_schedule(tick) < ETA_QUIESCENCE_GATE`` without importing
    the internal constants and recomputing the formula.
    """
    return ETA_0 * math.exp(-tick_count / TAU)


@dataclass(slots=True)
class EngineState:
    """Mutable orchestration state for a single solver run.

    Attributes:
        puzzle: The puzzle being solved; never mutated.
        run_id: Restart generation (0-based). Phase 5 runs only
            ``run_id = 0``; Phase 6 will re-seed on quiescent
            failure by incrementing this.
        tick_count: Number of completed calls to :meth:`tick`.
            Named ``tick_count`` rather than the design's ``tick``
            so the method ``tick()`` does not shadow the field.
        probs: ``(N, N, 10)`` float64 probability tensor summing to
            1 along the shape axis at every cell.
        shapes: ``(N, N)`` int8 array of argmax shape indices in
            ``0..9``. Never ``-1`` post-:meth:`fresh` — every cell
            has at least one allowed shape by the
            :func:`build_allowed_mask` invariant.
        chems: ``(N, N, K-1)`` float32 concentration tensor.
        allowed: ``(N, N, 10)`` bool mask from
            :func:`build_allowed_mask`; invariant post-:meth:`fresh`.
        _mutual: ``(N, N, 4)`` bool mask gating diffusion flow for
            the *current* shape grid. Must be refreshed whenever
            ``shapes`` changes.
        _warm_mutual: ``(N, N, 4)`` bool mask gating flow by walls
            and grid bounds only. Used during ``tick_count <
            T_WARM`` so the field spreads into every reachable cell
            instead of being trapped by the argmax-tiebreak initial
            shape topology (design §5.4).
        _sources: ``(M, 3)`` int64 Dirichlet source spec; invariant
            post-:meth:`fresh`.
    """

    puzzle: Puzzle
    run_id: int
    tick_count: int
    probs: NDArray[np.float64]
    shapes: NDArray[np.int8]
    chems: NDArray[np.float32]
    allowed: NDArray[np.bool_]
    _mutual: NDArray[np.bool_] = field(repr=False)
    _warm_mutual: NDArray[np.bool_] = field(repr=False)
    _sources: NDArray[np.int64] = field(repr=False)

    @classmethod
    def fresh(cls, puzzle: Puzzle, run_id: int = 0) -> "EngineState":
        """Construct a fresh engine state for ``puzzle``.

        Allocates all arrays without aliasing:

        1. ``allowed`` via :func:`build_allowed_mask` (Phase 3).
        2. ``probs`` uniform over the allowed shapes at each cell.
        3. ``shapes`` = argmax of ``probs``; tie-breaks to the
           lowest shape index by :func:`numpy.argmax`'s semantics,
           giving byte-for-byte deterministic seeding.
        4. ``chems`` all-zero via :func:`init_chems` (Phase 4).
        5. ``_mutual`` and ``_sources`` precomputed from the initial
           shape grid and waypoint set.

        Args:
            puzzle: The puzzle to seed the engine with.
            run_id: Restart generation; defaults to 0.

        Returns:
            A fully initialised :class:`EngineState`.

        Raises:
            PuzzleValidationError: Propagated from
                :func:`build_allowed_mask` if the puzzle has a cell
                with zero admissible shapes, or from
                :func:`init_chems` if ``K < 2``.
        """
        allowed = build_allowed_mask(puzzle)
        counts = allowed.sum(axis=-1, keepdims=True).astype(np.float64)
        probs = np.where(allowed, 1.0, 0.0).astype(np.float64) / counts
        shapes = probs.argmax(axis=-1).astype(np.int8)
        chems = init_chems(puzzle)
        mutual = build_mutual_open(shapes, puzzle)
        warm_mutual = build_open_mutual(puzzle)
        sources = build_sources(puzzle)
        return cls(
            puzzle=puzzle,
            run_id=run_id,
            tick_count=0,
            probs=probs,
            shapes=shapes,
            chems=chems,
            allowed=allowed,
            _mutual=mutual,
            _warm_mutual=warm_mutual,
            _sources=sources,
        )

    def tick(self) -> None:
        """Advance the engine by one tick of Algorithm 1 (steps 1-5).

        Order:

        1. ``diffuse_tick`` - update ``chems`` (sources re-asserted
           inside).
        2-3. If ``tick_count >= T_WARM``: score - multiplicative
           reward - renormalise ``probs``.
        4. If ``tick_count >= T_WARM``: per-cell noise injection via
           :func:`zip_ca.inject_noise` with the annealed ``eta(t)``
           schedule.
        5. If ``tick_count >= T_WARM``: re-select ``shapes`` via
           argmax and rebuild ``_mutual`` for next tick's diffusion.
        """
        # During warm-up, use the wall-gated fully-open mutual so the
        # field spreads symmetrically (design §5.4); switch to the
        # shape-gated mutual once Layer-1 dynamics engage.
        flow_mutual = self._warm_mutual if self.tick_count < T_WARM else self._mutual
        self.chems = diffuse_tick(self.chems, flow_mutual, self._sources)

        if self.tick_count >= T_WARM:
            scores = score_shapes(self.chems, self.puzzle, self.allowed)
            self.probs = _apply_reward(self.probs, scores, self.allowed)
            self.probs = inject_noise(
                self.probs,
                self.allowed,
                run_id=self.run_id,
                tick=self.tick_count,
                eta=eta_schedule(self.tick_count),
                beta=BETA,
            )
            self.shapes = self.probs.argmax(axis=-1).astype(np.int8)
            self._mutual = build_mutual_open(self.shapes, self.puzzle)

        self.tick_count += 1


def _apply_reward(
    probs: NDArray[np.float64],
    scores: NDArray[np.float32],
    allowed: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Apply the Phase 7 softmax-blend reward update.

    For each cell, compute the softmax distribution over allowed-shape
    scores with temperature :data:`TEMPERATURE`, then convex-blend it
    into the current ``probs`` with weight :data:`ALPHA`::

        softmax = exp((scores - row_max) / TEMPERATURE) / sum(exp(...))
        probs  <- (1 - ALPHA) * probs + ALPHA * softmax

    Replaces the §4.4 sign-band multiplicative update that was a
    structural fixed point whenever all allowed scores fell in one
    sign band (the Phase 6 project-defining finding). Any non-zero
    score gap now produces a non-uniform update regardless of sign,
    so argmax tracks accumulated chemical evidence rather than
    numpy's lowest-index tiebreak.

    Numerical stability comes from subtracting the per-row max-over-
    allowed before exponentiating, standard softmax practice. Disallowed
    slots keep their ``-np.inf`` score sentinel → ``exp(-inf/T) = 0`` →
    softmax mass is 0 there → convex blend with ``probs == 0`` keeps
    them at 0 exactly.

    Args:
        probs: Current ``(N, N, 10)`` float64 probability tensor;
            rows sum to 1 and disallowed slots are 0.
        scores: ``(N, N, 10)`` float32 score tensor with ``-np.inf``
            in disallowed slots.
        allowed: ``(N, N, 10)`` bool mask used to compute the
            per-row max-over-allowed for numerical stability.

    Returns:
        A new ``(N, N, 10)`` float64 tensor summing to 1 along the
        last axis at every cell; disallowed slots remain 0.
    """
    if probs.shape[-1] != NUM_SHAPES:
        msg = f"probs last axis must be {NUM_SHAPES} (got {probs.shape[-1]})"
        raise ValueError(msg)

    scores64 = scores.astype(np.float64)
    row_max = np.where(allowed, scores64, -np.inf).max(axis=-1, keepdims=True)
    shifted = np.where(allowed, (scores64 - row_max) / TEMPERATURE, -np.inf)
    exps = np.exp(shifted)
    softmax = exps / exps.sum(axis=-1, keepdims=True)

    blended = (1.0 - ALPHA) * probs + ALPHA * softmax
    return blended / blended.sum(axis=-1, keepdims=True)
