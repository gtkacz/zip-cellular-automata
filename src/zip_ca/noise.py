"""Per-cell stochastic noise for Layer-1 probability perturbation.

Implements ``docs/design.md`` §7 (Algorithm 1 step 4). Pure function;
no side effects. The per-cell PRNG is a splitmix64 finaliser whose
output is stable across Python versions, NumPy versions, and operating
systems, which Python's built-in :func:`hash` is not (PEP 456 hash
randomisation applies to str/bytes, and tuple hashing is not
contractually stable across minor releases either).

The noise is a convex β-mix between the current probability row and a
*per-cell random distribution over allowed shapes*, applied only on
cells whose per-cell gate draw falls below ``eta`` this tick. §7's
Algorithm 1 step 4 calls for "a distribution"; the design neither
prescribes nor forbids uniform. Empirically, mixing with
uniform-over-allowed is a no-op on cells whose current probs already
equal uniform (e.g. two-allowed-shape cells where the §4.4 reward
gives both shapes the same multiplicative factor because both scores
fall in the same sign band). Those cells then stay pinned at the
deterministic argmax tiebreak (lowest shape index) *across all
restart run_ids*, because uniform-over-allowed does not depend on
``run_id``.

The implemented fix draws an independent per-(i, j, shape) uniform
from a splitmix64 hash over ``(i, j, k_shape, run_id, tick)``,
zeros out disallowed shapes, and renormalises to a valid probability
row. This is a Dirichlet(1, ..., 1) sample over the allowed-shape
support; the mix ``(1 - β)·probs + β·D_{ij}(run_id, tick)`` is a
convex combination of two probability vectors and remains a valid
probability row. Crucially, ``D_{ij}`` depends on ``run_id``, so
distinct restarts can flip tied cells to different shapes.

The pure-CA property (each cell reads only its own state + per-cell
randomness) is preserved because every cell derives its own gate and
shape-distribution from its own coordinates.
"""

from typing import Final

import numpy as np
from numpy.typing import NDArray

# splitmix64 finaliser constants (Vigna 2014, public-domain reference
# implementation). Chosen because their output is bit-stable forever —
# pure integer arithmetic; no dependence on Python's hash(), NumPy's
# Generator stream stability, or SeedSequence spawning.
_M1: Final[np.uint64] = np.uint64(0x9E3779B97F4A7C15)
_M2: Final[np.uint64] = np.uint64(0xBF58476D1CE4E5B9)
_M3: Final[np.uint64] = np.uint64(0x94D049BB133111EB)

_U64_MASK: Final[int] = 0xFFFFFFFFFFFFFFFF

# Shift amounts are applied via np.uint64 operands so the resulting
# arithmetic stays inside NumPy's promoted dtype (uint64) and does not
# silently widen to Python int / object arrays.
_SHIFT_30: Final[np.uint64] = np.uint64(30)
_SHIFT_27: Final[np.uint64] = np.uint64(27)
_SHIFT_31: Final[np.uint64] = np.uint64(31)
_SHIFT_11: Final[np.uint64] = np.uint64(11)

# Top 53 bits of a uint64 fill an IEEE-754 double's mantissa exactly;
# dividing by 2**53 produces a uniform float in [0, 1) with the full
# available entropy and no bias from low-bit correlations.
_MANTISSA_SCALE: Final[float] = 1.0 / float(1 << 53)

# Independent splitmix64 constants for the shape-axis mix. Reusing _M1
# with a shape offset would create a diagonal correlation between the
# coordinate and shape hashes; a fourth constant keeps the two hash
# streams algebraically distinct.
_M4: Final[np.uint64] = np.uint64(0xD1B54A32D192ED03)


def _splitmix_finalise(h: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """Apply the splitmix64 finaliser in place-friendly form.

    Extracted so ``_per_cell_uniform`` and
    ``_per_cell_per_shape_uniform`` share the bit-mixing tail. All
    arithmetic is mod-2**64 by NumPy's uint64 semantics; callers must
    wrap the call site in ``np.errstate(over='ignore')`` because
    ``pytest`` promotes overflow warnings to errors.
    """
    h ^= h >> _SHIFT_30
    h *= _M2
    h ^= h >> _SHIFT_27
    h *= _M3
    h ^= h >> _SHIFT_31
    return h


def _per_cell_uniform(n: int, run_id: int, tick: int) -> NDArray[np.float64]:
    """Return an ``(N, N)`` float64 uniform draw in ``[0, 1)``.

    Every cell's draw is a splitmix64-finalised hash of its own
    ``(i, j, run_id, tick)`` tuple, giving per-cell independence
    within one tick and byte-equal reproducibility across calls with
    identical arguments.

    Args:
        n: Grid side length; ``N >= 2``.
        run_id: Restart generation for this run; mixed into the hash
            so different runs never collide.
        tick: Current tick count for this run.

    Returns:
        ``(N, N)`` :class:`numpy.float64` array of uniform draws.
    """
    # splitmix64 relies on mod-2^64 wrap-around in every multiply and
    # add; ``np.errstate(over='ignore')`` documents that overflow is
    # the intended arithmetic, not a bug. Without it, NumPy emits
    # RuntimeWarning which the pytest config (``filterwarnings =
    # ['error']``) promotes to a test failure.
    with np.errstate(over="ignore"):
        # Coordinate lift: ``+ 1`` prevents row-0 / col-0 coordinates
        # from annihilating the first multiplication term when run_id
        # and tick are both 0 at the origin cell.
        i, j = np.indices((n, n), dtype=np.uint64)
        h = (i + np.uint64(1)) * _M1
        h ^= (j + np.uint64(1)) * _M2
        h ^= np.uint64(run_id & _U64_MASK) * _M3
        h ^= np.uint64(tick & _U64_MASK) * _M1
        h = _splitmix_finalise(h)
        return (h >> _SHIFT_11).astype(np.float64) * _MANTISSA_SCALE


def _per_cell_per_shape_uniform(
    n: int,
    num_shapes: int,
    run_id: int,
    tick: int,
) -> NDArray[np.float64]:
    """Return an ``(N, N, num_shapes)`` uniform draw in ``[0, 1)``.

    Hashes ``(i, j, k_shape, run_id, tick)`` through splitmix64, giving
    per-(cell, shape) independence within one tick. This is what the
    convex-mix noise needs to break argmax ties on cells whose current
    probs equal uniform-over-allowed — the uniform distribution does
    not depend on ``run_id``, but this Dirichlet(1,...) draw does.
    """
    with np.errstate(over="ignore"):
        i, j, k = np.indices((n, n, num_shapes), dtype=np.uint64)
        h = (i + np.uint64(1)) * _M1
        h ^= (j + np.uint64(1)) * _M2
        h ^= (k + np.uint64(1)) * _M4
        h ^= np.uint64(run_id & _U64_MASK) * _M3
        h ^= np.uint64(tick & _U64_MASK) * _M1
        h = _splitmix_finalise(h)
        return (h >> _SHIFT_11).astype(np.float64) * _MANTISSA_SCALE


def inject_noise(
    probs: NDArray[np.float64],
    allowed: NDArray[np.bool_],
    *,
    run_id: int,
    tick: int,
    eta: float,
    beta: float,
) -> NDArray[np.float64]:
    """Inject per-cell stochastic noise into a probability tensor.

    Implements design §7 step 4. For each cell whose per-cell gate
    draw is below ``eta`` this tick, mix the row with an independent
    per-cell Dirichlet(1,...) draw over its allowed shapes using
    weight ``beta``. Input is not mutated; output is a fresh array
    whose rows still sum to 1 by convexity. See the module docstring
    for why the mix target is a per-cell random distribution rather
    than uniform-over-allowed.

    Args:
        probs: ``(N, N, 10)`` float64 probability tensor with every
            row summing to 1.
        allowed: ``(N, N, 10)`` bool allowed-action mask from
            :func:`zip_ca.build_allowed_mask`. Every cell must have
            at least one allowed shape.
        run_id: Restart generation for this run; 0 for the first
            run. Mixed into the per-cell PRNG.
        tick: Current tick count. Mixed into the per-cell PRNG; two
            distinct ticks within the same run produce independent
            draws.
        eta: Per-tick probability that any given cell is perturbed.
            Must lie in ``[0, 1]``. ``eta = 0.0`` fast-paths to a
            copy of ``probs``.
        beta: Convex-mix weight. Must lie in ``[0, 1]``. On a
            perturbed cell: ``probs_new = (1 - beta)·probs +
            beta·D_{ij}(run_id, tick)`` where ``D_{ij}`` is a fresh
            Dirichlet(1,...) draw over ``allowed[i, j]``.

    Returns:
        A fresh ``(N, N, 10)`` :class:`numpy.float64` tensor whose
        rows sum to 1.

    Raises:
        ValueError: If ``eta`` or ``beta`` lies outside ``[0, 1]``.
    """
    if not 0.0 <= eta <= 1.0:
        msg = f"eta must be in [0, 1] (got {eta})"
        raise ValueError(msg)
    if not 0.0 <= beta <= 1.0:
        msg = f"beta must be in [0, 1] (got {beta})"
        raise ValueError(msg)

    # Fast path: eta=0 is a common configuration for Phase-7
    # "noise-off" runs and hyperparameter sweeps at the boundary. A
    # defensive copy preserves the contract that callers may mutate
    # the returned array without touching the input.
    if eta == 0.0:
        return probs.astype(np.float64, copy=True)

    n, _, num_shapes = probs.shape
    raw = _per_cell_per_shape_uniform(n, num_shapes, run_id, tick)
    masked = np.where(allowed, raw, 0.0)
    row_sum = masked.sum(axis=-1, keepdims=True)
    # build_allowed_mask invariant guarantees each cell has at least
    # one allowed shape, so row_sum > 0 everywhere and the division is
    # safe. A splitmix64 draw of exactly 0.0 on every allowed shape of
    # a cell has probability < 2**(-53·count_allowed), which is below
    # the reproducibility test's 100-tick trial budget by many orders
    # of magnitude; we do not guard against it.
    random_dist = masked / row_sum

    r = _per_cell_uniform(n, run_id, tick)
    mask = (r < eta)[..., None]
    mixed = (1.0 - beta) * probs + beta * random_dist
    return np.where(mask, mixed, probs).astype(np.float64)
