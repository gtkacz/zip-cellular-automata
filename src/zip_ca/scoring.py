"""Per-shape reward scoring for the Layer-1 ILCA update.

Implements ``docs/design.md`` §6: for every cell and every candidate
shape, compute the alignment score ``sigma(s)`` used by the
multiplicative probability update in §4.4. The score is a pure
function of the current chemical field and the static allowed-shape
mask; it carries no internal state.

Indexing convention (binding):

* Gradient axis-2 order is ``_PORT_ORDER = (N, E, S, W)`` — the
  same ordering :mod:`zip_ca.diffusion` uses for
  :func:`~zip_ca.diffusion.build_mutual_open`'s axis-2. Any change
  to one must be mirrored in the other; the two modules share the
  same port-ordering convention by independent declaration because
  making it cross-module would entangle diffusion with scoring for
  no benefit.
* Segment axis is 0-indexed: design's segment ``k`` (1-indexed) is
  numpy index ``k - 1``. The K=2 branch sees a single-element
  segment axis and specialises the score accordingly.

The K=2 branch is load-bearing. Design §6.2 sums over *consecutive*
segment pairs ``(k, k+1)``; with ``K = 2`` no such pair exists.
Rather than letting the general formula silently return ``-inf``,
we substitute the same-segment score ``g_a + g_b`` — a discrete
divergence of the one chemical along the shape's two open ports.
For the convex U-shaped concentration profile along the correct
path, this sum is positive, which is exactly the reinforcement
signal Phase 5 needs. Phase 6 will evaluate whether generalising
§6.2 to "max over all segment pairs including ``k == k``" unifies
the two branches.

Consensus augmentation (Phase 5 §6 extension, plan R1 mitigation 4):
the K=2 same-segment base score is chemically-local — it rewards a
shape purely on the gradient at the cell's own ports, so a shape
can score well without its reciprocal neighbour having any viable
option. This leads the unaugmented solver to lock in
locally-greedy shortcut paths on winding Hamiltonians. The
consensus bonus per ``_consensus_bonus`` adds, for each candidate
shape, the sum over its open ports of the *best allowed
reciprocal shape's base score at the neighbour in that
direction*. Empirically on ``tiny_3x3``, this lifts argmax
convergence from a 5-cell shortcut prefix to 7 of 9 Hamiltonian
cells — a clear improvement, but not a full solve. The remaining
two cells sit in symmetric-gradient positions where no
purely-chemical reward (even augmented) can discriminate the
Hamiltonian choice from a shortcut because the multiplicative
thresholded update at §4.4 collapses score magnitudes to a
single R-vs-P bit and the argmax tiebreak then resolves to the
lowest shape index. Design R3 identifies this as the regime
Phase 6's noise/restart dynamics are designed to cover.
"""

from typing import Final

import numpy as np
from numpy.typing import NDArray

from .direction import Direction
from .puzzle import Puzzle
from .shapes import NUM_SHAPES, THROUGH_SHAPES, Shape, open_ports

_PORT_ORDER: Final[tuple[Direction, ...]] = (
    Direction.N,
    Direction.E,
    Direction.S,
    Direction.W,
)

_EXPECTED_NDIM: Final[int] = 3
_K2_SEGMENTS: Final[int] = 1

# Weight of the consensus bonus relative to the base §6 score. A port-
# aligned neighbour contributes its best reciprocal-shape score scaled
# by this factor; summed over a shape's 1–2 open ports, this is large
# enough to break the K=2 same-segment score's indifference to winding
# Hamiltonian interiors (plan R1 mitigation 4) without drowning out the
# base gradient signal.
CONSENSUS_WEIGHT: Final[float] = 10.0

# Number of message-passing iterations for the consensus bonus. One
# iteration only reasons about immediate neighbours; additional passes
# let a cell's score reflect the viability of two-step and longer
# reciprocal chains, which is what breaks the "disconnected endpoint
# vs. coherent far path" tie on the K=2 tiny puzzle where a single
# pass only sees the ambiguity between directly-adjacent candidates.
CONSENSUS_ITERATIONS: Final[int] = 1


def score_shapes(
    chems: NDArray[np.float32],
    puzzle: Puzzle,
    allowed: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """Compute the per-(cell, shape) alignment score ``sigma(s)``.

    Args:
        chems: Chemical concentration tensor of shape
            ``(N, N, K-1)`` where ``N = puzzle.size``. Must be C-
            contiguous ``float32``; typically the ``.chems`` field
            of an :class:`~zip_ca.engine.EngineState`.
        puzzle: The puzzle whose size gates the output shape.
        allowed: Boolean mask of shape ``(N, N, 10)`` from
            :func:`~zip_ca.actions.build_allowed_mask`. Disallowed
            entries receive ``-np.inf`` in the output so downstream
            ``argmax`` and threshold comparisons behave
            consistently.

    Returns:
        ``NDArray[np.float32]`` of shape ``(N, N, 10)``. Entry
        ``scores[i, j, s]`` is the §6 score for shape ``Shape(s)``
        at cell ``(i, j)``, or ``-np.inf`` if ``allowed[i, j, s]``
        is ``False``.

    Raises:
        ValueError: If ``chems`` has the wrong rank or its first
            two dimensions do not match ``puzzle.size``.
    """
    n = puzzle.size
    if chems.ndim != _EXPECTED_NDIM or chems.shape[:2] != (n, n):
        msg = (
            f"chems must have shape ({n}, {n}, K-1); got {chems.shape}"
        )
        raise ValueError(msg)
    k_segments = chems.shape[2]

    gradients = _build_gradients(chems, n)
    scores = np.full((n, n, NUM_SHAPES), -np.inf, dtype=np.float32)

    # Iterating over 10 shapes keeps the per-shape port-index
    # bookkeeping in one readable place; vectorising would require a
    # (10, 4) lookup table and still pay the cost of gather-selects.
    for shape in Shape:
        ports = open_ports(shape)
        # Sorting the port indices makes the assignment order
        # deterministic independent of frozenset iteration order,
        # which is important for reproducibility of any downstream
        # test that snapshots `scores`.
        port_indices = sorted(_PORT_ORDER.index(p) for p in ports)
        if shape in THROUGH_SHAPES:
            a, b = port_indices
            scores[..., int(shape)] = _through_score(gradients, a, b, k_segments)
        else:
            (a,) = port_indices
            scores[..., int(shape)] = _endpoint_score(gradients, a)

    base = np.where(allowed, scores, np.float32(-np.inf)).astype(np.float32)
    augmented = base
    for _ in range(CONSENSUS_ITERATIONS):
        bonus = _consensus_bonus(augmented, n)
        augmented = np.where(
            allowed,
            base + CONSENSUS_WEIGHT * bonus,
            np.float32(-np.inf),
        ).astype(np.float32)
    return augmented


def _build_gradients(
    chems: NDArray[np.float32], n: int,
) -> NDArray[np.float32]:
    """Build the ``(N, N, 4, K-1)`` directional-gradient tensor.

    Uses :func:`numpy.roll` for each of the four cardinal
    directions; rolled values that wrap around the grid boundary
    are explicitly zeroed so cells with an out-of-bounds port do
    not receive a spurious signal from the opposite edge. This is
    defence-in-depth — the allowed mask already forbids any shape
    opening such a port — but zeroing keeps the gradient tensor
    self-consistent for any future consumer that does not apply
    the mask.
    """
    u_north = np.roll(chems, shift=+1, axis=0)
    u_east = np.roll(chems, shift=-1, axis=1)
    u_south = np.roll(chems, shift=-1, axis=0)
    u_west = np.roll(chems, shift=+1, axis=1)
    u_neigh = np.stack([u_north, u_east, u_south, u_west], axis=2)
    gradients = (u_neigh - chems[..., None, :]).astype(np.float32)

    # Zero the wrapped rows/cols so the boundary gradient is well-
    # defined (0.0) rather than whatever the opposite edge happens
    # to hold.
    gradients[0, :, 0, :] = 0.0
    gradients[:, n - 1, 1, :] = 0.0
    gradients[n - 1, :, 2, :] = 0.0
    gradients[:, 0, 3, :] = 0.0
    return gradients


def _through_score(
    gradients: NDArray[np.float32],
    a: int,
    b: int,
    k_segments: int,
) -> NDArray[np.float32]:
    """Score a through-shape whose ports are at indices ``a``, ``b``.

    For ``K-1 == 1`` returns the same-segment score ``g_a + g_b``
    (see module docstring for motivation). For ``K-1 >= 2`` returns
    the design §6.2 best-consecutive-pair score, symmetrised over
    port orientation.
    """
    g_a = gradients[..., a, :]
    g_b = gradients[..., b, :]
    if k_segments == _K2_SEGMENTS:
        return (g_a + g_b).squeeze(-1).astype(np.float32)
    fwd = g_a[..., :-1] + g_b[..., 1:]
    bwd = g_a[..., 1:] + g_b[..., :-1]
    return np.maximum(fwd, bwd).max(axis=-1).astype(np.float32)


def _endpoint_score(
    gradients: NDArray[np.float32], a: int,
) -> NDArray[np.float32]:
    """Score an endpoint shape whose lone port is at index ``a``.

    Picks the best adjacent-segment gradient. For K=2 this trivially
    equals ``g_a[..., 0]``; for larger K it heuristically selects
    the segment with the steepest outward ascent, which degrades
    gracefully at the W_1 and W_K cells where only one segment is
    strongly non-zero at steady state.
    """
    return gradients[..., a, :].max(axis=-1).astype(np.float32)


def _consensus_bonus(
    base: NDArray[np.float32], n: int,
) -> NDArray[np.float32]:
    """Port-aligned neighbour-best bonus over the base scores.

    For each cell ``(i, j)`` and each candidate shape ``s`` with open
    ports ``P(s)``, accumulates ``Σ_{d ∈ P(s)} max_{s'} base(n_d, s')``
    where the inner max runs over shapes ``s'`` at the neighbour cell
    ``n_d = (i, j) + delta(d)`` that open the reciprocal port
    ``opposite(d)``. The bonus rewards shapes whose chosen ports have
    strong reciprocal candidates waiting on the other side,
    suppressing locally-greedy shortcut shapes in favour of
    globally-coherent path topologies (plan R1 mitigation 4).

    Boundary: out-of-grid neighbours contribute zero by explicit
    zeroing of the rolled rows/columns, matching the convention used
    by :func:`_build_gradients`. Cells whose neighbour has no
    allowed reciprocal shape yield ``-inf`` from the inner max; we
    clamp those to zero so a single degenerate neighbour cannot
    poison every shape at the current cell — the base-score ``-inf``
    mask on the current cell already enforces disallowance.
    """
    # Per-direction per-neighbour best score: for axis d in
    # _PORT_ORDER, neighbour_best[i, j, d] is the max base score
    # across shapes at the neighbour in direction d that open the
    # reciprocal port.
    neighbour_best = np.zeros((n, n, len(_PORT_ORDER)), dtype=np.float32)
    for d_idx, direction in enumerate(_PORT_ORDER):
        reciprocal = direction.opposite()
        reciprocal_shapes = [
            int(s) for s in Shape if reciprocal in open_ports(s)
        ]
        # Max over reciprocal shapes at each cell — this is the best
        # score any valid "connect back through port `reciprocal`"
        # choice achieves.
        local_best = base[..., reciprocal_shapes].max(axis=-1)
        local_best = np.where(np.isfinite(local_best), local_best, np.float32(0.0)).astype(np.float32)

        dr, dc = direction.delta
        # Roll by (-dr, -dc): values at position (i+dr, j+dc) land at
        # (i, j), i.e. the neighbour's score becomes visible at the
        # current cell.
        rolled = np.roll(local_best, shift=(-dr, -dc), axis=(0, 1)).astype(np.float32)
        neighbour_best[..., d_idx] = rolled

    # Zero wrap-around rows/cols so out-of-grid neighbours contribute
    # nothing (same convention as _build_gradients).
    neighbour_best[0, :, 0] = 0.0      # N: row 0 has no north neighbour
    neighbour_best[:, n - 1, 1] = 0.0  # E: col n-1 has no east neighbour
    neighbour_best[n - 1, :, 2] = 0.0  # S: row n-1 has no south neighbour
    neighbour_best[:, 0, 3] = 0.0      # W: col 0 has no west neighbour

    bonus = np.zeros((n, n, NUM_SHAPES), dtype=np.float32)
    for shape in Shape:
        for port in open_ports(shape):
            bonus[..., int(shape)] += neighbour_best[..., _PORT_ORDER.index(port)]
    return bonus
