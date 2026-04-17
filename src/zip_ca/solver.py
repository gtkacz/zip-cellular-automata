"""Restart-orchestrating solver - design §9.3-§9.4.

Wraps :class:`~zip_ca.EngineState` in an outer ``run_id`` loop. Each
run is independent: fresh engine state, independent splitmix64
stream, fresh :class:`~zip_ca.QuiescenceDetector`. The only state
that accumulates across runs is the append-only :class:`RunStats`
list used for diagnostics and best-prefix selection.

A run terminates on one of three outcomes:

- ``solved``: quiescence reached and :func:`~zip_ca.trace_path`
  accepts the shape grid. The solver returns immediately with
  ``ok=True``.
- ``quiescent_invalid``: quiescence reached but the trace is not a
  valid Hamiltonian path. Restart with the next ``run_id``.
- ``timeout``: ``max_ticks`` elapsed without quiescence. Restart.

On exhaustion, the solver returns the best (longest valid prefix)
trace observed across all runs so the operator can still diagnose
the failure visually.
"""

from dataclasses import dataclass
from typing import Final

from .engine import ETA_QUIESCENCE_GATE, T_WARM, EngineState, eta_schedule
from .puzzle import Puzzle
from .quiescence import QuiescenceDetector
from .trace import TraceResult, trace_path

R_MAX: Final[int] = 20
T_MAX: Final[int] = 5000

_OUTCOME_SOLVED: Final[str] = "solved"
_OUTCOME_QUIESCENT_INVALID: Final[str] = "quiescent_invalid"
_OUTCOME_TIMEOUT: Final[str] = "timeout"


@dataclass(frozen=True, slots=True)
class RunStats:
    """Per-run diagnostics emitted by :func:`solve`.

    Attributes:
        run_id: Zero-based restart generation for this run.
        tick_count: Number of ticks executed before termination.
        outcome: One of ``"solved"``, ``"quiescent_invalid"``,
            ``"timeout"``.
        trace: The :class:`TraceResult` at termination. On timeout
            this is the trace of whatever shape grid the engine was
            holding at the last tick.
        stable_ticks: The :class:`QuiescenceDetector`'s stable-tick
            counter at termination.
    """

    run_id: int
    tick_count: int
    outcome: str
    trace: TraceResult
    stable_ticks: int


@dataclass(frozen=True, slots=True)
class SolveResult:
    """Aggregate outcome of :func:`solve`.

    Attributes:
        ok: ``True`` iff at least one run solved the puzzle.
        runs: One :class:`RunStats` per attempted run, in call order.
        best_run_id: The run whose trace had the longest valid
            prefix. On ``ok=True`` this is the run that solved; on
            ``ok=False`` it is the closest-to-solution run, with
            earlier ``run_id`` winning ties.
        best_trace: The trace from ``best_run_id``.
    """

    ok: bool
    runs: tuple[RunStats, ...]
    best_run_id: int
    best_trace: TraceResult


def solve(
    puzzle: Puzzle,
    *,
    max_runs: int = R_MAX,
    max_ticks: int = T_MAX,
) -> SolveResult:
    """Run the dual-layer CA with restarts until solved or exhausted.

    Args:
        puzzle: The puzzle to solve.
        max_runs: Maximum number of independent runs before giving
            up. Defaults to :data:`R_MAX` (§12).
        max_ticks: Per-run tick budget before declaring timeout.
            Defaults to :data:`T_MAX` (§12).

    Returns:
        A :class:`SolveResult`. Early-exits on the first solving run;
        otherwise exhausts ``max_runs`` before returning the best
        partial trace.

    Raises:
        ValueError: If ``max_runs < 1`` or ``max_ticks < 1``.
    """
    if max_runs < 1:
        msg = f"max_runs must be >= 1 (got {max_runs})"
        raise ValueError(msg)
    if max_ticks < 1:
        msg = f"max_ticks must be >= 1 (got {max_ticks})"
        raise ValueError(msg)

    runs: list[RunStats] = []
    best_run_id = 0
    best_trace: TraceResult = TraceResult(ok=False, reason="no runs", path=())

    for run_id in range(max_runs):
        engine = EngineState.fresh(puzzle, run_id=run_id)
        detector = QuiescenceDetector()
        outcome, trace, stable_ticks = _run_one(engine, detector, max_ticks)
        runs.append(
            RunStats(
                run_id=run_id,
                tick_count=engine.tick_count,
                outcome=outcome,
                trace=trace,
                stable_ticks=stable_ticks,
            ),
        )
        if trace.ok:
            return SolveResult(
                ok=True,
                runs=tuple(runs),
                best_run_id=run_id,
                best_trace=trace,
            )
        # Ties resolved by earliest run_id (strict >) for deterministic
        # diagnostics; the first run that reached a given prefix length
        # wins.
        if len(trace.path) > len(best_trace.path):
            best_run_id = run_id
            best_trace = trace

    return SolveResult(
        ok=False,
        runs=tuple(runs),
        best_run_id=best_run_id,
        best_trace=best_trace,
    )


def _run_one(
    engine: EngineState,
    detector: QuiescenceDetector,
    max_ticks: int,
) -> tuple[str, TraceResult, int]:
    """Drive one engine to quiescence, solve, or timeout.

    Returns a ``(outcome, trace, stable_ticks)`` triple. On timeout,
    ``trace`` is the trace of the engine's final shape grid - useful
    for best-prefix selection even when quiescence was never reached.
    """
    while engine.tick_count < max_ticks:
        engine.tick()
        # The quiescence detector cannot report anything meaningful
        # during the warm-up window - shapes are invariant by design
        # §12 T_warm, which would trigger a spurious "quiescent"
        # signal on tick T_STABLE.
        if engine.tick_count <= T_WARM:
            continue
        quiescent = detector.update(engine.shapes)
        # R1 mitigation: do not trust quiescence while noise is still
        # strong. With high-magnitude noise firing on ~eta(t) cells per
        # tick, the argmax is sticky (reward reasserts the same winner)
        # but still in the middle of its exploration budget; reporting
        # "quiescent" here returns a trajectory that had no opportunity
        # to flip. Once eta(t) < ETA_QUIESCENCE_GATE the run has
        # effectively exhausted its noise budget and any remaining
        # stability is genuine.
        if quiescent and eta_schedule(engine.tick_count) < ETA_QUIESCENCE_GATE:
            trace = trace_path(engine.shapes, engine.puzzle)
            outcome = _OUTCOME_SOLVED if trace.ok else _OUTCOME_QUIESCENT_INVALID
            return outcome, trace, detector.ticks_since_flip()
    return (
        _OUTCOME_TIMEOUT,
        trace_path(engine.shapes, engine.puzzle),
        detector.ticks_since_flip(),
    )
