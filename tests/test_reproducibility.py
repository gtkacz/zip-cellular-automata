"""Byte-equal reproducibility invariant for the dual-layer CA.

Design §11.3.2 mandates this as a hard correctness gate: identical
``(puzzle, run_id)`` must yield byte-equal state sequences. Without
this property, restart loops are non-debuggable (a failing seed
cannot be re-played) and the splitmix64 PRNG choice in
:mod:`zip_ca.noise` is unverifiable.

Two tests here: the mandated byte-equal gate, and a counter-test that
catches accidental `run_id`-insensitivity in the noise hash (e.g. if
a regression dropped the `run_id` term from the splitmix64 mix, the
byte-equal test would still pass vacuously). Both tests live in the
same file because they validate the same invariant from opposite
sides.
"""

from pathlib import Path

import numpy as np
import pytest

from zip_ca import EngineState, Puzzle, load_puzzle

_TINY_PUZZLE: Path = Path("puzzles/tiny_3x3.json")
_TICKS: int = 100


@pytest.fixture
def tiny() -> Puzzle:
    return load_puzzle(_TINY_PUZZLE)


def test_byte_equal_under_repeat(tiny: Puzzle) -> None:
    """Design §11.3.2: ``(puzzle, run_id)`` must fully determine the trajectory."""
    state_a = EngineState.fresh(tiny, run_id=42)
    state_b = EngineState.fresh(tiny, run_id=42)
    for _ in range(_TICKS):
        state_a.tick()
        state_b.tick()
    assert np.array_equal(state_a.probs, state_b.probs), "probs diverged"
    assert np.array_equal(state_a.shapes, state_b.shapes), "shapes diverged"
    assert np.array_equal(state_a.chems, state_b.chems), "chems diverged"
    assert state_a.tick_count == state_b.tick_count


def test_distinct_run_ids_diverge(tiny: Puzzle) -> None:
    """Counter-test: distinct ``run_id`` values must yield distinct trajectories.

    If this passes but :func:`test_byte_equal_under_repeat` also passes
    vacuously (e.g. noise is disabled), the splitmix64 hash has lost
    its ``run_id`` dependency and reproducibility is a hollow guarantee.
    """
    state_0 = EngineState.fresh(tiny, run_id=0)
    state_1 = EngineState.fresh(tiny, run_id=1)
    for _ in range(_TICKS):
        state_0.tick()
        state_1.tick()
    same_probs = np.array_equal(state_0.probs, state_1.probs)
    same_shapes = np.array_equal(state_0.shapes, state_1.shapes)
    assert not (same_probs and same_shapes), "noise did not diverge trajectories across run_ids"
