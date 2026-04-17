"""Rolling-counter quiescence detector for the Layer-1 argmax.

Implements ``docs/design.md`` §9.1: an external observer that tracks
the per-tick argmax shape grid and reports quiescence once the grid
has been byte-identical for ``T_STABLE = 30`` consecutive ticks. The
detector is deliberately outside :class:`~zip_ca.engine.EngineState`
to preserve the §3.1 "pure-CA" property — ``EngineState.tick`` has
no termination condition; termination is an external decision.

The detector is dtype-agnostic to the shape array (any integer
dtype works with :func:`numpy.array_equal`), but typed against
:class:`numpy.int8` because that is what the engine produces. A
defensive ``copy()`` is taken on every flip to decouple the
detector's lifetime from any array the engine may later decide to
mutate in place.
"""

from typing import Final

import numpy as np
from numpy.typing import NDArray

T_STABLE: Final[int] = 30


class QuiescenceDetector:
    """Counts consecutive ticks with no argmax flip.

    A *flip* is any elementwise change in the shape grid between
    consecutive calls to :meth:`update`. The detector reports
    quiescence (``True``) as soon as the consecutive-stable count
    reaches ``window``; subsequent calls continue to report
    quiescence until a flip resets the counter.
    """

    __slots__ = ("_window", "_prev", "_stable_ticks")

    def __init__(self, window: int = T_STABLE) -> None:
        """Initialise the detector.

        Args:
            window: Number of consecutive flip-free ticks required
                to declare quiescence. Defaults to
                :data:`T_STABLE`.

        Raises:
            ValueError: If ``window < 1``.
        """
        if window < 1:
            msg = f"window must be >= 1 (got {window})"
            raise ValueError(msg)
        self._window = window
        self._prev: NDArray[np.int8] | None = None
        self._stable_ticks = 0

    def update(self, shapes: NDArray[np.int8]) -> bool:
        """Ingest the current tick's shape grid.

        Args:
            shapes: The engine's current ``(N, N)`` shape array.

        Returns:
            ``True`` iff the detector has observed at least
            ``window`` consecutive ticks without any cell flipping
            (the first call returns ``False`` unless ``window == 0``,
            which is forbidden by the constructor).
        """
        if self._prev is not None and np.array_equal(shapes, self._prev):
            self._stable_ticks += 1
        else:
            self._stable_ticks = 0
            self._prev = shapes.copy()
        return self._stable_ticks >= self._window

    def ticks_since_flip(self) -> int:
        """Return the current consecutive-stable tick count."""
        return self._stable_ticks
