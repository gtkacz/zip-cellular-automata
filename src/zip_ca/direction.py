"""Cardinal-direction primitives.

The ``Direction`` enum is the only sanctioned vocabulary type for grid ports
anywhere in the codebase. Plain strings ``"N"``/``"E"``/``"S"``/``"W"`` are
permitted **only** at the JSON parser boundary in :mod:`zip_ca.puzzle_io`
(binding rule from ``docs/design.md`` §11.1.1).

Coordinate convention (binding):

* Origin ``(0, 0)`` is the top-left cell.
* ``row`` index increases downward; ``col`` index increases rightward.
* ``N``: ``(-1, 0)``  ``E``: ``(0, +1)``  ``S``: ``(+1, 0)``  ``W``: ``(0, -1)``
"""

from enum import Enum


class Direction(Enum):
    """One of the four cardinal grid directions."""

    N = "N"
    E = "E"
    S = "S"
    W = "W"

    @property
    def delta(self) -> tuple[int, int]:
        """Return the ``(row_delta, col_delta)`` step for moving one cell in this direction."""
        return _DIRECTION_DELTAS[self]

    def opposite(self) -> "Direction":
        """Return the antipodal direction (``N`` ↔ ``S``, ``E`` ↔ ``W``)."""
        return _DIRECTION_OPPOSITES[self]


# Lookup tables live at module scope (not on the Enum) so that the type checker
# can give them precise types instead of inferring `Any` from
# `Enum.value`.
_DIRECTION_DELTAS: dict[Direction, tuple[int, int]] = {
    Direction.N: (-1, 0),
    Direction.E: (0, 1),
    Direction.S: (1, 0),
    Direction.W: (0, -1),
}

_DIRECTION_OPPOSITES: dict[Direction, Direction] = {
    Direction.N: Direction.S,
    Direction.S: Direction.N,
    Direction.E: Direction.W,
    Direction.W: Direction.E,
}
