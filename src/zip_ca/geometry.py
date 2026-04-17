"""Canonical undirected edges between grid cells.

A wall between two adjacent cells can be expressed in JSON from either
endpoint's perspective. ``Edge`` collapses both perspectives to a single
canonical form (``a < b`` lexicographically) so that ``frozenset[Edge]``
membership and equality are unambiguous.
"""

from dataclasses import dataclass

Cell = tuple[int, int]


@dataclass(frozen=True, slots=True)
class Edge:
    """An undirected edge between two distinct grid cells in canonical order.

    The ``__post_init__`` validator forbids the un-canonical form so that the
    only way to obtain an ``Edge`` with arbitrary endpoint order is through
    :func:`canonical_edge`.

    Attributes:
        a: The lexicographically smaller endpoint ``(row, col)``.
        b: The lexicographically larger endpoint ``(row, col)``.
    """

    a: Cell
    b: Cell

    def __post_init__(self) -> None:
        """Validate canonical ordering."""
        if self.a >= self.b:
            raise ValueError(
                f"Edge endpoints must satisfy a < b lexicographically; got a={self.a}, b={self.b}"
            )


def canonical_edge(c1: Cell, c2: Cell) -> Edge:
    """Build an :class:`Edge` in canonical form regardless of input order.

    Args:
        c1: One endpoint as ``(row, col)``.
        c2: The other endpoint as ``(row, col)``.

    Returns:
        An ``Edge`` with the smaller cell as ``a`` and the larger as ``b``.

    Raises:
        ValueError: If ``c1 == c2`` (self-loops are not edges).
    """
    if c1 == c2:
        raise ValueError(f"Self-loop edge from {c1} to itself is not permitted")
    if c1 < c2:
        return Edge(a=c1, b=c2)
    return Edge(a=c2, b=c1)
