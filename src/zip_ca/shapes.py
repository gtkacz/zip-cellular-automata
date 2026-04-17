"""Path-shape vocabulary for Layer 1 of the dual-layer CA.

The ``Shape`` ``IntEnum`` is the canonical vocabulary for the ten per-cell
path topologies from ``docs/design.md`` §4.1: six *through* shapes (two
open ports) and four *endpoint* shapes (one open port). Member values are
contiguous ``0..9`` so shapes are usable directly as a numpy index into
the ``(N, N, |S|)`` allowed-mask and probability arrays from §11.2.1.

Index ordering is binding — it is load-bearing once downstream arrays
exist, and must not be changed across releases:

==============  =======  =============  =========
Index           Member   Open ports     Class
==============  =======  =============  =========
0               H        {E, W}         through
1               V        {N, S}         through
2               NE       {N, E}         through
3               NW       {N, W}         through
4               SE       {S, E}         through
5               SW       {S, W}         through
6               END_N    {N}            endpoint
7               END_E    {E}            endpoint
8               END_S    {S}            endpoint
9               END_W    {W}            endpoint
==============  =======  =============  =========

The contiguous ``0..5`` / ``6..9`` split lets consumers slice
``allowed[..., :6]`` for through-shapes only and ``allowed[..., 6:]``
for endpoints only.
"""

from enum import IntEnum

from .direction import Direction


class Shape(IntEnum):
    """One of ten per-cell path-shape topologies."""

    H = 0
    V = 1
    NE = 2
    NW = 3
    SE = 4
    SW = 5
    END_N = 6
    END_E = 7
    END_S = 8
    END_W = 9


_OPEN_PORTS: dict[Shape, frozenset[Direction]] = {
    Shape.H: frozenset({Direction.E, Direction.W}),
    Shape.V: frozenset({Direction.N, Direction.S}),
    Shape.NE: frozenset({Direction.N, Direction.E}),
    Shape.NW: frozenset({Direction.N, Direction.W}),
    Shape.SE: frozenset({Direction.S, Direction.E}),
    Shape.SW: frozenset({Direction.S, Direction.W}),
    Shape.END_N: frozenset({Direction.N}),
    Shape.END_E: frozenset({Direction.E}),
    Shape.END_S: frozenset({Direction.S}),
    Shape.END_W: frozenset({Direction.W}),
}

_PORTS_TO_SHAPE: dict[frozenset[Direction], Shape] = {
    ports: shape for shape, ports in _OPEN_PORTS.items()
}

NUM_SHAPES: int = len(Shape)

THROUGH_SHAPES: frozenset[Shape] = frozenset(
    {Shape.H, Shape.V, Shape.NE, Shape.NW, Shape.SE, Shape.SW}
)
ENDPOINT_SHAPES: frozenset[Shape] = frozenset({Shape.END_N, Shape.END_E, Shape.END_S, Shape.END_W})


def open_ports(shape: Shape) -> frozenset[Direction]:
    """Return the set of open edge-ports for ``shape``.

    Args:
        shape: A ``Shape`` member.

    Returns:
        The ports this shape opens through. Through-shapes have two;
        endpoint shapes have one.
    """
    return _OPEN_PORTS[shape]


def shape_for_ports(ports: frozenset[Direction]) -> Shape:
    """Inverse of :func:`open_ports`: the unique ``Shape`` for a port-set.

    Args:
        ports: Desired set of open ports.

    Returns:
        The unique ``Shape`` opening exactly ``ports``.

    Raises:
        KeyError: If ``ports`` does not match one of the ten valid
            port-sets (empty, three-port, or four-port sets are invalid;
            opposite-corner pairs like ``{N, N}`` are impossible by set
            semantics).
    """
    return _PORTS_TO_SHAPE[ports]


# Index ordering is load-bearing downstream; catch accidental reorders at
# import time rather than during a later mysterious array-indexing bug.
assert [s.value for s in Shape] == list(range(NUM_SHAPES))
