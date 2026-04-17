"""JSON puzzle parser, validator, and canonical emitter.

This module is the only place in the codebase where puzzle JSON is read or
written. All eight validation invariants from ``docs/design.md`` §11.1.4 are
enforced here; nothing downstream needs to re-validate.
"""

import json
from pathlib import Path
from typing import Any, cast

from .direction import Direction
from .geometry import Cell, Edge, canonical_edge
from .puzzle import Puzzle, PuzzleValidationError, Waypoint

_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {"size", "waypoints", "walls", "_comment", "name", "source"}
)
_REQUIRED_TOP_LEVEL_KEYS: frozenset[str] = frozenset({"size", "waypoints"})
_WAYPOINT_KEYS: frozenset[str] = frozenset({"row", "col", "number"})
_WALL_KEYS: frozenset[str] = frozenset({"row", "col", "blocked"})
_DIR_STRINGS: frozenset[str] = frozenset(d.value for d in Direction)

# A puzzle smaller than 2x2 cannot host two distinct waypoints; a Zip puzzle
# with fewer than two waypoints is degenerate by definition.
_MIN_GRID_SIZE: int = 2
_MIN_WAYPOINT_COUNT: int = 2


def parse_puzzle_obj(obj: object) -> Puzzle:
    """Validate a parsed JSON object and return a :class:`Puzzle`.

    Args:
        obj: A Python object produced by ``json.loads`` (expected to be a
            dict matching the schema in ``docs/design.md`` §11.1.2).

    Returns:
        A fully validated, canonicalised ``Puzzle``.

    Raises:
        PuzzleValidationError: If any of the eight data-model invariants
            from ``docs/design.md`` §11.1.4 fail.
    """
    if not isinstance(obj, dict):
        raise PuzzleValidationError(
            f"Top-level JSON value must be an object; got {type(obj).__name__}"
        )
    data = cast(dict[str, Any], obj)

    keys: set[str] = set(data.keys())
    unknown = keys - _TOP_LEVEL_KEYS
    if unknown:
        raise PuzzleValidationError(f"Unknown top-level key(s): {sorted(unknown)}")
    missing = _REQUIRED_TOP_LEVEL_KEYS - keys
    if missing:
        raise PuzzleValidationError(f"Missing required top-level key(s): {sorted(missing)}")

    size = _parse_size(data["size"])
    waypoints = _parse_waypoints(data["waypoints"], size)
    walled_edges = _parse_walls(data.get("walls", []), size)
    name = _parse_optional_str(data.get("name"), "name")
    source = _parse_optional_str(data.get("source"), "source")

    return Puzzle(
        size=size,
        waypoints=waypoints,
        walled_edges=walled_edges,
        name=name,
        source=source,
    )


def parse_puzzle(text: str) -> Puzzle:
    """Parse JSON text into a validated :class:`Puzzle`.

    Args:
        text: JSON-encoded puzzle.

    Returns:
        A fully validated ``Puzzle``.

    Raises:
        PuzzleValidationError: On malformed JSON or any invariant violation.
    """
    try:
        obj: object = json.loads(text)

    except json.JSONDecodeError as exc:
        raise PuzzleValidationError(f"Invalid JSON: {exc}") from exc

    return parse_puzzle_obj(obj)


def load_puzzle(path: str | Path) -> Puzzle:
    """Read a JSON file from disk and return the validated :class:`Puzzle`.

    Args:
        path: Filesystem path to a puzzle JSON file.

    Returns:
        A fully validated ``Puzzle``.

    Raises:
        PuzzleValidationError: On invalid JSON or invariant violation.
        OSError: If the file cannot be read (propagated unchanged so genuine
            I/O bugs are not swallowed as input-validation errors).
    """
    return parse_puzzle(Path(path).read_text(encoding="utf-8"))


def to_json_obj(puzzle: Puzzle) -> dict[str, Any]:
    """Emit a :class:`Puzzle` as a JSON-ready dict in canonical key order.

    The canonical form is deterministic: keys are inserted in the order
    ``size, waypoints, walls, name, source`` (omitted when ``None`` or
    empty), waypoints are sorted by ``.number``, and each wall is emitted
    once with the lexicographically smaller cell as the owner.

    Args:
        puzzle: The puzzle to serialise.

    Returns:
        A dict suitable for ``json.dumps``.
    """
    out: dict[str, Any] = {
        "size": puzzle.size,
        "waypoints": [{"row": w.row, "col": w.col, "number": w.number} for w in puzzle.waypoints],
    }
    walls = _emit_walls(puzzle.walled_edges)
    if walls:
        out["walls"] = walls
    if puzzle.name is not None:
        out["name"] = puzzle.name
    if puzzle.source is not None:
        out["source"] = puzzle.source
    return out


def dump_puzzle(puzzle: Puzzle) -> str:
    """Serialise a :class:`Puzzle` to canonical JSON text (UTF-8, 2-space indent)."""
    return json.dumps(to_json_obj(puzzle), indent=2, sort_keys=False) + "\n"


def _parse_size(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PuzzleValidationError(f"size must be int; got {type(value).__name__}")
    if value < _MIN_GRID_SIZE:
        raise PuzzleValidationError(f"size must be >= {_MIN_GRID_SIZE}; got {value}")
    return value


def _parse_optional_str(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise PuzzleValidationError(
            f"{label} must be string or omitted; got {type(value).__name__}"
        )
    return value


def _parse_waypoints(raw: object, size: int) -> tuple[Waypoint, ...]:
    if not isinstance(raw, list):
        raise PuzzleValidationError(f"waypoints must be a list; got {type(raw).__name__}")

    entries = cast(list[Any], raw)
    waypoints = tuple(_parse_waypoint(entry, i, size) for i, entry in enumerate(entries))

    numbers = sorted(w.number for w in waypoints)
    k = len(numbers)

    if k < _MIN_WAYPOINT_COUNT:
        raise PuzzleValidationError(
            f"waypoints must contain at least {_MIN_WAYPOINT_COUNT} entries; got {k}"
        )

    expected = list(range(1, k + 1))

    if numbers != expected:
        raise PuzzleValidationError(
            f"waypoint numbers must form 1..{k} with no gaps or duplicates; got {numbers}"
        )

    coords = [Cell(w.row, w.col) for w in waypoints]

    if len(set(coords)) != len(coords):
        seen: set[Cell] = set()
        dupes: set[Cell] = set()

        for c in coords:
            if c in seen:
                dupes.add(c)

            else:
                seen.add(c)

        raise PuzzleValidationError(f"waypoints share coordinates: {sorted(dupes)}")

    return tuple(sorted(waypoints, key=lambda w: w.number))


def _parse_waypoint(entry: object, index: int, size: int) -> Waypoint:
    if not isinstance(entry, dict):
        raise PuzzleValidationError(
            f"waypoints[{index}] must be object; got {type(entry).__name__}"
        )
    data = cast(dict[str, Any], entry)
    keys: set[str] = set(data.keys())
    unknown = keys - _WAYPOINT_KEYS
    if unknown:
        raise PuzzleValidationError(f"waypoints[{index}] has unknown key(s): {sorted(unknown)}")
    missing = _WAYPOINT_KEYS - keys
    if missing:
        raise PuzzleValidationError(f"waypoints[{index}] missing key(s): {sorted(missing)}")

    row = _parse_int_field(data["row"], f"waypoints[{index}].row")
    col = _parse_int_field(data["col"], f"waypoints[{index}].col")
    number = _parse_int_field(data["number"], f"waypoints[{index}].number")

    if not 0 <= row < size:
        raise PuzzleValidationError(f"waypoints[{index}].row={row} out of bounds [0, {size})")
    if not 0 <= col < size:
        raise PuzzleValidationError(f"waypoints[{index}].col={col} out of bounds [0, {size})")
    if number < 1:
        raise PuzzleValidationError(f"waypoints[{index}].number={number} must be >= 1")

    return Waypoint(row=row, col=col, number=number)


def _parse_walls(raw: object, size: int) -> frozenset[Edge]:
    if not isinstance(raw, list):
        raise PuzzleValidationError(f"walls must be a list; got {type(raw).__name__}")
    entries = cast(list[Any], raw)

    edges: set[Edge] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise PuzzleValidationError(
                f"walls[{index}] must be object; got {type(entry).__name__}"
            )
        data = cast(dict[str, Any], entry)
        keys: set[str] = set(data.keys())
        unknown = keys - _WALL_KEYS
        if unknown:
            raise PuzzleValidationError(f"walls[{index}] has unknown key(s): {sorted(unknown)}")
        missing = _WALL_KEYS - keys
        if missing:
            raise PuzzleValidationError(f"walls[{index}] missing key(s): {sorted(missing)}")

        row = _parse_int_field(data["row"], f"walls[{index}].row")
        col = _parse_int_field(data["col"], f"walls[{index}].col")
        if not 0 <= row < size:
            raise PuzzleValidationError(f"walls[{index}].row={row} out of bounds [0, {size})")
        if not 0 <= col < size:
            raise PuzzleValidationError(f"walls[{index}].col={col} out of bounds [0, {size})")

        blocked_raw = data["blocked"]
        if not isinstance(blocked_raw, list):
            raise PuzzleValidationError(
                f"walls[{index}].blocked must be list; got {type(blocked_raw).__name__}"
            )
        blocked = cast(list[Any], blocked_raw)
        for j, dir_str in enumerate(blocked):
            if not isinstance(dir_str, str):
                raise PuzzleValidationError(
                    f"walls[{index}].blocked[{j}] must be string; got {type(dir_str).__name__}"
                )
            if dir_str not in _DIR_STRINGS:
                raise PuzzleValidationError(
                    f"walls[{index}].blocked[{j}]={dir_str!r} not in {sorted(_DIR_STRINGS)}"
                )
            direction = Direction(dir_str)
            dr, dc = direction.delta
            neighbour: Cell = Cell(row + dr, col + dc)
            # Perimeter walls (off-grid neighbour) are silently dropped:
            # the boundary filter in design §4.2 already removes shapes whose
            # open ports point off-grid, so a per-cell wall there is redundant.
            if not (0 <= neighbour[0] < size and 0 <= neighbour[1] < size):
                continue
            edges.add(canonical_edge(Cell(row, col), neighbour))

    return frozenset(edges)


def _parse_int_field(value: object, label: str) -> int:
    # `isinstance(True, int)` is True in Python; treat bools as type errors so
    # `{"row": true}` does not silently coerce to 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise PuzzleValidationError(f"{label} must be int; got {type(value).__name__}")
    return value


def _emit_walls(walled_edges: frozenset[Edge]) -> list[dict[str, Any]]:
    walls: list[dict[str, Any]] = []
    for edge in sorted(walled_edges, key=lambda e: (e.a, e.b)):
        dr = edge.b[0] - edge.a[0]
        dc = edge.b[1] - edge.a[1]
        # Canonical edges have a < b lexicographically, so the step from a to
        # b is always either (1, 0) (south) or (0, 1) (east); walls that span
        # adjacent cells cannot have any other step.
        if (dr, dc) == (1, 0):
            direction = "S"
        elif (dr, dc) == (0, 1):
            direction = "E"
        else:
            raise AssertionError(f"Non-canonical or non-adjacent edge slipped through: {edge}")
        walls.append({"row": edge.a[0], "col": edge.a[1], "blocked": [direction]})
    return walls
