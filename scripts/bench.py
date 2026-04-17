r"""Aggregate solver-rate benchmark across a puzzle corpus.

Usage:
    uv run python scripts/bench.py puzzles/tiny_3x3.json [...more puzzles]
    uv run python scripts/bench.py puzzles/*.json

For each puzzle listed on the command line, invokes
:func:`zip_ca.solve` with ``R_MAX`` runs and ``T_MAX`` ticks, records
solved/unsolved, and writes a structured report to ``out/bench.json``.
Exit code is 0 iff the aggregate solve rate meets or exceeds the
``--target`` threshold (default 0.80 per §12 / plan). The aggregate
report is intentionally machine-readable so the Phase-6 acceptance
gate can grep a single field without re-parsing logs.

The script is deliberately tolerant of missing ``.solution.json``
files - the §9.2 trace validator establishes solve status from the
shape grid alone; the solution file is only consulted at authoring
time to prove a valid Hamiltonian exists. If the operator points the
script at a puzzle whose solution has never been validated, the run
will still happen and the report will include ``solved: false`` on
false positives (which the trace validator should in principle
never report) - we do not silently skip unvalidated puzzles because
doing so would hide corpus maintenance bugs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Final, cast

from zip_ca import R_MAX, T_MAX, load_puzzle, solve

_DEFAULT_TARGET: Final[float] = 0.80
_DEFAULT_OUTPUT: Final[Path] = Path("out/bench.json")


def _bench_one(puzzle_path: Path, max_runs: int, max_ticks: int) -> dict[str, object]:
    """Run :func:`solve` on one puzzle and return a report dict."""
    puzzle = load_puzzle(puzzle_path)
    t0 = time.perf_counter()
    result = solve(puzzle, max_runs=max_runs, max_ticks=max_ticks)
    elapsed = time.perf_counter() - t0

    best = result.runs[result.best_run_id]
    return {
        "puzzle": str(puzzle_path),
        "size": puzzle.size,
        "solved": result.ok,
        "runs_attempted": len(result.runs),
        "best_run_id": result.best_run_id,
        "best_prefix_length": len(result.best_trace.path),
        "best_tick_count": best.tick_count,
        "best_stable_ticks": best.stable_ticks,
        "best_reason": result.best_trace.reason,
        "elapsed_seconds": round(elapsed, 3),
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point. Exit 0 iff solve-rate >= target."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "puzzles",
        nargs="+",
        type=Path,
        help="Puzzle JSON files to benchmark.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=R_MAX,
        help=f"Runs per puzzle (default {R_MAX}).",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=T_MAX,
        help=f"Ticks per run (default {T_MAX}).",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=_DEFAULT_TARGET,
        help=f"Required aggregate solve rate (default {_DEFAULT_TARGET}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Report JSON path (default {_DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args(argv)

    puzzles = cast(list[Path], args.puzzles)
    reports: list[dict[str, object]] = []
    for path in puzzles:
        report = _bench_one(path, int(args.max_runs), int(args.max_ticks))
        reports.append(report)
        status = "SOLVED" if report["solved"] else "FAILED"
        print(
            f"  {status:6s} {path!s:40s} "
            f"prefix={report['best_prefix_length']}/{report['size'] ** 2} "
            f"elapsed={report['elapsed_seconds']}s",
        )

    solved = sum(1 for r in reports if r["solved"])
    total = len(reports)
    solve_rate = solved / total if total else 0.0

    aggregate: dict[str, object] = {
        "total_puzzles": total,
        "solved": solved,
        "solve_rate": round(solve_rate, 4),
        "target": float(args.target),
        "passed": solve_rate >= float(args.target),
        "max_runs": int(args.max_runs),
        "max_ticks": int(args.max_ticks),
        "reports": reports,
    }

    output_path = cast(Path, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(aggregate, indent=2))

    print(
        f"\nAGGREGATE: {solved}/{total} solved ({solve_rate:.1%}); "
        f"target {float(args.target):.1%}; "
        f"report -> {output_path}",
    )
    return 0 if aggregate["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
