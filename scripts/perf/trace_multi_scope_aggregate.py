#!/usr/bin/env python3
"""Aggregate Chrome-trace launches for several profiler scopes in one pass.

Large Protenix traces can be multiple gigabytes.  Running
``trace_scope_aggregate.py`` separately for each scope works, but it rereads the
same file many times.  This helper does the same containment-based attribution
for several scopes after one interval pass and one event pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.perf.trace_aggregate import event_objects, top_rows
from scripts.perf.trace_scope_aggregate import (
    Interval,
    IntervalIndex,
    _event_interval,
)


def _add(table: defaultdict[str, list[float]], name: str, duration_us: float) -> None:
    row = table[name]
    row[0] += 1
    row[1] += duration_us
    row[2] = max(row[2], duration_us)


def _empty_scope() -> dict[str, Any]:
    return {
        "matched_events": 0,
        "category_counts": Counter(),
        "by_category": defaultdict(lambda: [0, 0.0, 0.0]),
        "kernels": defaultdict(lambda: [0, 0.0, 0.0]),
        "runtime": defaultdict(lambda: [0, 0.0, 0.0]),
        "copies": defaultdict(lambda: [0, 0.0, 0.0]),
        "cpu_ops": defaultdict(lambda: [0, 0.0, 0.0]),
    }


def _matches(name: str, scope: str, exact: bool) -> bool:
    return name == scope if exact else scope in name


def find_scope_intervals(
    trace: Path,
    scopes: list[str],
    exact: bool,
    category: str | None,
) -> dict[str, list[Interval]]:
    intervals = {scope: [] for scope in scopes}
    for event in event_objects(trace):
        if category is not None and str(event.get("cat", "")) != category:
            continue
        name = str(event.get("name", ""))
        for scope in scopes:
            if not _matches(name, scope, exact):
                continue
            interval = _event_interval(event)
            if interval is not None:
                intervals[scope].append(interval)
    for scope_intervals in intervals.values():
        scope_intervals.sort()
    return intervals


def add_event(scope_result: dict[str, Any], event: dict[str, Any]) -> None:
    name = str(event.get("name", ""))
    category = str(event.get("cat", ""))
    duration_us = float(event.get("dur") or 0.0)
    lower_category = category.lower()
    lower_name = name.lower()

    scope_result["matched_events"] += 1
    scope_result["category_counts"][category] += 1
    _add(scope_result["by_category"], f"{category}::{name}", duration_us)
    if "kernel" in lower_category:
        _add(scope_result["kernels"], name, duration_us)
    if "runtime" in lower_category or name.startswith(("cuda", "cu")):
        _add(scope_result["runtime"], name, duration_us)
    if (
        "memcpy" in lower_category
        or "memset" in lower_category
        or "memcpy" in lower_name
        or "memset" in lower_name
    ):
        _add(scope_result["copies"], name, duration_us)
    if lower_category == "cpu_op":
        _add(scope_result["cpu_ops"], name, duration_us)


def aggregate_scopes(
    trace: Path,
    intervals: dict[str, list[Interval]],
    top: int,
    overlap: bool,
) -> dict[str, Any]:
    indexes = {scope: IntervalIndex(scope_intervals) for scope, scope_intervals in intervals.items()}
    results = {scope: _empty_scope() for scope in intervals}

    for event in event_objects(trace):
        interval = _event_interval(event)
        if interval is None:
            continue
        start, end = interval
        for scope, index in indexes.items():
            inside = index.overlaps(start, end) if overlap else index.contains(start, end)
            if inside:
                add_event(results[scope], event)

    summarized: dict[str, Any] = {}
    for scope, raw in results.items():
        index = indexes[scope]
        summarized[scope] = {
            "scope_count": len(intervals[scope]),
            "merged_scope_intervals": len(index.intervals),
            "scope_total_ms": sum(end - start for start, end in index.intervals)
            / 1000.0,
            "matched_events": raw["matched_events"],
            "category_counts": raw["category_counts"].most_common(),
            "top_kernel": top_rows(raw["kernels"], top),
            "top_runtime": top_rows(raw["runtime"], top),
            "top_gpu_copy_memset": top_rows(raw["copies"], top),
            "top_cpu_op": top_rows(raw["cpu_ops"], top),
            "top_cat_name": top_rows(raw["by_category"], top),
        }
    return summarized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--scope", action="append", required=True)
    parser.add_argument("--scope-cat", help="Optional category for scope events.")
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--overlap", action="store_true")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    intervals = find_scope_intervals(args.trace, args.scope, args.exact, args.scope_cat)
    result = {
        "trace": str(args.trace),
        "exact": args.exact,
        "overlap": args.overlap,
        "scope_cat": args.scope_cat,
        "scopes": aggregate_scopes(args.trace, intervals, args.top, args.overlap),
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                scope: {
                    "scope_count": data["scope_count"],
                    "scope_total_ms": data["scope_total_ms"],
                    "top_kernel": data["top_kernel"][:5],
                }
                for scope, data in result["scopes"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
