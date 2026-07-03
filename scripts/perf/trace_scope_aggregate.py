#!/usr/bin/env python3
"""Aggregate Chrome-trace events that occur inside a named profiler scope.

PyTorch's top-level profiler table is useful for broad bottleneck screens, but
after a few rounds it becomes too global: the largest kernels may come from the
pairformer, diffusion transformer, confidence head, or file dumping mixed
together.  This helper answers the more surgical question:

    "Inside this record_function range, which launches dominate?"

It streams the Chrome trace twice instead of loading multi-GB traces into
memory.  The first pass finds scope intervals by name, and the second pass sums
events whose time interval overlaps those scopes.
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.perf.trace_aggregate import event_objects, top_rows


Interval = tuple[float, float]


def _event_interval(event: dict[str, Any]) -> Interval | None:
    if event.get("ph") != "X":
        return None
    duration = event.get("dur")
    timestamp = event.get("ts")
    if duration is None or timestamp is None:
        return None
    start = float(timestamp)
    end = start + float(duration)
    if end <= start:
        return None
    return start, end


def _matches_scope(name: str, pattern: str, exact: bool) -> bool:
    return name == pattern if exact else pattern in name


def find_scope_intervals(trace: Path, pattern: str, exact: bool) -> list[Interval]:
    intervals: list[Interval] = []
    for event in event_objects(trace):
        name = str(event.get("name", ""))
        if not _matches_scope(name, pattern, exact):
            continue
        interval = _event_interval(event)
        if interval is not None:
            intervals.append(interval)
    intervals.sort()
    return intervals


def _merge_intervals(intervals: list[Interval]) -> list[Interval]:
    if not intervals:
        return []
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


class IntervalIndex:
    def __init__(self, intervals: list[Interval]) -> None:
        self.intervals = _merge_intervals(intervals)
        self.starts = [start for start, _end in self.intervals]

    def overlaps(self, start: float, end: float) -> bool:
        # Check the interval that starts at or immediately before ``end``.  If
        # its end is after ``start``, the event and scope overlap.
        idx = bisect.bisect_left(self.starts, end)
        if idx == 0:
            return False
        scope_start, scope_end = self.intervals[idx - 1]
        return scope_start < end and scope_end > start


def _add(table: defaultdict[str, list[float]], name: str, duration_us: float) -> None:
    row = table[name]
    row[0] += 1
    row[1] += duration_us
    row[2] = max(row[2], duration_us)


def aggregate_scope(
    trace: Path,
    intervals: list[Interval],
    top: int,
) -> dict[str, Any]:
    index = IntervalIndex(intervals)
    category_counts: Counter[str] = Counter()
    by_category = defaultdict(lambda: [0, 0.0, 0.0])
    kernels = defaultdict(lambda: [0, 0.0, 0.0])
    runtime = defaultdict(lambda: [0, 0.0, 0.0])
    copies = defaultdict(lambda: [0, 0.0, 0.0])
    cpu_ops = defaultdict(lambda: [0, 0.0, 0.0])
    matched_events = 0

    for event in event_objects(trace):
        interval = _event_interval(event)
        if interval is None:
            continue
        if not index.overlaps(*interval):
            continue

        matched_events += 1
        name = str(event.get("name", ""))
        category = str(event.get("cat", ""))
        duration_us = float(event.get("dur") or 0.0)
        lower_category = category.lower()
        lower_name = name.lower()

        category_counts[category] += 1
        _add(by_category, f"{category}::{name}", duration_us)
        if "kernel" in lower_category:
            _add(kernels, name, duration_us)
        if "runtime" in lower_category or name.startswith(("cuda", "cu")):
            _add(runtime, name, duration_us)
        if (
            "memcpy" in lower_category
            or "memset" in lower_category
            or "memcpy" in lower_name
            or "memset" in lower_name
        ):
            _add(copies, name, duration_us)
        if lower_category == "cpu_op":
            _add(cpu_ops, name, duration_us)

    return {
        "matched_events": matched_events,
        "merged_scope_intervals": len(index.intervals),
        "scope_total_ms": sum(end - start for start, end in index.intervals) / 1000.0,
        "category_counts": category_counts.most_common(),
        "top_kernel": top_rows(kernels, top),
        "top_runtime": top_rows(runtime, top),
        "top_gpu_copy_memset": top_rows(copies, top),
        "top_cpu_op": top_rows(cpu_ops, top),
        "top_cat_name": top_rows(by_category, top),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--scope", required=True, help="Scope name or substring.")
    parser.add_argument("--exact", action="store_true", help="Require exact name match.")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    intervals = find_scope_intervals(args.trace, args.scope, args.exact)
    result = {
        "trace": str(args.trace),
        "scope": args.scope,
        "exact": args.exact,
        "scope_count": len(intervals),
        **aggregate_scope(args.trace, intervals, args.top),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
