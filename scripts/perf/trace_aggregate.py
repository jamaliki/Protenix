#!/usr/bin/env python3
"""Aggregate PyTorch profiler Chrome traces without loading them into memory."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def seek_trace_events_array(handle: Any) -> None:
    key = b'"traceEvents"'
    overlap = len(key) - 1
    seen = b""
    absolute = 0
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            raise RuntimeError("traceEvents key not found")
        haystack = seen + chunk
        index = haystack.find(key)
        if index >= 0:
            handle.seek(absolute - len(seen) + index + len(key))
            break
        seen = haystack[-overlap:]
        absolute += len(chunk)

    while True:
        char = handle.read(1)
        if not char:
            raise RuntimeError("traceEvents array not found")
        if char == b"[":
            return


def event_objects(path: Path, chunk_size: int = 16 * 1024 * 1024) -> Any:
    with path.open("rb") as handle:
        seek_trace_events_array(handle)
        in_string = False
        escape = False
        depth = 0
        buffer = bytearray()

        while chunk := handle.read(chunk_size):
            for char in chunk:
                if depth == 0:
                    if char == 123:  # {
                        depth = 1
                        buffer.clear()
                        buffer.append(char)
                        in_string = False
                        escape = False
                    elif char == 93:  # ]
                        return
                    continue

                buffer.append(char)
                if in_string:
                    if escape:
                        escape = False
                    elif char == 92:  # backslash
                        escape = True
                    elif char == 34:  # quote
                        in_string = False
                elif char == 34:
                    in_string = True
                elif char == 123:
                    depth += 1
                elif char == 125:
                    depth -= 1
                    if depth == 0:
                        yield json.loads(buffer)


def add(table: defaultdict[str, list[float]], name: str, duration_us: float) -> None:
    row = table[name]
    row[0] += 1
    row[1] += duration_us
    row[2] = max(row[2], duration_us)


def top_rows(table: dict[str, list[float]], limit: int) -> list[dict[str, float | str]]:
    rows = [
        {
            "name": name,
            "count": int(count),
            "total_us": total_us,
            "total_ms": total_us / 1000.0,
            "avg_us": total_us / count if count else 0.0,
            "max_us": max_us,
        }
        for name, (count, total_us, max_us) in table.items()
    ]
    rows.sort(key=lambda row: row["total_us"], reverse=True)
    return rows[:limit]


def aggregate(trace: Path, top: int) -> dict[str, Any]:
    cat_counts: Counter[str] = Counter()
    ph_counts: Counter[str] = Counter()
    by_category = defaultdict(lambda: [0, 0.0, 0.0])
    kernels = defaultdict(lambda: [0, 0.0, 0.0])
    runtime = defaultdict(lambda: [0, 0.0, 0.0])
    copies = defaultdict(lambda: [0, 0.0, 0.0])
    cpu_ops = defaultdict(lambda: [0, 0.0, 0.0])
    total_events = 0

    for event in event_objects(trace):
        total_events += 1
        name = str(event.get("name", ""))
        category = str(event.get("cat", ""))
        phase = str(event.get("ph", ""))
        duration_us = float(event.get("dur") or 0.0)
        lower_category = category.lower()
        lower_name = name.lower()

        cat_counts[category] += 1
        ph_counts[phase] += 1
        add(by_category, f"{category}::{name}", duration_us)
        if "kernel" in lower_category:
            add(kernels, name, duration_us)
        if "runtime" in lower_category or name.startswith(("cuda", "cu")):
            add(runtime, name, duration_us)
        if (
            "memcpy" in lower_category
            or "memset" in lower_category
            or "memcpy" in lower_name
            or "memset" in lower_name
        ):
            add(copies, name, duration_us)
        if lower_category == "cpu_op":
            add(cpu_ops, name, duration_us)

    return {
        "trace": str(trace),
        "total_events": total_events,
        "cat_counts": cat_counts.most_common(),
        "ph_counts": ph_counts.most_common(),
        "top_kernel": top_rows(kernels, top),
        "top_runtime": top_rows(runtime, top),
        "top_gpu_copy_memset": top_rows(copies, top),
        "top_cpu_op": top_rows(cpu_ops, top),
        "top_cat_name": top_rows(by_category, top),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top", type=int, default=80)
    args = parser.parse_args()

    result = aggregate(args.trace, args.top)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "total_events": result["total_events"],
                "cat_counts": result["cat_counts"][:12],
                "top_kernel": result["top_kernel"][:20],
                "top_runtime": result["top_runtime"][:20],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
