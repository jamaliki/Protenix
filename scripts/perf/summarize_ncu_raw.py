#!/usr/bin/env python3
"""Summarize Nsight Compute ``--csv --page raw`` exports.

NCU raw CSV files are extremely wide: each kernel launch has thousands of
metric columns.  For the optimization loop we usually need a smaller first
question: which launch families own wall time, and are they memory-, SM-, or
tensor-pipe-limited?  This script extracts the common duration/SOL columns and
prints both per-launch and family-aggregated Markdown tables.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path


TIME_COL = "gpu__time_duration.sum"
METRIC_COLS = {
    "dram %": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "mem %": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "SM %": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "tensor %": "TPC.TriageCompute.sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_active",
    "occ %": "sm__warps_active.avg.pct_of_peak_sustained_active",
}


@dataclass
class KernelFamily:
    name: str
    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0
    ids: list[str] = field(default_factory=list)

    def add(self, launch_id: str, elapsed_ms: float) -> None:
        self.count += 1
        self.total_ms += elapsed_ms
        self.max_ms = max(self.max_ms, elapsed_ms)
        if len(self.ids) < 8:
            self.ids.append(launch_id)


def parse_float(value: str | None) -> float:
    if value in (None, ""):
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def time_scale_to_ms(unit: str | None) -> float:
    """Return the multiplier that converts an NCU duration unit to ms.

    Nsight Compute writes units as the second CSV row, not in the column name.
    For example, ``gpu__time_duration.sum`` is usually reported in ``us`` for
    short kernels.  Keeping the output in milliseconds makes one-block NCU
    summaries comparable with CUDA-event hotspot timings and end-to-end logs.
    """

    normalized = (unit or "").strip().lower()
    return {
        "ns": 1e-6,
        "nsecond": 1e-6,
        "nseconds": 1e-6,
        "us": 1e-3,
        "usecond": 1e-3,
        "useconds": 1e-3,
        "ms": 1.0,
        "msecond": 1.0,
        "mseconds": 1.0,
        "s": 1e3,
        "second": 1e3,
        "seconds": 1e3,
    }.get(normalized, 1.0)


def normalize_kernel_name(name: str) -> str:
    """Collapse generated template details while preserving family identity."""
    name = re.sub(r"<.*", "<...>", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = re.sub(r"_[0-9]+(?=[^A-Za-z]|$)", "_N", name)
    return name[:110] if name else "<empty>"


def short_kernel_name(name: str, limit: int) -> str:
    name = re.sub(r"\s+", " ", name).strip()
    if len(name) <= limit:
        return name
    return name[: max(0, limit - 1)] + "..."


def markdown_row(values: list[str]) -> str:
    return "| " + " | ".join(values) + " |"


def print_launch_table(
    rows: list[dict[str, str]], *, top: int, name_width: int, time_ms_scale: float
) -> None:
    metric_labels = [label for label, col in METRIC_COLS.items() if col in rows[0]]
    headers = ["id", "kernel", "ms", *metric_labels]
    print(markdown_row(headers))
    print(markdown_row(["---", "---", "---:"] + ["---:"] * len(metric_labels)))
    for row in rows[:top]:
        elapsed = parse_float(row.get(TIME_COL)) * time_ms_scale
        if not math.isfinite(elapsed):
            continue
        values = [
            row.get("ID", ""),
            f"`{short_kernel_name(row.get('Kernel Name', ''), name_width)}`",
            f"{elapsed:.4f}",
        ]
        for label in metric_labels:
            value = parse_float(row.get(METRIC_COLS[label]))
            values.append("" if not math.isfinite(value) else f"{value:.1f}")
        print(markdown_row(values))


def print_family_table(
    rows: list[dict[str, str]], *, top: int, time_ms_scale: float
) -> None:
    families: dict[str, KernelFamily] = {}
    total_ms = 0.0
    for row in rows:
        elapsed = parse_float(row.get(TIME_COL)) * time_ms_scale
        name = row.get("Kernel Name", "")
        if not math.isfinite(elapsed) or not name:
            continue
        total_ms += elapsed
        key = normalize_kernel_name(name)
        families.setdefault(key, KernelFamily(key)).add(row.get("ID", ""), elapsed)

    ordered = sorted(families.values(), key=lambda family: family.total_ms, reverse=True)
    print(markdown_row(["kernel family", "launches", "total ms", "max ms", "ids"]))
    print(markdown_row(["---", "---:", "---:", "---:", "---"]))
    for family in ordered[:top]:
        print(
            markdown_row(
                [
                    f"`{family.name}`",
                    str(family.count),
                    f"{family.total_ms:.4f}",
                    f"{family.max_ms:.4f}",
                    ", ".join(family.ids),
                ]
            )
        )
    print(f"\nTotal captured kernel time: {total_ms:.4f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--name-width", type=int, default=96)
    args = parser.parse_args()

    with args.csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        units_row = next(reader, None)
        rows = list(reader)
    if not rows:
        raise SystemExit(f"no rows found in {args.csv_path}")
    if TIME_COL not in rows[0]:
        raise SystemExit(f"{TIME_COL!r} not found in {args.csv_path}")
    time_ms_scale = time_scale_to_ms(None if units_row is None else units_row.get(TIME_COL))

    rows = sorted(rows, key=lambda row: parse_float(row.get(TIME_COL)), reverse=True)
    rows = [row for row in rows if row.get("Kernel Name")]

    print("## Top launches\n")
    print_launch_table(
        rows, top=args.top, name_width=args.name_width, time_ms_scale=time_ms_scale
    )
    print("\n## Kernel families\n")
    print_family_table(rows, top=args.top, time_ms_scale=time_ms_scale)


if __name__ == "__main__":
    main()
