#!/usr/bin/env python3
"""Summarize paired Protenix batch-inference gate logs.

The performance gates in ``docs/perf/inference_throughput.md`` run several
``CASE_START`` / ``CASE_DONE`` blocks inside one Slurm allocation.  The shell
markers go to stdout while Python logging goes to stderr, so manual analysis is
easy to get wrong.  This parser aligns both streams by timestamp and sums the
runner's structured ``Batch model timing total`` lines per case.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any


CASE_START_RE = re.compile(
    r"CASE_START label=(?P<label>\S+).*date_utc=(?P<date>\S+)"
)
CASE_DONE_RE = re.compile(r"CASE_DONE label=(?P<label>\S+).*date_utc=(?P<date>\S+)")
TIMING_RE = re.compile(
    r"^(?P<stamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .*"
    r"Batch model timing total "
    r"\((?P<source>[^,]+), (?P<items>\d+) input\(s\), "
    r"predict (?P<predict>[0-9.]+)s\): (?P<fields>.*)$"
)
FIELD_RE = re.compile(r"(?P<key>[A-Za-z0-9_]+)=(?P<seconds>[0-9.]+)s")


def _parse_iso_utc(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _parse_log_stamp(value: str) -> dt.datetime:
    # Protenix logging timestamps in these Slurm logs are emitted in UTC but do
    # not carry an offset.  Treat them as UTC so they can be aligned with the
    # shell's explicit ``date -u --iso-8601=seconds`` markers.
    return dt.datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=dt.timezone.utc
    )


def _read_lines(paths: list[Path]) -> list[str]:
    lines: list[str] = []
    for path in paths:
        if path.is_file():
            lines.extend(path.read_text(errors="replace").splitlines())
        elif path.is_dir():
            for child in sorted((path / "slurm_logs").glob("*")):
                if child.is_file():
                    lines.extend(child.read_text(errors="replace").splitlines())
            for child in sorted(path.glob("*")):
                if child.is_file():
                    lines.extend(child.read_text(errors="replace").splitlines())
        else:
            raise FileNotFoundError(path)
    return lines


def _parse_cases(lines: list[str]) -> list[dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for line in lines:
        if match := CASE_START_RE.search(line):
            label = match.group("label")
            cases[label] = {
                "label": label,
                "start": _parse_iso_utc(match.group("date")),
                "end": None,
                "batches": [],
            }
        elif match := CASE_DONE_RE.search(line):
            label = match.group("label")
            case = cases.setdefault(label, {"label": label, "batches": []})
            case["end"] = _parse_iso_utc(match.group("date"))

    ordered = sorted(
        cases.values(),
        key=lambda case: case.get("start")
        or dt.datetime.max.replace(tzinfo=dt.timezone.utc),
    )
    return ordered


def _case_for_time(
    cases: list[dict[str, Any]], when: dt.datetime
) -> dict[str, Any] | None:
    for case in cases:
        start = case.get("start")
        end = case.get("end")
        if start is None:
            continue
        if start <= when and (end is None or when <= end):
            return case
    return None


def _parse_timing_fields(fields: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for match in FIELD_RE.finditer(fields):
        values[match.group("key")] = float(match.group("seconds"))
    return values


def summarize(paths: list[Path], samples_per_input: int) -> list[dict[str, Any]]:
    lines = _read_lines(paths)
    cases = _parse_cases(lines)
    if not cases:
        raise RuntimeError("No CASE_START markers found")

    for line in lines:
        match = TIMING_RE.match(line)
        if not match:
            continue
        when = _parse_log_stamp(match.group("stamp"))
        case = _case_for_time(cases, when)
        if case is None:
            continue
        case["batches"].append(
            {
                "time": when.isoformat(),
                "source": match.group("source"),
                "items": int(match.group("items")),
                "predict_sec": float(match.group("predict")),
                "fields": _parse_timing_fields(match.group("fields")),
            }
        )

    summaries = []
    for case in cases:
        batches = case["batches"]
        fields: dict[str, float] = {}
        for batch in batches:
            for key, value in batch["fields"].items():
                fields[key] = fields.get(key, 0.0) + value
        items = sum(batch["items"] for batch in batches)
        predict_sec = sum(batch["predict_sec"] for batch in batches)
        generated = items * samples_per_input
        wall_sec = None
        if case.get("start") is not None and case.get("end") is not None:
            wall_sec = (case["end"] - case["start"]).total_seconds()
        summaries.append(
            {
                "label": case["label"],
                "batches": len(batches),
                "items": items,
                "samples_per_input": samples_per_input,
                "generated_samples": generated,
                "predict_sec": predict_sec,
                "inputs_per_sec": items / predict_sec if predict_sec else None,
                "samples_per_sec": generated / predict_sec if predict_sec else None,
                "wall_sec": wall_sec,
                "fields": fields,
                "sources": sorted({batch["source"] for batch in batches}),
            }
        )
    return summaries


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_markdown(summaries: list[dict[str, Any]]) -> None:
    columns = [
        "label",
        "batches",
        "items",
        "generated_samples",
        "predict_sec",
        "inputs_per_sec",
        "samples_per_sec",
        "pairformer",
        "diffusion",
        "diffusion_conditioning_sec",
        "diffusion_atom_encoder_sec",
        "diffusion_transformer_sec",
        "diffusion_atom_decoder_sec",
        "confidence",
    ]
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |")
    for summary in summaries:
        row = []
        for column in columns:
            if column in summary:
                row.append(_fmt(summary[column]))
            else:
                row.append(_fmt(summary["fields"].get(column)))
        print("| " + " | ".join(row) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Run dir or log files")
    parser.add_argument(
        "--samples-per-input",
        type=int,
        default=1,
        help="Generated diffusion samples per input record.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead")
    args = parser.parse_args()

    summaries = summarize(args.paths, samples_per_input=args.samples_per_input)
    if args.json:
        print(json.dumps(summaries, indent=2, default=str))
    else:
        print_markdown(summaries)


if __name__ == "__main__":
    main()
