# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure helpers for packing many inference JSONs into one campaign.

The model runner can batch records only after they are visible in the same
dataloader.  Protein-design campaigns often produce one JSON per design, so
running each file separately keeps the model loaded but prevents the exact-shape
batcher from ever filling a GPU batch.  These helpers merge records across JSON
files without touching the model inputs themselves; the existing tensor-tree
signature still decides which records are safe to stack.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

_GENERATED_JSON_SUFFIXES = ("-update-msa", "-final-updated")


def _has_source_json(path: Path, all_jsons: set[Path]) -> bool:
    """Return true when ``path`` is a generated preprocessing sibling.

    MSA/template preprocessing writes files such as ``foo-update-msa.json`` next
    to ``foo.json``.  Directory campaigns should not infer both, otherwise a
    rerun or paired benchmark silently duplicates work.  If the generated file
    is the only JSON present, keep it: users may intentionally point a directory
    at preprocessed inputs.
    """
    for suffix in _GENERATED_JSON_SUFFIXES:
        if path.stem.endswith(suffix):
            source = path.with_name(f"{path.stem[: -len(suffix)]}{path.suffix}")
            return source in all_jsons
    return False


def resolve_inference_jsons(json_file: str) -> list[str]:
    """Resolve a JSON file or directory into a deterministic list of JSON files."""
    path = Path(json_file)
    if path.is_dir():
        all_jsons = {file for file in path.rglob("*.json") if file.is_file()}
        jsons = sorted(
            str(file) for file in all_jsons if not _has_source_json(file, all_jsons)
        )
        if not jsons:
            raise RuntimeError(f"Can not read a valid json file in {json_file}")
        return jsons
    if path.is_file() and path.suffix == ".json":
        return [str(path)]
    if path.is_file():
        raise RuntimeError(f"Input file must be a JSON file, got: {json_file}")
    raise RuntimeError(f"Can not read a special file: {json_file}")


def load_inference_records(json_path: str) -> list[dict]:
    """Load one Protenix inference JSON and validate its top-level shape."""
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list) or not records:
        raise ValueError(
            f"Input JSON must be a non-empty top-level list, got "
            f"{type(records).__name__} from {json_path}"
        )
    if any(not isinstance(record, dict) for record in records):
        raise ValueError(f"Every input record in {json_path} must be a JSON object")
    return records


def seed_key_for_json(json_path: str, default_seeds: list[int]) -> tuple[int, ...]:
    records = load_inference_records(json_path)
    seed_in_json = records[0].get("modelSeeds")
    if seed_in_json:
        return tuple(int(seed) for seed in seed_in_json)
    return tuple(int(seed) for seed in default_seeds)


def group_inference_jsons_by_seed(
    json_paths: list[str],
    default_seeds: list[int],
    use_seeds_in_json: bool,
) -> dict[tuple[int, ...], list[str]]:
    """Group campaign JSONs so merged files preserve the existing seed policy.

    ``infer_predict`` historically reads ``modelSeeds`` from the first record in
    a JSON file.  When we merge many one-record files for cross-file batching,
    files with different first-record seeds must therefore stay in separate
    transient JSONs.  The common CLI-seed path forms one group.
    """
    groups: dict[tuple[int, ...], list[str]] = {}
    default_key = tuple(int(seed) for seed in default_seeds)
    for json_path in json_paths:
        key = (
            seed_key_for_json(json_path, default_seeds)
            if use_seeds_in_json
            else default_key
        )
        groups.setdefault(key, []).append(json_path)
    return groups


def write_campaign_json(
    json_paths: list[str],
    out_dir: str,
) -> tuple[str, Optional[str]]:
    """Merge JSON records for one campaign batch and return ``(path, cleanup)``.

    If there is only one source JSON, no file is written and cleanup is ``None``.
    For multiple files we write a short-lived JSON under the output directory so
    the existing dataloader and exact-shape batcher can run unchanged.
    """
    if len(json_paths) == 1:
        return json_paths[0], None

    records = []
    for json_path in json_paths:
        records.extend(load_inference_records(json_path))

    campaign_dir = os.path.join(out_dir, ".protenix_campaign_inputs")
    os.makedirs(campaign_dir, exist_ok=True)
    fd, merged_path = tempfile.mkstemp(
        prefix="campaign-", suffix=".json", dir=campaign_dir
    )
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    return merged_path, merged_path
