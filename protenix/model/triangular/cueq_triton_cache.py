"""Small Protenix overlay for CUEQ Triton autotune caches.

CUEQ already ships a large site-wide cache of tuned Triton tile choices.  The
Protenix-v2 B32/N251 pairformer shape exposed two missing H100 entries for the
fused sigmoid-gated GEMM used by triangle multiplication; ONDEMAND tuning found
better tiles, but paying that tuning cost on first inference is not acceptable.

Rather than replacing CUEQ's cache, this module merges a tiny Protenix overlay
into a user-writable cache directory and points CUEQ at that merged directory
before the CUEQ modules are imported.  If users have already set
``CUEQ_TRITON_CACHE_DIR`` we leave their choice alone.
"""

from __future__ import annotations

import importlib.resources
import importlib.util
import json
import os
from pathlib import Path
from typing import Any


_FALSE_ENV_VALUES = {"0", "false", "off", "no"}
_OVERLAY_DIR = "cueq_h100_cache"


def _env_enabled(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).lower() not in _FALSE_ENV_VALUES


def _user_cache_dir() -> Path:
    override = os.getenv("PROTENIX_CUEQ_TRITON_CACHE_DIR")
    if override:
        return Path(override)
    root = os.getenv("XDG_CACHE_HOME")
    if root:
        return Path(root) / "protenix" / "cueq-triton-cache"
    return Path.home() / ".cache" / "protenix" / "cueq-triton-cache"


def _cueq_site_cache_dir() -> Path | None:
    spec = importlib.util.find_spec("cuequivariance_ops")
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).parent / "triton" / "cache"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _merge_cache(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """Merge GPU-keyed CUEQ cache dictionaries in place."""
    for gpu_key, src_entries in src.items():
        if not isinstance(src_entries, dict):
            dst[gpu_key] = src_entries
            continue
        dst_entries = dst.setdefault(gpu_key, {})
        if isinstance(dst_entries, dict):
            dst_entries.update(src_entries)
        else:
            dst[gpu_key] = src_entries


def _write_json_if_changed(path: Path, data: dict[str, Any]) -> None:
    text = json.dumps(data, indent=4, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def configure_cueq_h100_triton_cache() -> None:
    """Install Protenix's H100 CUEQ tuning overlay unless the user opted out."""
    if not _env_enabled("PROTENIX_CUEQ_H100_TRITON_CACHE", "1"):
        return
    if os.getenv("CUEQ_TRITON_CACHE_DIR"):
        return

    site_cache = _cueq_site_cache_dir()
    if site_cache is None or not site_cache.exists():
        return

    try:
        overlay_root = importlib.resources.files(__package__).joinpath(_OVERLAY_DIR)
    except (AttributeError, ModuleNotFoundError):
        return

    cache_dir = _user_cache_dir()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        wrote_any = False
        for overlay in overlay_root.iterdir():
            if overlay.name.startswith(".") or not overlay.name.endswith(".json"):
                continue
            target = cache_dir / overlay.name
            merged = _load_json(site_cache / overlay.name)
            _merge_cache(merged, _load_json(target))
            _merge_cache(merged, json.loads(overlay.read_text(encoding="utf-8")))
            _write_json_if_changed(target, merged)
            wrote_any = True
    except OSError:
        return

    if wrote_any:
        os.environ["CUEQ_TRITON_CACHE_DIR"] = str(cache_dir)
