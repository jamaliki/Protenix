"""Make direct perf-script execution import this checkout.

Tokyo profiling jobs usually run commands such as
``python scripts/perf/trunk_attention_hotspot.py`` from the repository root.
For that launch style Python puts ``scripts/perf`` on ``sys.path``, not the
repo root.  If the environment also has an older editable Protenix install, a
benchmark can silently measure the wrong checkout.  Import this module before
project imports in perf scripts to keep measurements tied to the logged commit.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
