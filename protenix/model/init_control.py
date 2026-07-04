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

from collections.abc import Iterator
from contextlib import contextmanager


_SKIP_WEIGHT_INIT_DEPTH = 0


def skip_weight_init_enabled() -> bool:
    """Return True while a caller has proved initial weights will be overwritten.

    This is deliberately a process-local context flag rather than a general
    environment switch. Training and non-strict checkpoint loads must always run
    their normal initializers; strict inference is the safe case because
    `load_state_dict(strict=True)` fails if any parameter or buffer is missing.
    """

    return _SKIP_WEIGHT_INIT_DEPTH > 0


@contextmanager
def skip_weight_init() -> Iterator[None]:
    """Temporarily skip expensive module initializers during strict inference.

    Constructing Protenix-v2 normally fills many large Linear weights with
    SciPy truncated-normal samples. In inference those random values are
    immediately replaced by checkpoint tensors, so the fills are pure startup
    overhead when the subsequent checkpoint load is strict.
    """

    global _SKIP_WEIGHT_INIT_DEPTH
    _SKIP_WEIGHT_INIT_DEPTH += 1
    try:
        yield
    finally:
        _SKIP_WEIGHT_INIT_DEPTH -= 1
