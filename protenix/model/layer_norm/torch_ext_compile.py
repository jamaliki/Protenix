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


import os
import re
import shutil
import subprocess
from typing import Any, Optional

from torch.utils.cpp_extension import load


def _cuda_arch_to_compute(arch: str) -> Optional[str]:
    arch = arch.strip().lower().removesuffix("+ptx")
    for prefix in ("sm_", "compute_"):
        if arch.startswith(prefix):
            arch = arch[len(prefix) :]
            break
    if "." in arch:
        major, minor = arch.split(".", 1)
        arch = f"{major}{minor}"
    return arch if arch.isdigit() else None


def _env_arch_list() -> Optional[list[str]]:
    raw_arches = os.getenv("PROTENIX_CUDA_ARCH_LIST") or os.getenv(
        "TORCH_CUDA_ARCH_LIST"
    )
    if not raw_arches:
        return None

    arches = []
    for token in re.split(r"[\s,;]+", raw_arches):
        arch = _cuda_arch_to_compute(token)
        if arch:
            arches.append(arch)
    return arches or None


def _supported_nvcc_arches() -> set[str]:
    from torch.utils.cpp_extension import CUDA_HOME

    nvcc = shutil.which("nvcc")
    if CUDA_HOME:
        candidate = os.path.join(CUDA_HOME, "bin", "nvcc")
        if os.path.isfile(candidate):
            nvcc = candidate

    try:
        out = subprocess.check_output(
            [nvcc, "--list-gpu-arch"], text=True, stderr=subprocess.STDOUT
        )
        return set(re.findall(r"compute_(\d+)", out))
    except Exception:
        return {"70", "80", "86", "90"}


def _selected_arches(supported: set[str]) -> list[str]:
    # Keep Blackwell opt-in for now. Some CUDA 13.0 toolchain combinations report
    # sm_100 support but fail while compiling the generated extension stubs, and
    # compiling unused architectures also slows first-run inference startup.
    default_arches = ["70", "80", "86", "89", "90"]
    requested_arches = _env_arch_list() or default_arches
    arches = [arch for arch in requested_arches if arch in supported]
    return arches or ["80"]


def _gencode_flags(arches: list[str]) -> list[str]:
    flags = []
    for arch in arches:
        flags += ["-gencode", f"arch=compute_{arch},code=sm_{arch}"]
    return flags


def _torch_arch_list(arches: list[str]) -> str:
    return ";".join(f"{int(arch) // 10}.{int(arch) % 10}" for arch in arches)


def compile(
    name: str,
    sources: list[str],
    extra_include_paths: list[str],
    build_directory: Optional[str] = None,
) -> Any:
    arches = _selected_arches(_supported_nvcc_arches())
    os.environ["TORCH_CUDA_ARCH_LIST"] = _torch_arch_list(arches)

    return load(
        name=name,
        sources=sources,
        extra_include_paths=extra_include_paths,
        extra_cflags=[
            "-O3",
            "-DVERSION_GE_1_1",
            "-DVERSION_GE_1_3",
            "-DVERSION_GE_1_5",
        ],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-DVERSION_GE_1_1",
            "-DVERSION_GE_1_3",
            "-DVERSION_GE_1_5",
            "-std=c++17",
            "-maxrregcount=32",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ]
        + _gencode_flags(arches),
        verbose=True,
        build_directory=build_directory,
    )
