#!/usr/bin/env python3
"""Experimental CUTLASS SM90 candidate for the transition output epilogue.

Candidate ABI:

    transition_output_epilogue(b, weight, gate, residual) -> out

The hotspot target is ``sigmoid(gate) * (b @ weight.T) + residual`` with
``b=[samples,tokens,hidden]`` and ``gate=[1,tokens,c_a]``.  We treat tokens as
the CUTLASS batch dimension ``L`` so the broadcasted gate is affine in the
epilogue: ``gate_stride_m=0``, ``gate_stride_n=1``, ``gate_stride_l=c_a``.

This file is intentionally a benchmark candidate, not production model code.
It compiles only on the Tokyo CUDA/CUTLASS environment used by
``tokyo_transition_epilogue_hotspot.sbatch``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_EXT = None
_WEIGHT_CACHE_KEY = None
_WEIGHT_CACHE = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _header_include_paths(cutlass_include: str) -> list[str]:
    """Return include roots needed by the out-of-tree CUDA extension.

    PyTorch's extension builder supplies the PyTorch headers, but not every
    cluster environment exposes CUDA runtime headers through ``CUDA_HOME/include``.
    On Tokyo, the active Python environment provides CUDA 13 headers through the
    ``nvidia-cu13`` wheel, while ``CUDA_HOME`` mainly provides ``nvcc``.  Passing
    both locations keeps the benchmark candidate reproducible without hardcoding
    one user's exact Python minor version.
    """

    paths = [Path(cutlass_include)]
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        root = Path(cuda_home)
        paths.extend([root / "include", root / "targets/x86_64-linux/include"])

    for root in [Path(sys.prefix), Path(torch.__file__).resolve().parents[1]]:
        paths.extend(root.glob("lib/python*/site-packages/nvidia/cu*/include"))

    include_paths: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if path.is_dir():
            resolved = str(path.resolve())
            if resolved not in seen:
                include_paths.append(resolved)
                seen.add(resolved)
    return include_paths


def _extension():
    global _EXT
    if _EXT is not None:
        return _EXT

    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch_list is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    elif "9.0a" not in arch_list and "90a" not in arch_list:
        raise RuntimeError(
            "CUTLASS GMMA kernels require a Hopper arch-specific build; "
            "set TORCH_CUDA_ARCH_LIST=9.0a"
        )

    cutlass_include = os.environ.get("CUTLASS_INCLUDE_DIR")
    if not cutlass_include:
        raise RuntimeError("CUTLASS_INCLUDE_DIR must point at CUTLASS/CuTe headers")

    source = _repo_root() / "scripts/perf/transition_output_epilogue_cutlass_sm90.cu"
    _EXT = load(
        name="protenix_transition_epilogue_cutlass_sm90",
        sources=[str(source)],
        extra_include_paths=_header_include_paths(cutlass_include),
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "-DCUTLASS_ARCH_MMA_SM90_SUPPORTED",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        with_cuda=True,
        verbose=True,
    )
    return _EXT


def _bf16_weight(weight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    global _WEIGHT_CACHE_KEY, _WEIGHT_CACHE
    key = (weight.data_ptr(), weight._version, dtype)
    if key != _WEIGHT_CACHE_KEY:
        _WEIGHT_CACHE = weight.to(dtype=dtype).contiguous()
        _WEIGHT_CACHE_KEY = key
    return _WEIGHT_CACHE


def transition_output_epilogue(
    b: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    if b.dtype is not torch.bfloat16:
        raise TypeError(f"CUTLASS candidate expects BF16 b, got {b.dtype}")
    if gate.dtype is not torch.bfloat16 or residual.dtype is not torch.bfloat16:
        raise TypeError("CUTLASS candidate expects BF16 gate and residual")
    if gate.shape[0] != 1:
        raise ValueError("CUTLASS candidate currently supports broadcast gate_batch=1")

    ext = _extension()
    weight_bf16 = _bf16_weight(weight, b.dtype)
    return ext.forward(b.contiguous(), weight_bf16, gate.contiguous(), residual.contiguous())
