#!/usr/bin/env python3
"""Reference candidate for the transition-output epilogue harness.

This module intentionally does not optimize anything.  It implements the same
math as the baseline boundary behind the candidate ABI:

    transition_output_epilogue(b, weight, gate, residual) -> out

Use it to smoke-test ``CANDIDATE=...`` loading, positional argument order, dtype
behavior, broadcasted gates, and result JSON writing before swapping in a native
CuTe/CUTLASS extension candidate.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F


def _with_fused_elementwise_enabled() -> tuple[str | None, str]:
    old = os.environ.get("PROTENIX_TRITON_FUSED_ELEMENTWISE")
    os.environ["PROTENIX_TRITON_FUSED_ELEMENTWISE"] = "1"
    return old, "PROTENIX_TRITON_FUSED_ELEMENTWISE"


def _restore_env(old: str | None, name: str) -> None:
    if old is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = old


def transition_output_epilogue(
    b: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """Candidate ABI-compatible baseline implementation."""
    from protenix.model.modules.fused_elementwise_triton import fused_sigmoid_mul_add

    projected = F.linear(b, weight)
    old, name = _with_fused_elementwise_enabled()
    try:
        return fused_sigmoid_mul_add(gate.contiguous(), projected, residual)
    finally:
        _restore_env(old, name)
