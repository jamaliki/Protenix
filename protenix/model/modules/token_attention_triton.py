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

"""Opt-in Triton path for pairformer token attention with dense pair bias.

The post-atom H100 trace showed pairformer token attention falling back to
PyTorch's decomposed math SDPA for BF16 tensors shaped like
``[B, 16 heads, N_token, head_dim=24]`` with a dense additive pair bias.  Making
the bias physically contiguous does not fix this: cuDNN has no valid plan for
the head_dim-24 dense-bias case in the current Tokyo runtime.

This helper is intentionally a narrow experiment.  It reuses Protenix's
existing streaming Triton ``TriAttentionFunction`` with a dummy triangle axis
and pads the 24-channel head to 32 because Triton block pointers require a
power-of-two head block.  That adds copy overhead, so the path stays opt-in
until a full model gate proves the local kernel win survives integration.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

_FALSE_ENV_VALUES = {"0", "false", "off", "no"}
_FAILED_SIGNATURES: set[tuple] = set()
_SUPPORTED_DTYPES = {torch.bfloat16}
_HEADS = 16
_HEAD_DIM = 24
_PADDED_HEAD_DIM = 32
_DEFAULT_MIN_TOKENS = 160
_MAX_TOKENS = 256


def triton_token_attention_enabled() -> bool:
    value = os.getenv("PROTENIX_TRITON_TOKEN_ATTENTION", "0")
    return value.lower() not in _FALSE_ENV_VALUES


def _min_tokens() -> int:
    value = os.getenv(
        "PROTENIX_TRITON_TOKEN_ATTENTION_MIN_TOKENS", str(_DEFAULT_MIN_TOKENS)
    )
    try:
        return max(1, int(value))
    except ValueError:
        return _DEFAULT_MIN_TOKENS


def _signature(q: torch.Tensor, attn_bias: torch.Tensor) -> tuple:
    return (
        tuple(q.shape),
        tuple(q.stride()),
        tuple(attn_bias.shape),
        tuple(attn_bias.stride()),
        q.dtype,
        attn_bias.dtype,
        q.device.type,
        q.device.index,
    )


def _can_try_token_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: Optional[torch.Tensor],
) -> bool:
    if not triton_token_attention_enabled():
        return False
    if torch.is_grad_enabled() or attn_bias is None:
        return False
    if q.shape != k.shape or q.shape != v.shape:
        return False
    if q.dim() != 4 or attn_bias.dim() != 4:
        return False
    if q.shape[:3] != attn_bias.shape[:3] or attn_bias.shape[-1] != q.shape[-2]:
        return False
    if q.shape[1] != _HEADS or q.shape[-1] != _HEAD_DIM:
        return False
    # The current reuse path pads/copies D=24 heads to D=32 for the existing
    # TriAttention kernel.  That overhead is measurable: it lost at N=124 but
    # won at N=220 in the H100 hotspot screen.  Keep a conservative token gate
    # so opt-in end-to-end tests exercise only the shape range where the kernel
    # has evidence of being beneficial.
    if q.shape[-2] < _min_tokens() or q.shape[-2] > _MAX_TOKENS:
        return False
    if q.dtype not in _SUPPORTED_DTYPES or attn_bias.dtype != q.dtype:
        return False
    return q.is_cuda and k.is_cuda and v.is_cuda and attn_bias.is_cuda


def _pad_head_dim(x: torch.Tensor) -> torch.Tensor:
    # Attention._prep_qkv produces logical [B, H, N, D] views.  TriAttention
    # consumes [B, dummy_N, N, H, D].  The extra channels must be zero for q/k
    # so the padded dot product is mathematically identical to head_dim 24.
    x = x.transpose(1, 2).unsqueeze(1).contiguous()
    out = x.new_empty(*x.shape[:-1], _PADDED_HEAD_DIM)
    out[..., :_HEAD_DIM] = x
    out[..., _HEAD_DIM:] = 0
    return out


def triton_token_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Return attention output in ``[B, H, N, D]`` layout or ``None``.

    ``q`` is already scaled by ``1/sqrt(head_dim)`` in
    :meth:`protenix.model.modules.primitives.Attention._prep_qkv`, matching the
    SDPA call's explicit ``scale=1.0``.  ``TriAttentionFunction`` computes the
    same dense additive-bias softmax using an online streaming algorithm.
    """

    if not _can_try_token_attention(q, k, v, attn_bias):
        return None
    assert attn_bias is not None
    sig = _signature(q, attn_bias)
    if sig in _FAILED_SIGNATURES:
        return None

    try:
        from protenix.model.tri_attention.op import TriAttentionFunction

        bias1 = q.new_zeros(q.shape[0], 1, 1, 1, q.shape[-2])
        bias2 = attn_bias.unsqueeze(1).contiguous()
        out = TriAttentionFunction.apply(
            _pad_head_dim(q),
            _pad_head_dim(k),
            _pad_head_dim(v),
            bias1,
            bias2,
        )
    except Exception:
        # Compilation/autotune failures are shape/runtime properties.  Remember
        # the signature so one bad path does not repeatedly pay exception cost.
        _FAILED_SIGNATURES.add(sig)
        return None

    return out.squeeze(1).transpose(1, 2)[..., :_HEAD_DIM].contiguous()
