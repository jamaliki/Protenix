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

"""Triton diffusion token attention with compact sample-invariant pair bias.

The production low-sample diffusion path flattens records and samples before
token attention:

    q/k/v: [record * sample, head, token, head_dim]

That shape keeps PyTorch/cuDNN SDPA on its fast rank-4 schedule, but the pair
bias is sample-invariant.  Materializing it as
``[record * sample, head, token, token]`` repeats the same bias for every
sample lane and was the live-memory cliff in the Protenix-v2 MPS gate.

This kernel keeps the fast flat q/k/v ABI, but loads bias from compact
``[record, head, token, token]`` storage by mapping
``record = flat_batch // sample_count`` inside the attention program.  It is
still shape-specialized: one Triton program owns a query tile and the full key
axis, so it is meant for the short diffusion-token lengths used here rather
than arbitrary long-sequence attention.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional runtime dependency.
    triton = None
    tl = None


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


_SUPPORTED_DTYPES = {torch.bfloat16, torch.float16}
_MAX_TOKENS = 256
_MAX_HEAD_DIM = 64
_DEFAULT_BLOCK_M = 16
_DEFAULT_NUM_WARPS = 4


def _is_power_of_2(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _compact_bias_4d(bias: torch.Tensor) -> torch.Tensor | None:
    """Normalize accepted compact bias layouts to ``[record, head, query, key]``.

    Model code caches compact diffusion bias as ``[record, 1, head, N, N]`` so
    ordinary SDPA can broadcast the singleton sample axis if this Triton path is
    disabled.  The kernel itself only needs the squeezed rank-4 view.
    """

    if bias.dim() == 5:
        if bias.shape[1] != 1:
            return None
        return bias[:, 0]
    if bias.dim() == 4:
        return bias
    return None


if triton is not None and tl is not None:

    @triton.jit
    def _compact_bias_attention_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        bias_ptr,
        out_ptr,
        n_tokens: tl.constexpr,
        head_dim: tl.constexpr,
        sample_count: tl.constexpr,
        q_stride_b: tl.constexpr,
        q_stride_h: tl.constexpr,
        q_stride_n: tl.constexpr,
        q_stride_d: tl.constexpr,
        k_stride_b: tl.constexpr,
        k_stride_h: tl.constexpr,
        k_stride_n: tl.constexpr,
        k_stride_d: tl.constexpr,
        v_stride_b: tl.constexpr,
        v_stride_h: tl.constexpr,
        v_stride_n: tl.constexpr,
        v_stride_d: tl.constexpr,
        bias_stride_b: tl.constexpr,
        bias_stride_h: tl.constexpr,
        bias_stride_q: tl.constexpr,
        bias_stride_k: tl.constexpr,
        out_stride_b: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_n: tl.constexpr,
        out_stride_d: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_d: tl.constexpr,
        dot_input_precision: tl.constexpr,
    ):
        input_dtype = q_ptr.dtype.element_ty
        query_block = tl.program_id(0)
        head = tl.program_id(1)
        flat_batch = tl.program_id(2)
        record = flat_batch // sample_count

        offs_m = query_block * block_m + tl.arange(0, block_m)
        offs_n = tl.arange(0, block_n)
        offs_d = tl.arange(0, block_d)
        valid_m = offs_m < n_tokens
        valid_n = offs_n < n_tokens
        valid_d = offs_d < head_dim

        q_base = q_ptr + flat_batch * q_stride_b + head * q_stride_h
        k_base = k_ptr + flat_batch * k_stride_b + head * k_stride_h
        v_base = v_ptr + flat_batch * v_stride_b + head * v_stride_h
        bias_base = bias_ptr + record * bias_stride_b + head * bias_stride_h

        q = tl.load(
            q_base + offs_m[:, None] * q_stride_n + offs_d[None, :] * q_stride_d,
            mask=valid_m[:, None] & valid_d[None, :],
            other=0.0,
        )
        k = tl.load(
            k_base + offs_n[:, None] * k_stride_n + offs_d[None, :] * k_stride_d,
            mask=valid_n[:, None] & valid_d[None, :],
            other=0.0,
        )
        bias = tl.load(
            bias_base
            + offs_m[:, None] * bias_stride_q
            + offs_n[None, :] * bias_stride_k,
            mask=valid_m[:, None] & valid_n[None, :],
            other=-float("inf"),
        )

        scores = tl.dot(q, tl.trans(k), input_precision=dot_input_precision) + bias
        scores = tl.where(valid_n[None, :], scores, -float("inf"))
        scores = tl.where(valid_m[:, None], scores, 0.0)
        scores = scores - tl.max(scores, axis=1)[:, None]
        probs = tl.exp(scores)
        probs = probs / tl.sum(probs, axis=1)[:, None]

        v = tl.load(
            v_base + offs_n[:, None] * v_stride_n + offs_d[None, :] * v_stride_d,
            mask=valid_n[:, None] & valid_d[None, :],
            other=0.0,
        )
        out = tl.dot(
            probs.to(input_dtype),
            v,
            input_precision=dot_input_precision,
        )

        out_base = out_ptr + flat_batch * out_stride_b + head * out_stride_h
        tl.store(
            out_base
            + offs_m[:, None] * out_stride_n
            + offs_d[None, :] * out_stride_d,
            out,
            mask=valid_m[:, None] & valid_d[None, :],
        )


def triton_compact_bias_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    *,
    sample_count: int,
    block_m: int = _DEFAULT_BLOCK_M,
    block_n: int | None = None,
    num_warps: int = _DEFAULT_NUM_WARPS,
    dot_input_precision: str = "tf32",
) -> torch.Tensor | None:
    """Return attention output for flat q/k/v and compact record-level bias.

    ``q``, ``k``, and ``v`` are shaped
    ``[record * sample_count, head, token, head_dim]``.  ``bias`` is shaped
    either ``[record, head, token, token]`` or
    ``[record, 1, head, token, token]``.  The result has the same shape/layout
    as ``q``.

    Return ``None`` when the shape is outside this specialized kernel's
    contract.  Callers should then fall back to the normal SDPA path, expanding
    compact bias if necessary, so correctness does not depend on this kernel.
    """

    if triton is None or tl is None:
        return None
    if sample_count <= 1:
        return None
    if not (q.is_cuda and k.is_cuda and v.is_cuda and bias.is_cuda):
        return None
    if q.dim() != 4 or k.shape != q.shape or v.shape != q.shape:
        return None
    bias = _compact_bias_4d(bias)
    if bias is None:
        return None
    if q.dtype not in _SUPPORTED_DTYPES:
        return None

    flat_batch, heads, n_tokens, head_dim = q.shape
    if n_tokens <= 0 or head_dim <= 0:
        return None
    if flat_batch % sample_count != 0:
        return None
    if bias.shape != (flat_batch // sample_count, heads, n_tokens, n_tokens):
        return None
    if head_dim > _MAX_HEAD_DIM or n_tokens > _MAX_TOKENS:
        return None

    block_n = _next_power_of_2(n_tokens) if block_n is None else block_n
    if block_m <= 0 or not _is_power_of_2(block_n) or block_n < n_tokens:
        return None
    block_d = _next_power_of_2(head_dim)
    out = torch.empty_like(q)
    grid = (triton.cdiv(n_tokens, block_m), heads, flat_batch)
    _compact_bias_attention_kernel[grid](
        q,
        k,
        v,
        bias,
        out,
        n_tokens,
        head_dim,
        sample_count,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        bias.stride(0),
        bias.stride(1),
        bias.stride(2),
        bias.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        block_m,
        block_n,
        block_d,
        dot_input_precision,
        num_warps=num_warps,
    )
    return out
