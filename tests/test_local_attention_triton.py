import os

import torch

from protenix.model.modules.local_attention_triton import (
    _flatten_bias_for_sample_axis,
)

os.environ.setdefault("LAYERNORM_TYPE", "torch")


def test_flatten_bias_reuses_record_bias_over_diffusion_samples():
    # Atom local attention sees activations as [record, sample, ...], but the
    # cached pair bias is invariant across diffusion samples.  The Triton kernel
    # should consume the compact [record, 1, ...] bias directly instead of
    # materializing [record, sample, ...] or falling back to the FP32 PyTorch path.
    tail_shape = (4, 2, 32, 128)
    bias = torch.empty(3, 1, *tail_shape)

    result = _flatten_bias_for_sample_axis(
        trunked_attn_bias=bias,
        batch_shape=torch.Size((3, 5)),
        tail_shape=tail_shape,
    )

    assert result is not None
    flat, repeat = result
    assert flat.shape == (3, *tail_shape)
    assert repeat == 5
    assert flat.data_ptr() == bias.data_ptr()


def test_flatten_bias_exact_batch_keeps_one_to_one_rows():
    tail_shape = (4, 2, 32, 128)
    bias = torch.empty(3, 5, *tail_shape)

    result = _flatten_bias_for_sample_axis(
        trunked_attn_bias=bias,
        batch_shape=torch.Size((3, 5)),
        tail_shape=tail_shape,
    )

    assert result is not None
    flat, repeat = result
    assert flat.shape == (15, *tail_shape)
    assert repeat == 1
    assert flat.data_ptr() == bias.data_ptr()


def test_flatten_bias_rejects_non_sample_axis_broadcast():
    tail_shape = (4, 2, 32, 128)
    bias = torch.empty(1, 5, *tail_shape)

    assert (
        _flatten_bias_for_sample_axis(
            trunked_attn_bias=bias,
            batch_shape=torch.Size((3, 5)),
            tail_shape=tail_shape,
        )
        is None
    )


def test_local_key_mask_keeps_sample_invariant_bias_prefix():
    from protenix.model.modules.transformer import AttentionPairBias

    n_records = 3
    n_sample = 5
    n_atom = 64
    n_heads = 2
    n_queries = 32
    n_keys = 128
    n_trunks = (n_atom + n_queries - 1) // n_queries

    module = AttentionPairBias(
        c_a=8,
        c_s=8,
        c_z=4,
        n_heads=n_heads,
        cross_attention_mode=True,
    )
    captured = {}

    def capture_attention(
        q_x,
        kv_x,
        attn_bias=None,
        trunked_attn_bias=None,
        n_queries=None,
        n_keys=None,
        inf=1e10,
        inplace_safe=False,
        chunk_size=None,
    ):
        captured["trunked_attn_bias_shape"] = tuple(trunked_attn_bias.shape)
        return q_x

    module.attention.forward = capture_attention
    q = torch.randn(n_records, n_sample, n_atom, 8)
    kv = torch.randn(n_records, 1, n_atom, 8)
    z = torch.randn(n_records, 1, n_trunks, n_queries, n_keys, 4)
    atom_mask = torch.ones(n_records, n_atom, dtype=torch.bool)

    out = module.local_multihead_attention(
        q=q,
        kv=kv,
        z=z,
        n_queries=n_queries,
        n_keys=n_keys,
        key_mask=atom_mask,
    )

    assert out.shape == q.shape
    assert captured["trunked_attn_bias_shape"] == (
        n_records,
        1,
        n_heads,
        n_trunks,
        n_queries,
        n_keys,
    )
