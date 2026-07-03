import os

import torch

from protenix.model.modules.local_attention_triton import (
    _flatten_bias_for_sample_axis,
    triton_local_attention_bf16_output_enabled,
    triton_local_attention_enabled,
    triton_local_attention_gate_fusion_enabled,
)
from protenix.model.modules.local_attention_bias_triton import (
    triton_fused_local_attention_bias_enabled,
)

os.environ.setdefault("LAYERNORM_TYPE", "torch")


def test_triton_local_attention_defaults_to_guarded_fast_path(monkeypatch):
    monkeypatch.delenv("PROTENIX_TRITON_LOCAL_ATTN", raising=False)
    monkeypatch.delenv("PROTENIX_TRITON_LOCAL_ATTN_OUTPUT_BF16", raising=False)
    monkeypatch.delenv("PROTENIX_TRITON_LOCAL_ATTN_FUSE_GATE", raising=False)
    monkeypatch.delenv("PROTENIX_TRITON_FUSED_LOCAL_ATTN_BIAS", raising=False)

    assert triton_local_attention_enabled()
    assert triton_local_attention_bf16_output_enabled()
    assert triton_fused_local_attention_bias_enabled()
    assert not triton_local_attention_gate_fusion_enabled()


def test_triton_local_attention_opt_out_also_disables_bias_fusion(monkeypatch):
    monkeypatch.setenv("PROTENIX_TRITON_LOCAL_ATTN", "0")
    monkeypatch.delenv("PROTENIX_TRITON_LOCAL_ATTN_OUTPUT_BF16", raising=False)
    monkeypatch.delenv("PROTENIX_TRITON_FUSED_LOCAL_ATTN_BIAS", raising=False)

    assert not triton_local_attention_enabled()
    assert not triton_fused_local_attention_bias_enabled()
    # The BF16 store policy is independent; it only matters if the guarded
    # local-attention kernel actually runs.
    assert triton_local_attention_bf16_output_enabled()


def test_fused_local_attention_bias_can_be_overridden(monkeypatch):
    monkeypatch.setenv("PROTENIX_TRITON_LOCAL_ATTN", "0")
    monkeypatch.setenv("PROTENIX_TRITON_FUSED_LOCAL_ATTN_BIAS", "1")

    assert triton_fused_local_attention_bias_enabled()


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


def test_cached_atom_encoder_handles_record_and_sample_axes():
    from protenix.model.modules.transformer import AtomAttentionEncoder

    n_records = 3
    n_sample = 5
    n_atom = 64
    n_token = 8
    c_atom = 8
    c_token = 8
    c_atompair = 4
    c_s = 6
    c_z = 4
    n_queries = 32
    n_keys = 128
    n_trunks = (n_atom + n_queries - 1) // n_queries

    encoder = AtomAttentionEncoder(
        has_coords=True,
        c_token=c_token,
        c_atom=c_atom,
        c_atompair=c_atompair,
        c_s=c_s,
        c_z=c_z,
        n_blocks=1,
        n_heads=2,
        n_queries=n_queries,
        n_keys=n_keys,
    )
    encoder.eval()

    atom_to_token_idx = (
        torch.arange(n_atom)[None, :].expand(n_records, n_atom) % n_token
    ).long()
    ref_pos = torch.zeros(n_records, n_atom, 3)
    ref_charge = torch.zeros(n_records, n_atom)
    ref_mask = torch.ones(n_records, n_atom)
    ref_atom_name_chars = torch.zeros(n_records, n_atom, 4, 64)
    ref_element = torch.zeros(n_records, n_atom, 128)
    d_lm = torch.zeros(n_records, n_trunks, n_queries, n_keys, 3)
    v_lm = torch.ones(n_records, n_trunks, n_queries, n_keys, 1)
    pad_info = {
        "mask_trunked": torch.ones(n_records, n_trunks, n_queries, n_keys)
    }
    r_l = torch.randn(n_records, n_sample, n_atom, 3)
    s = torch.randn(n_records, 1, n_token, c_s)
    z = torch.randn(n_records, 1, n_token, n_token, c_z)
    p_lm = torch.randn(n_records, 1, n_trunks, n_queries, n_keys, c_atompair)
    c_l = torch.randn(n_records, n_atom, c_atom)
    atom_mask = torch.ones(n_records, n_atom, dtype=torch.bool)

    with torch.inference_mode():
        a, q_skip, c_skip, p_skip = encoder(
            atom_to_token_idx=atom_to_token_idx,
            ref_pos=ref_pos,
            ref_charge=ref_charge,
            ref_mask=ref_mask,
            ref_atom_name_chars=ref_atom_name_chars,
            ref_element=ref_element,
            d_lm=d_lm,
            v_lm=v_lm,
            pad_info=pad_info,
            r_l=r_l,
            s=s,
            z=z,
            p_lm=p_lm,
            c_l=c_l,
            atom_mask=atom_mask,
        )

    assert a.shape == (n_records, n_sample, n_token, c_token)
    assert q_skip.shape == (n_records, n_sample, n_atom, c_atom)
    assert c_skip.shape == (n_records, 1, n_atom, c_atom)
    assert p_skip.shape == (
        n_records,
        1,
        n_trunks,
        n_queries,
        n_keys,
        c_atompair,
    )
    assert torch.isfinite(a).all()
