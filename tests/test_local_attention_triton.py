import torch

from protenix.model.modules.local_attention_triton import (
    _flatten_bias_for_sample_axis,
)


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
