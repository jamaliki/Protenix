import torch

from protenix.model.protenix import (
    _flatten_record_sample_axes,
    _flatten_record_sample_mask,
)


def test_flatten_record_sample_axes_broadcasts_singleton_sample_lane():
    # Diffusion conditioning is sample-invariant during sampling: each record
    # has one computed conditioning lane that should be reused for every sample.
    # Flattening must produce record-major rows:
    #   (record0,sample0..N), then (record1,sample0..N), ...
    tensor = torch.tensor([[[10.0], [11.0]], [[20.0], [21.0]]]).unsqueeze(1)

    flattened = _flatten_record_sample_axes(tensor, n_sample=3)

    assert flattened.shape == (6, 2, 1)
    assert torch.equal(flattened[:3], tensor[0].expand(3, 2, 1))
    assert torch.equal(flattened[3:], tensor[1].expand(3, 2, 1))


def test_flatten_record_sample_axes_preserves_existing_sample_lanes():
    tensor = torch.arange(2 * 3 * 4).reshape(2, 3, 4)

    flattened = _flatten_record_sample_axes(tensor, n_sample=3)

    assert flattened.shape == (6, 4)
    assert torch.equal(flattened, tensor.reshape(6, 4))


def test_flatten_record_sample_mask_matches_activation_order():
    mask = torch.tensor([[True, True, False], [True, False, False]])

    flattened = _flatten_record_sample_mask(mask, n_sample=2)

    expected = torch.tensor(
        [
            [True, True, False],
            [True, True, False],
            [True, False, False],
            [True, False, False],
        ]
    )
    assert torch.equal(flattened, expected)
