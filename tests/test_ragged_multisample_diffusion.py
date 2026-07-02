import os
from types import SimpleNamespace

import torch

from protenix.model.protenix import (
    _flatten_record_sample_axes,
    _flatten_record_sample_mask,
)
from runner.inference import _batched_token_diffusion_enabled


def _configs(n_sample: int, guidance: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        sample_diffusion=SimpleNamespace(
            N_sample=n_sample,
            guidance=SimpleNamespace(enable=guidance),
        )
    )


def _set_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


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


def test_low_sample_diffusion_batch_gate_stays_guarded():
    old_enabled = os.environ.get("PROTENIX_BATCH_DIFFUSION_TRANSFORMER")
    old_max_samples = os.environ.get("PROTENIX_BATCH_DIFFUSION_MAX_SAMPLES")
    try:
        _set_env("PROTENIX_BATCH_DIFFUSION_TRANSFORMER", None)
        _set_env("PROTENIX_BATCH_DIFFUSION_MAX_SAMPLES", None)

        assert _batched_token_diffusion_enabled(_configs(5), batch_size=2)
        assert not _batched_token_diffusion_enabled(_configs(6), batch_size=2)
        assert not _batched_token_diffusion_enabled(_configs(5), batch_size=1)
        assert not _batched_token_diffusion_enabled(
            _configs(5, guidance=True), batch_size=2
        )

        _set_env("PROTENIX_BATCH_DIFFUSION_MAX_SAMPLES", "6")
        assert _batched_token_diffusion_enabled(_configs(6), batch_size=2)

        _set_env("PROTENIX_BATCH_DIFFUSION_TRANSFORMER", "0")
        assert not _batched_token_diffusion_enabled(_configs(5), batch_size=2)
    finally:
        _set_env("PROTENIX_BATCH_DIFFUSION_TRANSFORMER", old_enabled)
        _set_env("PROTENIX_BATCH_DIFFUSION_MAX_SAMPLES", old_max_samples)
