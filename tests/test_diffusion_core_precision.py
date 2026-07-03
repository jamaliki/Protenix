import torch

from protenix.model.modules.diffusion import (
    DiffusionModule,
    atom_attention_bf16_enabled,
    diffusion_core_bf16_enabled,
)


def test_diffusion_core_prefers_bf16_on_supported_cuda(monkeypatch):
    monkeypatch.delenv("PROTENIX_BF16_DIFFUSION_CORE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True, raising=False)

    assert diffusion_core_bf16_enabled()
    assert DiffusionModule._diffusion_core_dtype(None, torch.float32) is torch.bfloat16
    assert DiffusionModule._diffusion_core_dtype(None, torch.bfloat16) is torch.bfloat16
    assert DiffusionModule._diffusion_core_dtype(None, torch.float16) is torch.float16


def test_diffusion_core_keeps_conservative_opt_out(monkeypatch):
    monkeypatch.setenv("PROTENIX_BF16_DIFFUSION_CORE", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True, raising=False)

    assert not diffusion_core_bf16_enabled()
    assert DiffusionModule._diffusion_core_dtype(None, torch.float32) is torch.float32


def test_diffusion_core_default_requires_bf16_cuda(monkeypatch):
    monkeypatch.delenv("PROTENIX_BF16_DIFFUSION_CORE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False, raising=False)

    assert not diffusion_core_bf16_enabled()
    assert DiffusionModule._diffusion_core_dtype(None, torch.float32) is torch.float32


def test_atom_attention_prefers_bf16_on_supported_cuda(monkeypatch):
    monkeypatch.delenv("PROTENIX_BF16_ATOM_ATTENTION", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True, raising=False)

    assert atom_attention_bf16_enabled()


def test_atom_attention_keeps_conservative_opt_out(monkeypatch):
    monkeypatch.setenv("PROTENIX_BF16_ATOM_ATTENTION", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True, raising=False)

    assert not atom_attention_bf16_enabled()


def test_atom_attention_default_requires_bf16_cuda(monkeypatch):
    monkeypatch.delenv("PROTENIX_BF16_ATOM_ATTENTION", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False, raising=False)

    assert not atom_attention_bf16_enabled()
