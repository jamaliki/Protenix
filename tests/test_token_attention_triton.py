import os
from unittest import mock

import torch

from protenix.model.modules.token_attention_triton import (
    triton_token_attention,
    triton_token_attention_enabled,
)


def test_triton_token_attention_policy_and_cpu_fallback():
    with mock.patch.dict(os.environ, {}, clear=True):
        assert not triton_token_attention_enabled()
    with mock.patch.dict(os.environ, {"PROTENIX_TRITON_TOKEN_ATTENTION": "1"}):
        assert triton_token_attention_enabled()

    q = torch.randn(2, 16, 8, 24, dtype=torch.bfloat16)
    bias = torch.randn(2, 16, 8, 8, dtype=torch.bfloat16)
    with mock.patch.dict(
        os.environ, {"PROTENIX_TRITON_TOKEN_ATTENTION": "1"}
    ), torch.inference_mode():
        assert triton_token_attention(q, q, q, bias) is None
