import os
import unittest
from unittest import mock

import torch

os.environ.setdefault("LAYERNORM_TYPE", "torch")

from protenix.model.triangular.triangular import (  # noqa: E402
    TriangleAttention,
    ending_attention_norm_first_enabled,
)


class TestTriangleAttentionLayout(unittest.TestCase):
    def test_ending_norm_first_flag_defaults_on(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(ending_attention_norm_first_enabled())
        with mock.patch.dict(
            os.environ, {"PROTENIX_ENDING_ATTENTION_NORM_FIRST": "0"}
        ):
            self.assertFalse(ending_attention_norm_first_enabled())

    def test_layer_norm_commutes_with_pair_axis_transpose(self):
        torch.manual_seed(17)
        module = TriangleAttention(c_in=5, c_hidden=8, no_heads=2, starting=False)
        x = torch.randn(2, 3, 4, 5)

        norm_after_transpose = module.layer_norm(x.transpose(-2, -3))
        transpose_after_norm = module.layer_norm(x).transpose(-2, -3)

        torch.testing.assert_close(norm_after_transpose, transpose_after_norm)


if __name__ == "__main__":
    unittest.main()
