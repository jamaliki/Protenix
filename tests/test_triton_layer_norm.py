import os
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

# These tests exercise the fallback path that Tokyo currently uses when the
# built-in CUDA extension cannot compile.  Avoid spending test time trying to
# build that extension before we can even import the Triton candidate.
os.environ.setdefault("PROTENIX_DISABLE_FAST_LAYER_NORM", "1")

from protenix.model.layer_norm.layer_norm import FusedLayerNorm
from protenix.model.layer_norm.layer_norm_triton import (
    triton_layer_norm,
    triton_layer_norm_available,
)


class TestTritonLayerNorm(unittest.TestCase):
    def test_disabled_path_returns_none(self):
        x = torch.randn(2, 4)
        with mock.patch.dict(os.environ, {"PROTENIX_TRITON_LAYER_NORM": "0"}):
            self.assertIsNone(triton_layer_norm(x, torch.Size([4]), None, None, 1e-5))

    def test_cpu_path_returns_none_when_enabled(self):
        x = torch.randn(2, 4)
        with mock.patch.dict(os.environ, {"PROTENIX_TRITON_LAYER_NORM": "1"}):
            self.assertIsNone(triton_layer_norm(x, torch.Size([4]), None, None, 1e-5))

    @unittest.skipUnless(
        torch.cuda.is_available() and triton_layer_norm_available(),
        "CUDA + Triton required",
    )
    def test_cuda_forward_matches_torch(self):
        torch.manual_seed(7)
        shapes = [(33, 128), (4, 17, 384), (2, 5, 13, 768)]
        dtypes = [torch.float32, torch.bfloat16]
        with mock.patch.dict(os.environ, {"PROTENIX_TRITON_LAYER_NORM": "1"}):
            for shape in shapes:
                for dtype in dtypes:
                    with self.subTest(shape=shape, dtype=dtype):
                        x = torch.randn(shape, device="cuda", dtype=dtype)
                        weight = torch.randn(shape[-1], device="cuda")
                        bias = torch.randn(shape[-1], device="cuda")
                        with torch.no_grad():
                            actual = triton_layer_norm(
                                x,
                                torch.Size([shape[-1]]),
                                weight,
                                bias,
                                1e-5,
                            )
                            expected = F.layer_norm(
                                x,
                                torch.Size([shape[-1]]),
                                weight.to(dtype=dtype),
                                bias.to(dtype=dtype),
                                1e-5,
                            )
                        self.assertIsNotNone(actual)
                        torch.testing.assert_close(
                            actual,
                            expected,
                            atol=3e-2 if dtype is torch.bfloat16 else 2e-5,
                            rtol=3e-2 if dtype is torch.bfloat16 else 2e-5,
                        )

    @unittest.skipUnless(
        torch.cuda.is_available() and triton_layer_norm_available(),
        "CUDA + Triton required",
    )
    def test_fused_layer_norm_uses_triton_in_inference(self):
        torch.manual_seed(11)
        module = FusedLayerNorm(128).cuda()
        x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        with mock.patch("protenix.model.layer_norm.layer_norm.fast_layer_norm_cuda_v2", None):
            env = {"PROTENIX_TRITON_LAYER_NORM": "1"}
            with mock.patch.dict(os.environ, env):
                with torch.no_grad():
                    actual = module(x)
                    expected = F.layer_norm(
                        x,
                        torch.Size([128]),
                        module.weight.to(dtype=x.dtype),
                        module.bias.to(dtype=x.dtype),
                        module.eps,
                    )
        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


if __name__ == "__main__":
    unittest.main()
