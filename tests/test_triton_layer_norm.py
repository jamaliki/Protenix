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
    triton_layer_norm_enabled,
    triton_layer_norm_enabled_for_shape,
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

    def test_auto_policy_defaults_to_low_precision_and_large_fp32_shapes(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(triton_layer_norm_enabled(torch.bfloat16))
            self.assertTrue(triton_layer_norm_enabled(torch.float16))
            # Shape-free policy remains conservative for FP32.  The runtime
            # helper below is what promotes the large diffusion-token shapes
            # that actually won the H100 gate.
            self.assertFalse(triton_layer_norm_enabled(torch.float32))
            self.assertFalse(
                triton_layer_norm_enabled_for_shape(torch.float32, 512, 384)
            )
            self.assertTrue(
                triton_layer_norm_enabled_for_shape(torch.float32, 8192, 384)
            )
            self.assertTrue(
                triton_layer_norm_enabled_for_shape(torch.float32, 16000, 768)
            )
            self.assertFalse(
                triton_layer_norm_enabled_for_shape(torch.float32, 16000, 128)
            )

        with mock.patch.dict(os.environ, {"PROTENIX_TRITON_LAYER_NORM": "1"}):
            self.assertTrue(triton_layer_norm_enabled(torch.float32))
            self.assertTrue(
                triton_layer_norm_enabled_for_shape(torch.float32, 512, 128)
            )

        with mock.patch.dict(os.environ, {"PROTENIX_TRITON_LAYER_NORM": "0"}):
            self.assertFalse(triton_layer_norm_enabled(torch.bfloat16))
            self.assertFalse(
                triton_layer_norm_enabled_for_shape(torch.float32, 16000, 768)
            )

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
                            for use_weight, use_bias in (
                                (True, True),
                                (True, False),
                                (False, True),
                                (False, False),
                            ):
                                weight_arg = weight if use_weight else None
                                bias_arg = bias if use_bias else None
                                actual = triton_layer_norm(
                                    x,
                                    torch.Size([shape[-1]]),
                                    weight_arg,
                                    bias_arg,
                                    1e-5,
                                )
                                expected = F.layer_norm(
                                    x,
                                    torch.Size([shape[-1]]),
                                    None
                                    if weight_arg is None
                                    else weight_arg.to(dtype=dtype),
                                    None if bias_arg is None else bias_arg.to(dtype=dtype),
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
    def test_fused_layer_norm_uses_triton_by_default_for_bf16_inference(self):
        torch.manual_seed(11)
        module = FusedLayerNorm(128).cuda()
        x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        with mock.patch("protenix.model.layer_norm.layer_norm.fast_layer_norm_cuda_v2", None):
            with mock.patch.dict(os.environ, {}, clear=True):
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
