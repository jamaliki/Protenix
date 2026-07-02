# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import time
import unittest

import torch

os.environ["LAYERNORM_TYPE"] = "torch"
from protenix.model.modules.transformer import DiffusionTransformer


class TestDiffusionTransformer(unittest.TestCase):
    def setUp(self) -> None:
        self._start_time = time.time()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        super().setUp()

    def get_model(
        self,
        c_a: int = 128,
        c_s: int = 384,
        c_z: int = 64,
        n_blocks: int = 3,
        n_heads: int = 4,
    ):

        model = DiffusionTransformer(
            c_a=c_a, c_s=c_s, c_z=c_z, n_blocks=n_blocks, n_heads=n_heads
        ).to(self.device)

        return model

    def test_shape(self) -> None:

        n_heads = 2
        c_a = 13 * n_heads
        c_s = 23
        c_z = 17

        N = 45
        bs_dims = (2, 3)

        inputs = {
            "a": torch.rand(size=(*bs_dims, N, c_a)).to(self.device),
            "s": torch.rand(size=(*bs_dims, N, c_s)).to(self.device),
            "z": torch.rand(size=(*bs_dims, N, N, c_z)).to(self.device),
            "n_queries": None,
            "n_keys": None,
        }

        model = self.get_model(c_a=c_a, c_s=c_s, c_z=c_z, n_heads=n_heads)

        out = model(**inputs)
        target_shape = (*bs_dims, N, c_a)
        self.assertEqual(out.shape, out.reshape(target_shape).shape)

        N_q = 32
        N_k = 128
        N_blocks = math.ceil(N / N_q)

        inputs = {
            "a": torch.rand(size=(*bs_dims, N, c_a)).to(self.device),
            "s": torch.rand(size=(*bs_dims, N, c_s)).to(self.device),
            "z": torch.rand(size=(*bs_dims, N_blocks, N_q, N_k, c_z)).to(self.device),
            "n_queries": 32,
            "n_keys": 128,
        }

        out = model(**inputs)
        target_shape = (*bs_dims, N, c_a)
        self.assertEqual(out.shape, out.reshape(target_shape).shape)

    def test_sample_invariant_pair_bias_matches_repeated_z(self) -> None:
        torch.manual_seed(7)
        n_heads = 2
        c_a = 8
        c_s = 6
        c_z = 4
        n_records = 2
        n_sample = 3
        n_token = 5

        model = self.get_model(
            c_a=c_a, c_s=c_s, c_z=c_z, n_blocks=1, n_heads=n_heads
        )
        model.eval()

        a = torch.randn(n_records, n_sample, n_token, c_a, device=self.device)
        s = torch.randn(n_records, n_sample, n_token, c_s, device=self.device)
        z = torch.randn(n_records, n_token, n_token, c_z, device=self.device)
        token_mask = torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, False]],
            device=self.device,
        )
        a_flat = a.reshape(n_records * n_sample, n_token, c_a)
        s_flat = s.reshape(n_records * n_sample, n_token, c_s)
        token_mask_flat = (
            token_mask[:, None]
            .expand(n_records, n_sample, n_token)
            .reshape(n_records * n_sample, n_token)
        )

        with torch.no_grad():
            for efficient in (False, True):
                if efficient:
                    z_shared = z.permute(0, 3, 1, 2).contiguous()
                    z_repeated = (
                        z_shared[:, None]
                        .expand(n_records, n_sample, c_z, n_token, n_token)
                        .reshape(n_records * n_sample, c_z, n_token, n_token)
                    )
                else:
                    z_shared = z
                    z_repeated = (
                        z[:, None]
                        .expand(n_records, n_sample, n_token, n_token, c_z)
                        .reshape(n_records * n_sample, n_token, n_token, c_z)
                    )

                out_repeated = model(
                    a=a_flat,
                    s=s_flat,
                    z=z_repeated,
                    enable_efficient_fusion=efficient,
                    token_mask=token_mask_flat,
                )
                out_shared = model(
                    a=a_flat,
                    s=s_flat,
                    z=z_shared,
                    enable_efficient_fusion=efficient,
                    token_mask=token_mask_flat,
                    z_sample_count=n_sample,
                )

                torch.testing.assert_close(out_shared, out_repeated)

    def test_sample_axis_pair_bias_matches_repeated_z(self) -> None:
        torch.manual_seed(11)
        n_heads = 2
        c_a = 8
        c_s = 6
        c_z = 4
        n_records = 2
        n_sample = 3
        n_token = 5

        model = self.get_model(
            c_a=c_a, c_s=c_s, c_z=c_z, n_blocks=1, n_heads=n_heads
        )
        model.eval()

        a = torch.randn(n_records, n_sample, n_token, c_a, device=self.device)
        s = torch.randn(n_records, 1, n_token, c_s, device=self.device)
        z = torch.randn(n_records, n_token, n_token, c_z, device=self.device)
        token_mask = torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, False]],
            device=self.device,
        )
        a_flat = a.reshape(n_records * n_sample, n_token, c_a)
        s_flat = (
            s.expand(n_records, n_sample, n_token, c_s)
            .reshape(n_records * n_sample, n_token, c_s)
            .contiguous()
        )
        token_mask_flat = (
            token_mask[:, None]
            .expand(n_records, n_sample, n_token)
            .reshape(n_records * n_sample, n_token)
        )

        with torch.no_grad():
            for efficient in (False, True):
                if efficient:
                    z_shared = z.permute(0, 3, 1, 2).contiguous()
                    z_repeated = (
                        z_shared[:, None]
                        .expand(n_records, n_sample, c_z, n_token, n_token)
                        .reshape(n_records * n_sample, c_z, n_token, n_token)
                    )
                else:
                    z_shared = z
                    z_repeated = (
                        z[:, None]
                        .expand(n_records, n_sample, n_token, n_token, c_z)
                        .reshape(n_records * n_sample, n_token, n_token, c_z)
                    )

                out_repeated = model(
                    a=a_flat,
                    s=s_flat,
                    z=z_repeated,
                    enable_efficient_fusion=efficient,
                    token_mask=token_mask_flat,
                ).reshape(n_records, n_sample, n_token, c_a)
                out_sample_axis = model(
                    a=a,
                    s=s,
                    z=z_shared,
                    enable_efficient_fusion=efficient,
                    token_mask=token_mask[:, None],
                    z_sample_count=n_sample,
                    z_sample_axis=True,
                )

                torch.testing.assert_close(out_sample_axis, out_repeated)

    def tearDown(self):
        elapsed_time = time.time() - self._start_time
        print(f"Test {self.id()} took {elapsed_time:.6f}s")


if __name__ == "__main__":
    unittest.main()
