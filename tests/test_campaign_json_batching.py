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

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from runner.inference import (
    _effective_batch_mode,
    _input_batch_signature,
    _pad_token_trunk_tree,
    _run_prediction_batch,
)
from runner.campaign_inputs import (
    group_inference_jsons_by_seed,
    load_inference_records,
    resolve_inference_jsons,
    write_campaign_json,
)


def _write_json(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f)


class TestCampaignJsonBatching(unittest.TestCase):
    def test_directory_jsons_are_sorted_and_filtered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = os.path.join(tmpdir, "b.json")
            second = os.path.join(tmpdir, "a.json")
            source = os.path.join(tmpdir, "source.json")
            generated_msa = os.path.join(tmpdir, "source-update-msa.json")
            generated_template = os.path.join(tmpdir, "source-final-updated.json")
            standalone_generated = os.path.join(tmpdir, "only-update-msa.json")
            ignored = os.path.join(tmpdir, "notes.txt")
            _write_json(first, [{"name": "b", "sequences": []}])
            _write_json(second, [{"name": "a", "sequences": []}])
            _write_json(source, [{"name": "source", "sequences": []}])
            _write_json(generated_msa, [{"name": "source-msa", "sequences": []}])
            _write_json(
                generated_template,
                [{"name": "source-template", "sequences": []}],
            )
            _write_json(
                standalone_generated,
                [{"name": "standalone", "sequences": []}],
            )
            with open(ignored, "w", encoding="utf-8") as f:
                f.write("not json")

            self.assertEqual(
                resolve_inference_jsons(tmpdir),
                [second, first, standalone_generated, source],
            )

    def test_seed_grouping_preserves_json_seed_policy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = os.path.join(tmpdir, "first.json")
            second = os.path.join(tmpdir, "second.json")
            third = os.path.join(tmpdir, "third.json")
            _write_json(first, [{"name": "first", "modelSeeds": [1, 2]}])
            _write_json(second, [{"name": "second", "modelSeeds": [3]}])
            _write_json(third, [{"name": "third"}])

            grouped = group_inference_jsons_by_seed(
                [first, second, third],
                default_seeds=[101],
                use_seeds_in_json=True,
            )

            self.assertEqual(grouped[(1, 2)], [first])
            self.assertEqual(grouped[(3,)], [second])
            self.assertEqual(grouped[(101,)], [third])

    def test_campaign_json_merges_records_for_one_inference_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = os.path.join(tmpdir, "first.json")
            second = os.path.join(tmpdir, "second.json")
            _write_json(first, [{"name": "first", "sequences": []}])
            _write_json(second, [{"name": "second", "sequences": []}])

            merged_path, cleanup_path = write_campaign_json([first, second], tmpdir)
            try:
                records = load_inference_records(merged_path)
                self.assertEqual(
                    [record["name"] for record in records], ["first", "second"]
                )
                self.assertEqual(cleanup_path, merged_path)
            finally:
                if cleanup_path is not None:
                    os.remove(cleanup_path)

    def test_single_json_does_not_create_transient_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "one.json")
            _write_json(path, [{"name": "one", "sequences": []}])

            merged_path, cleanup_path = write_campaign_json([path], tmpdir)

            self.assertEqual(merged_path, path)
            self.assertIsNone(cleanup_path)

    def test_token_batch_signature_ignores_atom_only_shapes(self):
        def make_data(n_atom: int) -> dict:
            n_token = 4
            return {
                "N_token": torch.tensor([n_token]),
                "input_feature_dict": {
                    "residue_index": torch.zeros(n_token, dtype=torch.long),
                    "token_index": torch.zeros(n_token, dtype=torch.long),
                    "asym_id": torch.zeros(n_token, dtype=torch.long),
                    "token_bonds": torch.zeros(n_token, n_token),
                    "msa": torch.zeros(1, n_token, dtype=torch.long),
                    "has_deletion": torch.zeros(1, n_token),
                    "deletion_value": torch.zeros(1, n_token),
                    "ref_pos": torch.zeros(n_atom, 3),
                    "ref_mask": torch.ones(n_atom),
                    "atom_to_token_idx": torch.zeros(n_atom, dtype=torch.long),
                    "bond_mask": torch.zeros(n_atom, n_atom),
                }
            }

        short_atom = make_data(20)
        long_atom = make_data(23)

        self.assertNotEqual(
            _input_batch_signature(short_atom, "exact"),
            _input_batch_signature(long_atom, "exact"),
        )
        self.assertEqual(
            _input_batch_signature(short_atom, "token"),
            _input_batch_signature(long_atom, "token"),
        )
        self.assertEqual(
            _input_batch_signature(short_atom, "auto"),
            _input_batch_signature(long_atom, "auto"),
        )

    def test_auto_mode_prefers_full_batch_when_exact_shapes_match(self):
        def make_data(n_atom: int) -> dict:
            n_token = 4
            return {
                "N_token": torch.tensor([n_token]),
                "input_feature_dict": {
                    "residue_index": torch.zeros(n_token, dtype=torch.long),
                    "token_bonds": torch.zeros(n_token, n_token),
                    "ref_pos": torch.zeros(n_atom, 3),
                    "atom_to_token_idx": torch.zeros(n_atom, dtype=torch.long),
                    "bond_mask": torch.zeros(n_atom, n_atom),
                }
            }

        same_a = make_data(20)
        same_b = make_data(20)
        ragged = make_data(23)

        self.assertEqual(
            _effective_batch_mode([(same_a, None), (same_b, None)], "auto"),
            "exact",
        )
        self.assertEqual(
            _effective_batch_mode([(same_a, None), (ragged, None)], "auto"),
            "token",
        )

    def test_auto_mode_uses_padded_trunk_for_different_token_counts(self):
        def make_data(n_token: int, n_atom: int) -> dict:
            return {
                "N_token": torch.tensor([n_token]),
                "input_feature_dict": {
                    "residue_index": torch.zeros(n_token, dtype=torch.long),
                    "token_index": torch.zeros(n_token, dtype=torch.long),
                    "token_bonds": torch.zeros(n_token, n_token),
                    "msa": torch.zeros(1, n_token, dtype=torch.long),
                    "restype": torch.zeros(n_token, 32),
                    "ref_pos": torch.zeros(n_atom, 3),
                    "atom_to_token_idx": torch.zeros(n_atom, dtype=torch.long),
                },
            }

        short_token = make_data(32, 200)
        long_token = make_data(40, 250)

        self.assertEqual(
            _input_batch_signature(short_token, "auto"),
            _input_batch_signature(long_token, "auto"),
        )
        self.assertEqual(
            _effective_batch_mode([(short_token, None), (long_token, None)], "auto"),
            "padded",
        )

    def test_token_padding_does_not_pad_restype_class_axis(self):
        n_token = 32
        feature_tree = {
            "restype": torch.zeros(n_token, 32),
            "token_bonds": torch.zeros(n_token, n_token),
        }

        padded = _pad_token_trunk_tree(feature_tree, n_token=n_token, max_tokens=40)

        self.assertEqual(tuple(padded["restype"].shape), (40, 32))
        self.assertEqual(tuple(padded["token_bonds"].shape), (40, 40))

    def test_singleton_prediction_batch_keeps_exact_path(self):
        class DummyRunner:
            def __init__(self):
                self.configs_seen = []
                self.predicted = None

            def update_model_configs(self, configs):
                self.configs_seen.append(configs)

            def predict(self, data):
                self.predicted = data
                return {"ok": True}

            def predict_token_batch(self, data_items):
                raise AssertionError("singleton exact path should not use trunk batch")

        data = {
            "N_token": torch.tensor([4]),
            "input_feature_dict": {
                "residue_index": torch.zeros(4, dtype=torch.long),
                "ref_pos": torch.zeros(20, 3),
            },
        }
        configs = SimpleNamespace(
            model_name="protenix_base_default_v1.0.0",
            skip_amp=SimpleNamespace(),
        )
        runner = DummyRunner()

        self.assertEqual(
            _run_prediction_batch(runner, configs, [(data, None)], "auto"),
            {"ok": True},
        )
        self.assertIs(runner.predicted, data)
        self.assertEqual(len(runner.configs_seen), 1)


if __name__ == "__main__":
    unittest.main()
