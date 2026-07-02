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
            ignored = os.path.join(tmpdir, "notes.txt")
            _write_json(first, [{"name": "b", "sequences": []}])
            _write_json(second, [{"name": "a", "sequences": []}])
            with open(ignored, "w", encoding="utf-8") as f:
                f.write("not json")

            self.assertEqual(resolve_inference_jsons(tmpdir), [second, first])

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


if __name__ == "__main__":
    unittest.main()
