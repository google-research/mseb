# Copyright 2025 The MSEB Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from mseb.abstasks import meta_data


class TaskMetaDataTest(parameterized.TestCase):

  def _get_valid_params(self) -> dict[str, Any]:
    return {
        "name": "MyTestTask",
        "description": "A test task for validation.",
        "reference": "https://example.com/reference",
        "dataset": meta_data.Dataset(
            path="mteb/my-test-dataset",
            revision="1.0.0",
        ),
        "type": "Clustering",
        "category": "p2p",
        "eval_splits": ["test"],
        "eval_langs": ["eng-Latn"],
        "main_score": "v_measure",
        "revision": "v1.0.0",
        "scores": [
            meta_data.Score(
                metric="v_measure",
                description="The V-Measure score.",
                min=0,
                max=1,
            )
        ],
        "domains": ["Legal", "Medical"],
        "task_subtypes": ["Political"],
    }

  def test_successful_instantiation(self):
    meta_data.TaskMetadata(**self._get_valid_params())

  def test_is_frozen(self):
    metadata = meta_data.TaskMetadata(**self._get_valid_params())
    with self.assertRaises(dataclasses.FrozenInstanceError):
      metadata.name = "A New Name"

  @parameterized.product(
      field=[
          "name",
          "description",
          "reference",
          "type",
          "category",
          "main_score",
          "revision"
      ],
      invalid_value=[None, ""],
  )
  def test_invalid_required_string_fields(self, field, invalid_value):
    params = self._get_valid_params()
    params[field] = invalid_value
    with self.assertRaisesRegex(
        TypeError,
        f"Metadata attribute '{field}' must be a non-empty string."
    ):
      meta_data.TaskMetadata(**params)

  @parameterized.named_parameters(
      (
          "not_a_list",
          "eval_splits",
          "not-a-list",
          "must be a list"
      ),
      (
          "non_empty_list_fail",
          "eval_langs",
          [],
          "must be a non-empty list"
      ),
      (
          "non_string_item",
          "domains",
          ["Legal", None],
          "All items.*must be strings"
      ),
  )
  def test_invalid_list_fields(self, field, invalid_value, message):
    params = self._get_valid_params()
    params[field] = invalid_value
    with self.assertRaisesRegex(TypeError, message):
      meta_data.TaskMetadata(**params)

  def test_dataset_instantiation_requires_path(self):
    with self.assertRaisesRegex(
        TypeError,
        "got an unexpected keyword argument 'wrong_key'"
    ):
      meta_data.Dataset(wrong_key="some-path")  # type: ignore

  def test_main_score_must_be_in_scores_list(self):
    params = self._get_valid_params()
    params["main_score"] = "accuracy"
    with self.assertRaisesRegex(
        ValueError,
        "main_score 'accuracy' is not defined in the 'scores' list."
    ):
      meta_data.TaskMetadata(**params)


if __name__ == "__main__":
  absltest.main()
