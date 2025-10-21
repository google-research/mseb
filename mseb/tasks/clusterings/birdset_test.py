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

"""Tests for Birdset clustering tasks."""

import json
import os

from absl.testing import absltest
from absl.testing import flagsaver
from mseb import types
from mseb.evaluators import clustering_evaluator
from mseb.tasks.clusterings import birdset
import numpy as np
import pandas as pd


class BirdsetClusteringTest(absltest.TestCase):
  """Tests for the BirdsetClustering class."""

  def setUp(self):
    super().setUp()
    self.testdata_dir = self.create_tempdir()

    mock_data = dict(
        audio=[
            {"waveform": np.ones(32000), "sample_rate": 32_000},
            {"waveform": np.ones(32000), "sample_rate": 32_000},
            {"waveform": np.ones(32000), "sample_rate": 32_000},
            {"waveform": np.ones(32000), "sample_rate": 32_000},
        ],
        filepath=[
            "fake/path/audio.ogg",
            "fake/path/audio2.ogg",
            "fake/path/audio3.ogg",
            "fake/path/audio4.ogg",
        ],
        ebird_code_multilabel=[
            [0],  # astcal
            [1, 2],  # brnthr, rufwar1
            [1],  # brnthr
            [],  # no_label
        ],
        sex=["male", "female", "male", "unknown"],
        other_col=[123, 456, 789, 1011],
    )
    self.fake_df = pd.DataFrame(mock_data)
    class_lists = {"some_list": ["astcal", "brnthr", "rufwar1"]}
    config_to_class_list = {"HSN": "some_list"}

    # Write fake data to temporary files
    fake_parquet_path = os.path.join(
        self.testdata_dir.full_path, "birdset_HSN_test.parquet"
    )
    self.fake_df.to_parquet(fake_parquet_path)

    fake_class_lists_path = os.path.join(
        self.testdata_dir.full_path, "class_lists.json"
    )
    with open(fake_class_lists_path, "w") as f:
      json.dump(class_lists, f)

    fake_config_path = os.path.join(
        self.testdata_dir.full_path, "config_to_class_list.json"
    )
    with open(fake_config_path, "w") as f:
      json.dump(config_to_class_list, f)

    self.enter_context(
        flagsaver.flagsaver(dataset_basepath=self.testdata_dir.full_path)
    )

  def test_sub_tasks(self):
    task = birdset.BirdsetClusteringHSN()
    task.setup()
    self.assertEqual(task.sub_tasks, ["clustering"])

  def test_examples(self):
    task = birdset.BirdsetClusteringHSN()
    task.setup()
    examples = list(task.examples("clustering"))
    expected_examples = [
        clustering_evaluator.ClusteringExample("fake/path/audio.ogg", "astcal"),
        clustering_evaluator.ClusteringExample(
            "fake/path/audio2.ogg", "brnthr"
        ),
        clustering_evaluator.ClusteringExample(
            "fake/path/audio3.ogg", "brnthr"
        ),
        clustering_evaluator.ClusteringExample(
            "fake/path/audio4.ogg", "no_label"
        ),
    ]
    self.assertEqual(examples, expected_examples)

  def test_sounds(self):
    task = birdset.BirdsetClusteringHSN()
    task.setup()
    sounds = list(task.sounds())
    self.assertLen(sounds, 4)
    self.assertIsInstance(sounds[0], types.Sound)
    self.assertEqual(sounds[0].context.id, "fake/path/audio.ogg")
    self.assertEqual(sounds[1].context.id, "fake/path/audio2.ogg")
    self.assertEqual(sounds[2].context.id, "fake/path/audio3.ogg")
    self.assertEqual(sounds[3].context.id, "fake/path/audio4.ogg")


if __name__ == "__main__":
  absltest.main()
