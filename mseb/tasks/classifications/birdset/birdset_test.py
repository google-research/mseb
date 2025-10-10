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

"""Tests for Birdset classification tasks."""

import json
import os

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from mseb import dataset
from mseb.tasks.classifications.birdset import birdset
import numpy as np
import pandas as pd


FLAGS = flags.FLAGS


class BirdsetHSNClassificationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.testdata_dir = self.create_tempdir()

    mock_data = dict(
        audio=[
            {"waveform": np.ones(32000), "sample_rate": 32_000},
            {"waveform": np.ones(32000), "sample_rate": 32_000},
            {"waveform": np.ones(32000), "sample_rate": 32_000},
        ],
        filepath=[
            "fake/path/audio.ogg",
            "fake/path/audio2.ogg",
            "fake/path/audio3.ogg",
        ],
        ebird_code_multilabel=[
            [0],  # ['astcal']
            [1, 2],  # ['brnthr', 'rufwar1']
            [],   # No labels.
        ],
        sex=["male", "female", "male"],
        other_col=[123, 456, 789],
    )
    self.fake_df = pd.DataFrame(mock_data)
    class_lists = {"some_list": ["astcal", "brnthr", "rufwar1"]}
    config_to_class_list = {"HSN": "some_list"}

    # Write fake data to temporary files
    fake_parquet_path = os.path.join(
        self.testdata_dir.full_path, "birdset_HSN_test_5s.parquet"
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
        flagsaver.flagsaver((
            dataset._DATASET_BASEPATH,
            self.testdata_dir.full_path,
        ))
    )

  def test_birdset_classification_sounds(self):
    task = birdset.BirdsetHSNClassification()
    sounds = list(task.sounds())
    self.assertLen(sounds, 3)
    self.assertEqual(sounds[0].context.id, "fake/path/audio.ogg")
    self.assertEqual(sounds[0].context.text, "astcal")
    self.assertLen(sounds[0].waveform, 32000)
    self.assertEqual(sounds[1].context.id, "fake/path/audio2.ogg")
    self.assertEqual(sounds[1].context.text, "brnthr,rufwar1")
    self.assertLen(sounds[1].waveform, 32000)
    self.assertEqual(sounds[2].context.id, "fake/path/audio3.ogg")
    self.assertEqual(sounds[2].context.text, "")
    self.assertLen(sounds[2].waveform, 32000)

  def test_birdset_classification_examples(self):
    task = birdset.BirdsetHSNClassification()
    examples = list(task.examples("ebird_classification"))
    self.assertLen(examples, 3)
    self.assertEqual(examples[0].example_id, "fake/path/audio.ogg")
    self.assertEqual(examples[0].label_ids, ["astcal"])
    self.assertEqual(examples[1].example_id, "fake/path/audio2.ogg")
    self.assertEqual(examples[1].label_ids, ["brnthr", "rufwar1"])
    self.assertEqual(examples[2].example_id, "fake/path/audio3.ogg")
    self.assertEqual(examples[2].label_ids, [])

  def test_birdset_classification_class_labels(self):
    task = birdset.BirdsetHSNClassification()
    self.assertEqual(task.sub_tasks, ["ebird_classification"])
    class_labels = list(task.class_labels())
    self.assertEqual(class_labels, ["astcal", "brnthr", "rufwar1"])


if __name__ == "__main__":
  absltest.main()
