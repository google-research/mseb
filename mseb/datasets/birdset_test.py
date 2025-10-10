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

"""Tests for Birdset dataset."""

import json
import os

from absl.testing import absltest
from mseb import types
from mseb.datasets import birdset
import numpy as np
import pandas as pd


class BirdsetDatasetTest(absltest.TestCase):
  """Tests for the BirdsetDataset class."""

  def setUp(self):
    super().setUp()
    self.testdata_dir = self.create_tempdir()

    mock_data = dict(
        audio=[
            {"waveform": np.ones(32000), "sampling_rate": 32_000},
            {"waveform": np.ones(32000), "sampling_rate": 32_000},
        ],
        filepath=["fake/path/audio.ogg", "fake/path/audio2.ogg"],
        ebird_code_multilabel=[
            [0],  # ['astcal']
            [1, 2],  # ['brnthr', 'rufwar1']
        ],
        sex=["male", "female"],
        other_col=[123, 456],
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

  def test_initialization_invalid_split_raises_error(self):
    with self.assertRaisesRegex(ValueError, "Split must be"):
      birdset.BirdsetDataset(
          base_path=self.testdata_dir.full_path,
          split="validation",
          configuration="HSN",
      )

  def test_loading_and_parsing(self):
    ds = birdset.BirdsetDataset(
        base_path=self.testdata_dir.full_path,
        split="test_5s",
        configuration="HSN",
    )

    self.assertLen(ds, 2)

    task_df = ds.get_task_data()
    self.assertIsInstance(task_df, pd.DataFrame)
    self.assertLen(task_df, 2)

    record1 = task_df.iloc[0]
    self.assertEqual(record1.filepath, "fake/path/audio.ogg")
    self.assertEqual(record1.ebird_code_multilabel, ["astcal"])
    self.assertEqual(record1.sex, "male")

    record2 = task_df.iloc[1]
    self.assertEqual(record2.filepath, "fake/path/audio2.ogg")
    self.assertEqual(record2.ebird_code_multilabel, ["brnthr", "rufwar1"])
    self.assertEqual(record2.sex, "female")

    self.assertEqual(ds._ebird_code_names, ["astcal", "brnthr", "rufwar1"])

  def test_get_sound(self):
    ds = birdset.BirdsetDataset(
        base_path=self.testdata_dir.full_path,
        split="test_5s",
        configuration="HSN",
    )
    record = ds.get_task_data().iloc[0]
    sound = ds.get_sound(record)

    self.assertIsInstance(sound, types.Sound)
    self.assertEqual(sound.context.id, "fake/path/audio.ogg")
    self.assertEqual(sound.context.speaker_gender, "male")
    self.assertEqual(sound.context.text, "astcal")
    self.assertEqual(sound.context.sample_rate, 32_000)


if __name__ == "__main__":
  absltest.main()
