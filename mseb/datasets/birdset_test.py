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

import json
import os
from unittest import mock

from absl.testing import absltest
from mseb import types
from mseb.datasets import birdset
import numpy as np
import pandas as pd


class BirdsetDatasetTest(absltest.TestCase):
  """Tests for the BirdsetDataset class."""

  def setUp(self):
    super().setUp()
    self.fake_df = pd.DataFrame({
        "audio": [{"waveform": np.ones(32000), "sampling_rate": 32_000}],
        "filepath": ["fake/path/audio.ogg"],
        "ebird_code": [0],  # 'astcal'
        "sex": ["male"],
        "other_col": [123],
    })
    self.class_lists = {"some_list": ["astcal", "brnthr"]}
    self.config_to_class_list = {"HSN": "some_list"}

  def test_initialization_invalid_split_raises_error(self):
    with self.assertRaisesRegex(ValueError, "Split must be"):
      birdset.BirdsetDataset(
          base_path=".",
          split="validation",
          configuration="HSN"
      )

  @mock.patch("pandas.read_parquet")
  @mock.patch("builtins.open", new_callable=mock.mock_open)
  def test_load_metadata_and_init_success(
      self, mock_open, mock_read_parquet
  ):
    mock_read_parquet.return_value = self.fake_df
    mock_open.return_value.__enter__.return_value.read.side_effect = [
        json.dumps(self.class_lists),
        json.dumps(self.config_to_class_list),
    ]

    ds = birdset.BirdsetDataset(
        base_path="/fake/cache", split="test_5s", configuration="HSN"
    )

    cache_filename = "birdset_HSN_test_5s.parquet"
    cache_path = os.path.join("/fake/cache", cache_filename)
    mock_read_parquet.assert_called_once_with(cache_path)
    self.assertEqual(mock_open.call_count, 2)
    mock_open.assert_any_call("/fake/cache/class_lists.json", "r")
    mock_open.assert_any_call("/fake/cache/config_to_class_list.json", "r")

    pd.testing.assert_frame_equal(
        ds._metadata[["filepath", "ebird_code", "sex", "other_col"]],
        self.fake_df[["filepath", "ebird_code", "sex", "other_col"]],
    )
    self.assertEqual(ds._ebird_code_names, ["astcal", "brnthr"])

  @mock.patch("pandas.read_parquet")
  @mock.patch("builtins.open", new_callable=mock.mock_open)
  def test_getitem_and_get_sound_parsing(
      self, mock_open, mock_read_parquet
  ):
    mock_read_parquet.return_value = self.fake_df
    mock_open.return_value.__enter__.return_value.read.side_effect = [
        json.dumps(self.class_lists),
        json.dumps(self.config_to_class_list),
    ]

    ds = birdset.BirdsetDataset(
        base_path="/fake/cache", split="train", configuration="HSN"
    )

    sound_item = ds[0]

    self.assertIsInstance(sound_item, types.Sound)
    self.assertEqual(sound_item.context.id, "fake/path/audio.ogg")
    self.assertEqual(sound_item.context.speaker_gender, "male")
    self.assertEqual(sound_item.context.text, "astcal")
    self.assertEqual(sound_item.context.sample_rate, 32_000)


if __name__ == "__main__":
  absltest.main()
