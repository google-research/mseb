# Copyright 2026 The MSEB Authors.
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

import os
from typing import Any

from absl.testing import absltest
from mseb import dataset
from mseb import types
import numpy as np
import pandas as pd


class MockDataset(dataset.Dataset):
  """A mock dataset for testing the Dataset's shared functionality."""

  def __init__(self, split: str, base_path: str):
    self.mock_metadata_path = os.path.join(base_path, f"{split}.csv")
    self.mock_data = {
        "sound_id": ["a", "b", "c", "d", "e"],
        "label": ["cat", "dog", "cat", "bird", "dog"],
        "value": [10, 20, 30, 40, 50]
    }
    pd.DataFrame(self.mock_data).to_csv(self.mock_metadata_path, index=False)
    super().__init__(split, base_path=base_path)

  @property
  def metadata(self) -> types.DatasetMetadata:
    return types.DatasetMetadata(
        name="MockDataset",
        description="A fake dataset for testing.",
        homepage="http://fake.com",
        version="1.0",
        license="N/A",
        mseb_tasks=["testing"]
    )

  def _load_metadata(self) -> pd.DataFrame:
    return pd.read_csv(self.mock_metadata_path)

  def _get_sound(self, record: dict[str, Any]) -> types.Sound:
    return types.Sound(
        waveform=np.zeros(16000, dtype=np.float32),
        context=types.SoundContextParams(
            id=record["sound_id"],
            sample_rate=16000,
            length=16000,
            text=record["label"]
        )
    )


class DatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = self.create_tempdir().full_path
    self.dataset = MockDataset(split="test", base_path=self.test_dir)

  def test_dataset_len(self):
    self.assertLen(self.dataset, 5)

  def test_dataset_getitem(self):
    item = self.dataset[1]
    self.assertIsInstance(item, types.Sound)
    self.assertEqual(item.context.id, "b")
    self.assertEqual(item.context.text, "dog")

  def test_dataset_iter(self):
    items = list(self.dataset)
    self.assertLen(items, 5)
    self.assertEqual(items[2].context.id, "c")

  def test_available_labels(self):
    expected_labels = ["sound_id", "label", "value"]
    self.assertEqual(self.dataset.available_labels, expected_labels)

  def test_get_labels(self):
    labels = self.dataset.get_labels("label")
    expected = ["cat", "dog", "cat", "bird", "dog"]
    self.assertEqual(labels, expected)

  def test_get_labels_raises_error_for_invalid_name(self):
    with self.assertRaises(ValueError):
      self.dataset.get_labels("non_existent_label")

  def test_batch_iterator_creation(self):
    batch_iterator = self.dataset.as_batch_iterator(batch_size=2)
    self.assertIsNotNone(batch_iterator)

  def test_batch_iterator_len(self):
    # 5 items, batch size 2 -> 3 batches (2, 2, 1)
    batch_iterator = self.dataset.as_batch_iterator(batch_size=2)
    self.assertLen(batch_iterator, 3)

    # 5 items, batch size 5 -> 1 batch
    batch_iterator = self.dataset.as_batch_iterator(batch_size=5)
    self.assertLen(batch_iterator, 1)

  def test_batch_iterator_yields_correct_batches(self):
    batch_iterator = self.dataset.as_batch_iterator(batch_size=2)
    batches = list(batch_iterator)
    self.assertLen(batches, 3)
    self.assertLen(batches[0], 2)
    self.assertLen(batches[1], 2)
    self.assertLen(batches[2], 1)
    # Check content of the first batch
    self.assertEqual(batches[0][0].context.id, "a")
    self.assertEqual(batches[0][1].context.id, "b")


if __name__ == "__main__":
  absltest.main()
