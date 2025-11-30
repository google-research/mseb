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

"""Tests for the ParquetDataset."""

import os
import tempfile

from absl.testing import absltest
from mseb.datasets import parquet
import numpy as np
import pandas as pd


class ParquetDatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Create a temporary directory for the test data.
    self.temp_dir = tempfile.TemporaryDirectory()
    self.base_path = self.temp_dir.name

    # Create a dummy parquet file.
    self.dataset_name = 'DummyDataset'
    self.task_name = 'dummy_task'
    self.num_samples = 5
    self.sample_rate = 16000

    # Create dummy audio data.
    audio_data = []
    for _ in range(self.num_samples):
      audio_data.append({
          'waveform': np.random.randn(self.sample_rate).astype(np.float32),
          'sample_rate': self.sample_rate,
      })

    # Create a dummy DataFrame.
    df = pd.DataFrame({
        'id': range(self.num_samples),
        'audio': audio_data,
    })

    # Save the DataFrame to a parquet file.
    self.parquet_filename = f'{self.dataset_name}_{self.task_name}.parquet'
    self.parquet_path = os.path.join(self.base_path, self.parquet_filename)
    df.to_parquet(self.parquet_path)

  def tearDown(self):
    super().tearDown()
    self.temp_dir.cleanup()

  def test_parquet_dataset_loading(self):
    """Tests if the ParquetDataset can be loaded correctly."""
    dataset = parquet.ParquetDataset(
        dataset_name=self.dataset_name,
        task_name=self.task_name,
        base_path=self.base_path,
    )
    self.assertLen(dataset, self.num_samples)

  def test_get_task_data(self):
    """Tests the get_task_data method."""
    dataset = parquet.ParquetDataset(
        dataset_name=self.dataset_name,
        task_name=self.task_name,
        base_path=self.base_path,
    )
    task_data = dataset.get_task_data(self.task_name)
    self.assertIsInstance(task_data, pd.DataFrame)
    self.assertLen(task_data, self.num_samples)

  def test_get_sound(self):
    """Tests the get_sound method."""
    dataset = parquet.ParquetDataset(
        dataset_name=self.dataset_name,
        task_name=self.task_name,
        base_path=self.base_path,
    )
    task_data = dataset.get_task_data(self.task_name)
    first_record = task_data.iloc[0].to_dict()
    sound = dataset.get_sound(first_record)
    self.assertEqual(sound.context.sample_rate, self.sample_rate)
    self.assertLen(sound.waveform, self.sample_rate)


if __name__ == '__main__':
  absltest.main()
