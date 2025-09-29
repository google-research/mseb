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

import os
from unittest import mock

from absl.testing import absltest
from mseb.datasets import fsd50k
import numpy as np
import pandas as pd


class FSD50KDatasetTest(absltest.TestCase):
  """Tests for the FSD50KDataset class."""

  def setUp(self):
    """Set up a temporary directory and mock data files."""
    super().setUp()
    self.testdata_dir = self.create_tempdir()

    # Create a mock vocabulary.csv file, which the class needs to load.
    labels_dir = os.path.join(self.testdata_dir.full_path, 'labels')
    os.makedirs(labels_dir)
    mock_vocab_data = {
        'index': [0, 1, 2],
        'mid': ['/m/0284w', '/m/032s6g', '/m/09x0r'],
        'display_name': ['Dog', 'Siren', 'Speech'],
    }
    pd.DataFrame(mock_vocab_data).to_csv(
        os.path.join(labels_dir, 'vocabulary.csv'), header=False, index=False
    )

  @mock.patch('librosa.resample')
  @mock.patch('mseb.datasets.fsd50k.datasets.load_dataset')
  def test_loading_and_label_methods(
      self,
      mock_load_dataset,
      mock_resample_audio
  ):
    # Mock the return value of datasets.load_dataset
    mock_hf_data = {
        'fname': ['1000', '1001'],
        'labels': ['Dog,Siren', 'Speech'],
        'audio': [
            {
                'array': np.zeros(16000, dtype=np.float32),
                'sampling_rate': 16000
            },
            {
                'array': np.zeros(22050, dtype=np.float32),
                'sampling_rate': 22050
            },
        ],
    }
    mock_df = pd.DataFrame(mock_hf_data)
    mock_loader_result = mock.Mock()
    mock_loader_result.to_pandas.return_value = mock_df
    mock_load_dataset.return_value = mock_loader_result

    # Mock the return value of the resampler
    mock_resample_audio.return_value = np.ones(22050, dtype=np.float32)

    dataset = fsd50k.FSD50KDataset(
        base_path=self.testdata_dir.full_path,
        split='validation',
        target_sr=22050  # Set a target SR to test resampling
    )

    mock_load_dataset.assert_called_once_with(
        'Fhrozen/FSD50k',
        name='default',
        split='validation',
        cache_dir=self.testdata_dir.full_path
    )
    self.assertLen(dataset, 2)
    self.assertListEqual(dataset.class_labels, ['Dog', 'Siren', 'Speech'])

    sound1 = dataset[0]
    self.assertEqual(sound1.context.id, '1000')
    # Should be target_sr
    self.assertEqual(sound1.context.sample_rate, 22050)
    # Should be resampled waveform
    self.assertLen(sound1.waveform, 22050)

    mock_resample_audio.assert_called_once()

    string_labels = dataset.get_string_labels(0)
    self.assertListEqual(string_labels, ['Dog', 'Siren'])

    multi_hot_labels = dataset.get_multi_hot_labels(0)
    expected_multi_hot = np.array([1, 1, 0], dtype=np.int64)
    np.testing.assert_array_equal(multi_hot_labels, expected_multi_hot)

    sound2 = dataset[1]
    self.assertEqual(sound2.context.id, '1001')
    self.assertEqual(sound2.context.sample_rate, 22050)
    # Ensure resample was not called again
    self.assertEqual(mock_resample_audio.call_count, 1)


if __name__ == '__main__':
  absltest.main()
