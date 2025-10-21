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
from mseb import types
from mseb.datasets import fsd50k
import numpy as np
import pandas as pd
from scipy.io import wavfile


class FSD50KDatasetTest(absltest.TestCase):
  """Tests for the FSD50KDataset class."""

  def setUp(self):
    """Set up a temporary directory and mock data files."""
    super().setUp()
    self.testdata_dir = self.create_tempdir()

    # Create a mock vocabulary.csv file, which the class needs to load.
    labels_dir = os.path.join(self.testdata_dir.full_path, 'labels')
    os.makedirs(labels_dir)
    vocab_data = {
        'index': [0, 1, 2, 3, 4],
        'display_name': [
            'Electric_guitar',
            'Guitar',
            'Plucked_string_instrument',
            'Musical_instrument',
            'Music',
        ],
        'mid': ['/m/02sgy', '/m/0342h', '/m/0fx80y', '/m/04szw', '/m/04rlf'],
    }
    pd.DataFrame(vocab_data).to_csv(
        os.path.join(labels_dir, 'vocabulary.csv'), header=False, index=False
    )
    eval_data = {
        'fname': [37199, 175151],
        'labels': [
            'Electric_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music',
            'Electric_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music',
        ],
        'mids': [
            '/m/02sgy,/m/0342h,/m/0fx80y,/m/04szw,/m/04rlf',
            '/m/02sgy,/m/0342h,/m/0fx80y,/m/04szw,/m/04rlf',
        ],
        'split': ['test', 'test'],
    }
    pd.DataFrame(eval_data).to_csv(
        os.path.join(labels_dir, 'eval.csv'), index=False
    )
    clips_dir = os.path.join(self.testdata_dir.full_path, 'clips', 'eval')
    os.makedirs(clips_dir)
    wavfile.write(
        os.path.join(clips_dir, '37199.wav'),
        16000,
        np.zeros(16000, dtype=np.float32),
    )
    wavfile.write(
        os.path.join(clips_dir, '175151.wav'),
        16000,
        np.zeros(32000, dtype=np.float32),
    )

  @mock.patch('mseb.utils.download_from_hf')
  def test_loading_and_label_methods(
      self,
      mock_download_from_hf,
  ):
    dataset = fsd50k.FSD50KDataset(
        split='test',
        base_path=self.testdata_dir.full_path,
    )

    mock_download_from_hf.assert_called_once()
    self.assertLen(dataset, 2)
    self.assertListEqual(
        dataset.class_labels,
        [
            'Electric_guitar',
            'Guitar',
            'Plucked_string_instrument',
            'Musical_instrument',
            'Music',
        ],
    )
    task_data = dataset.get_task_data()
    self.assertEqual(
        task_data.iloc[0]['labels'],
        (
            'Electric_guitar,'
            'Guitar,'
            'Plucked_string_instrument,'
            'Musical_instrument,'
            'Music'
        ),
    )
    self.assertEqual(
        task_data.iloc[1]['labels'],
        (
            'Electric_guitar,'
            'Guitar,'
            'Plucked_string_instrument,'
            'Musical_instrument,'
            'Music'
        ),
    )
    sound1 = dataset.get_sound(task_data.iloc[0].to_dict())
    self.assertIsInstance(sound1, types.Sound)
    self.assertEqual(sound1.context.id, '37199')
    self.assertLen(sound1.waveform, 16000)

    sound2 = dataset.get_sound(task_data.iloc[1].to_dict())
    self.assertEqual(sound2.context.id, '175151')
    self.assertLen(sound2.waveform, 32000)

if __name__ == '__main__':
  absltest.main()
