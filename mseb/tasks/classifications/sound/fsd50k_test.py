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
from absl.testing import flagsaver
from mseb import dataset
from mseb.tasks.classifications.sound import fsd50k
import numpy as np
import pandas as pd
from scipy.io import wavfile


class FSD50KTestClassificationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.testdata_dir = self.create_tempdir()
    labels_dir = os.path.join(self.testdata_dir.full_path, 'labels')
    os.makedirs(labels_dir)

    vocab_data = pd.DataFrame({
        'index': [0, 1, 2],
        'display_name': ['Bark', 'Meow', 'Siren'],
        'mid': ['m0', 'm1', 'm2'],
    })
    vocab_data.to_csv(
        os.path.join(labels_dir, 'vocabulary.csv'),
        index=False,
        header=False
    )

    # This mock data ONLY contains 'test' records.
    eval_data = pd.DataFrame({
        'fname': [1234, 5678, 121314],
        'labels': ['Bark', 'Meow', 'Bark,Siren'],
        'split': ['test', 'test', 'test'],
    })
    eval_data.to_csv(os.path.join(labels_dir, 'eval.csv'), index=False)

    clips_dir = os.path.join(self.testdata_dir.full_path, 'clips', 'eval')
    os.makedirs(clips_dir)
    wavfile.write(
        os.path.join(clips_dir, '1234.wav'),
        16000,
        np.zeros(16000, dtype=np.int16)
    )
    wavfile.write(
        os.path.join(clips_dir, '5678.wav'),
        16000,
        np.zeros(24000, dtype=np.int16)
    )
    wavfile.write(
        os.path.join(clips_dir, '121314.wav'),
        16000,
        np.zeros(32000, dtype=np.int16)
    )

    self.enter_context(
        flagsaver.flagsaver(
            (dataset._DATASET_BASEPATH, self.testdata_dir.full_path)
        )
    )
    self.enter_context(
        mock.patch('mseb.utils.download_from_hf', return_value=None)
    )

  def test_sounds(self):
    task = fsd50k.FSD50KTestClassification()
    sounds = list(task.sounds())
    self.assertLen(sounds, 3)
    self.assertEqual(sounds[0].context.id, '1234')
    self.assertEqual(sounds[1].context.id, '5678')
    self.assertEqual(sounds[2].context.id, '121314')

  def test_examples(self):
    task = fsd50k.FSD50KTestClassification()
    examples = list(task.examples('classification'))
    self.assertLen(examples, 3)
    self.assertEqual(examples[0].example_id, '1234')
    self.assertListEqual(examples[0].label_ids, ['Bark'])

  def test_class_labels(self):
    task = fsd50k.FSD50KTestClassification()
    labels = task.class_labels()
    self.assertListEqual(list(labels), ['Bark', 'Meow', 'Siren'])


if __name__ == '__main__':
  absltest.main()
