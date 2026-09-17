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
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from mseb import dataset
from mseb import types
from mseb.evaluators import clustering_evaluator
from mseb.tasks.clusterings import fsd50k
import numpy as np
import pandas as pd
from scipy.io import wavfile


class FSD50KClusteringTest(absltest.TestCase):
  """Tests for the FSD50KClustering class."""

  def setUp(self):
    """Set up a temporary directory and mock data files."""
    super().setUp()
    self.testdata_dir = self.create_tempdir()

    # Create a mock vocabulary.csv file.
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

    # Mock eval.csv for the 'test' split.
    eval_data = {
        'fname': [37199, 175151],
        'labels': [
            'Electric_guitar,Guitar,Plucked_string_instrument',
            'Music',
        ],
        'split': ['test', 'test'],
    }
    pd.DataFrame(eval_data).to_csv(
        os.path.join(labels_dir, 'eval.csv'), index=False
    )

    # Mock dev.csv for the 'validation' split.
    dev_data = {
        'fname': [1000, 2000, 3000],
        'labels': [
            'Siren,Vehicle',
            'Speech',
            'Music,Song',
        ],
        'split': ['val', 'val', 'train'],  # Only 'val' should be used.
    }
    pd.DataFrame(dev_data).to_csv(
        os.path.join(labels_dir, 'dev.csv'), index=False
    )

    # Create mock wav files.
    clips_eval_dir = os.path.join(self.testdata_dir.full_path, 'clips', 'eval')
    os.makedirs(clips_eval_dir)
    wavfile.write(
        os.path.join(clips_eval_dir, '37199.wav'),
        16000,
        np.zeros(16000, dtype=np.float32),
    )
    wavfile.write(
        os.path.join(clips_eval_dir, '175151.wav'),
        16000,
        np.zeros(32000, dtype=np.float32),
    )

    clips_dev_dir = os.path.join(self.testdata_dir.full_path, 'clips', 'dev')
    os.makedirs(clips_dev_dir)
    wavfile.write(
        os.path.join(clips_dev_dir, '1000.wav'),
        16000,
        np.zeros(16000, dtype=np.float32),
    )
    wavfile.write(
        os.path.join(clips_dev_dir, '2000.wav'),
        16000,
        np.zeros(16000, dtype=np.float32),
    )
    wavfile.write(
        os.path.join(clips_dev_dir, '3000.wav'),
        16000,
        np.zeros(16000, dtype=np.float32),
    )

    self.enter_context(
        flagsaver.flagsaver(
            (dataset._DATASET_BASEPATH, self.testdata_dir.full_path)
        )
    )
    self.enter_context(
        mock.patch('mseb.utils.download_from_hf', return_value=None)
    )

  def test_test_clustering_sub_tasks(self):
    task = fsd50k.FSD50KTestClustering()
    task.setup()
    self.assertEqual(task.sub_tasks, ['sound_event'])

  def test_test_clustering_examples(self):
    task = fsd50k.FSD50KTestClustering()
    task.setup()
    examples = list(task.examples('sound_event'))
    expected_examples = [
        clustering_evaluator.ClusteringExample('37199', 'Electric_guitar'),
        clustering_evaluator.ClusteringExample('175151', 'Music'),
    ]
    self.assertEqual(examples, expected_examples)

  def test_test_clustering_sounds(self):
    task = fsd50k.FSD50KTestClustering()
    task.setup()
    sounds = list(task.multimodal_inputs())
    self.assertLen(sounds, 2)
    self.assertIsInstance(sounds[0], types.Sound)
    self.assertEqual(sounds[0].context.id, '37199')
    self.assertEqual(sounds[1].context.id, '175151')


class BaseClassTest(absltest.TestCase):
  """Tests for FSD50KClustering base class attributes."""

  def test_base_split_is_none(self):
    self.assertIsNone(fsd50k.FSD50KClustering.split)

  def test_sub_tasks(self):
    # sub_tasks is a property, but we can check via an instance with mocked
    # split.
    task = fsd50k.FSD50KTestClustering()
    self.assertEqual(task.sub_tasks, ['sound_event'])

  def test_split_none_raises(self):
    task = fsd50k.FSD50KClustering()
    with self.assertRaises(ValueError):
      _ = task._fsd_dataset

  def test_get_label_single(self):
    task = fsd50k.FSD50KClustering()
    self.assertEqual(task._get_label({'labels': 'Music'}), 'Music')

  def test_get_label_multi(self):
    task = fsd50k.FSD50KClustering()
    self.assertEqual(task._get_label({'labels': 'Bark,Siren,Speech'}), 'Bark')


class FSD50KTestClusteringMetadataTest(absltest.TestCase):
  """Tests for FSD50KTestClustering metadata and class attributes."""

  def test_split(self):
    self.assertEqual(fsd50k.FSD50KTestClustering.split, 'test')

  def test_metadata_name(self):
    self.assertEqual(
        fsd50k.FSD50KTestClustering.metadata.name, 'FSD50KTestClustering'
    )

  def test_metadata_type(self):
    self.assertEqual(fsd50k.FSD50KTestClustering.metadata.type, 'Clustering')

  def test_metadata_main_score(self):
    self.assertEqual(
        fsd50k.FSD50KTestClustering.metadata.main_score, 'VMeasure'
    )

  def test_metadata_category(self):
    self.assertEqual(fsd50k.FSD50KTestClustering.metadata.category, 'audio')

  def test_inherits_from_base(self):
    self.assertTrue(
        issubclass(fsd50k.FSD50KTestClustering, fsd50k.FSD50KClustering)
    )


if __name__ == '__main__':
  absltest.main()
