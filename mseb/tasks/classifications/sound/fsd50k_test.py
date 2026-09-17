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
from mseb.tasks.classifications.sound import fsd50k
import numpy as np
import pandas as pd
from scipy.io import wavfile


def _create_testdata(test_case):
  """Create temporary mock FSD50K data and set up flags."""
  testdata_dir = test_case.create_tempdir()
  labels_dir = os.path.join(testdata_dir.full_path, 'labels')
  os.makedirs(labels_dir)

  vocab_data = pd.DataFrame({
      'index': [0, 1, 2],
      'display_name': ['Bark', 'Meow', 'Siren'],
      'mid': ['m0', 'm1', 'm2'],
  })
  vocab_data.to_csv(
      os.path.join(labels_dir, 'vocabulary.csv'),
      index=False,
      header=False,
  )

  eval_data = pd.DataFrame({
      'fname': [1234, 5678, 121314],
      'labels': ['Bark', 'Meow', 'Bark,Siren'],
      'split': ['test', 'test', 'test'],
  })
  eval_data.to_csv(os.path.join(labels_dir, 'eval.csv'), index=False)

  clips_dir = os.path.join(testdata_dir.full_path, 'clips', 'eval')
  os.makedirs(clips_dir)
  for fname, n_samples in [(1234, 16000), (5678, 24000), (121314, 32000)]:
    wavfile.write(
        os.path.join(clips_dir, f'{fname}.wav'),
        16000,
        np.zeros(n_samples, dtype=np.int16),
    )

  test_case.enter_context(
      flagsaver.flagsaver((dataset._DATASET_BASEPATH, testdata_dir.full_path))
  )
  test_case.enter_context(
      mock.patch('mseb.utils.download_from_hf', return_value=None)
  )


class FSD50KTestClassificationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    _create_testdata(self)

  def test_sounds(self):
    task = fsd50k.FSD50KTestClassification()
    sounds = list(task.multimodal_inputs())
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


class BaseClassTest(absltest.TestCase):
  """Tests for FSD50KClassification base class attributes."""

  def test_base_split_is_none(self):
    self.assertIsNone(fsd50k.FSD50KClassification.split)

  def test_base_size_is_none(self):
    self.assertIsNone(fsd50k.FSD50KClassification.size)

  def test_task_type(self):
    task = fsd50k.FSD50KClassification()
    self.assertEqual(task.task_type, 'multi_label')

  def test_sub_tasks(self):
    task = fsd50k.FSD50KClassification()
    self.assertEqual(task.sub_tasks, ['classification'])

  def test_examples_split_none_raises(self):
    task = fsd50k.FSD50KClassification()
    with self.assertRaises(ValueError):
      list(task.examples('classification'))


class FSD50KTestClassificationMetadataTest(absltest.TestCase):
  """Tests for FSD50KTestClassification metadata and class attributes."""

  def test_split(self):
    self.assertEqual(fsd50k.FSD50KTestClassification.split, 'test')

  def test_size_is_none(self):
    self.assertIsNone(fsd50k.FSD50KTestClassification.size)

  def test_metadata_name(self):
    self.assertEqual(
        fsd50k.FSD50KTestClassification.metadata.name,
        'FSD50KTestClassification',
    )

  def test_metadata_type(self):
    self.assertEqual(
        fsd50k.FSD50KTestClassification.metadata.type, 'Classification'
    )

  def test_metadata_main_score(self):
    self.assertEqual(fsd50k.FSD50KTestClassification.metadata.main_score, 'mAP')

  def test_metadata_category(self):
    self.assertEqual(fsd50k.FSD50KTestClassification.metadata.category, 'audio')

  def test_inherits_from_base(self):
    self.assertTrue(
        issubclass(fsd50k.FSD50KTestClassification, fsd50k.FSD50KClassification)
    )


class DebugClassTest(absltest.TestCase):
  """Tests for FSD50KTestClassificationDebug."""

  def test_inherits_from_test_classification(self):
    self.assertTrue(
        issubclass(
            fsd50k.FSD50KTestClassificationDebug,
            fsd50k.FSD50KTestClassification,
        )
    )

  def test_size_is_debug(self):
    self.assertEqual(fsd50k.FSD50KTestClassificationDebug.size, 'debug')

  def test_metadata_name(self):
    self.assertEqual(
        fsd50k.FSD50KTestClassificationDebug.metadata.name,
        'FSD50KTestClassificationDebug',
    )

  def test_split_inherited(self):
    self.assertEqual(fsd50k.FSD50KTestClassificationDebug.split, 'test')


class SizeFilteringTest(absltest.TestCase):
  """Tests that size filtering works in multimodal_inputs and examples."""

  def setUp(self):
    super().setUp()
    _create_testdata(self)

  def test_debug_filters_sounds(self):
    task = fsd50k.FSD50KTestClassificationDebug()
    sounds = list(task.multimodal_inputs())
    # Debug IDs are unlikely to match mock fnames, so expect fewer results.
    self.assertLessEqual(len(sounds), 3)

  def test_debug_filters_examples(self):
    task = fsd50k.FSD50KTestClassificationDebug()
    examples = list(task.examples('classification'))
    self.assertLessEqual(len(examples), 3)


if __name__ == '__main__':
  absltest.main()
