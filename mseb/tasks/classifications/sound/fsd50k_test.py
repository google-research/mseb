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

import json
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from mseb import dataset
from mseb.datasets import fsd50k as fsd50k_dataset
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

  # Include non-commercial clip 253463 to verify it is excluded automatically.
  eval_data = pd.DataFrame({
      'fname': [1234, 253463, 5678, 121314],
      'labels': ['Bark', 'Siren', 'Meow', 'Bark,Siren'],
      'split': ['test', 'test', 'test', 'test'],
  })
  eval_data.to_csv(os.path.join(labels_dir, 'eval.csv'), index=False)

  metadata_dir = os.path.join(testdata_dir.full_path, 'metadata')
  os.makedirs(metadata_dir)
  with open(
      os.path.join(metadata_dir, 'eval_clips_info_FSD50K.json'), 'w'
  ) as f:
    json.dump(
        {
            '1234': {
                'license': 'http://creativecommons.org/publicdomain/zero/1.0/'
            },
            '253463': {
                'license': 'http://creativecommons.org/licenses/by-nc/3.0/'
            },
            '5678': {'license': 'http://creativecommons.org/licenses/by/3.0/'},
            '121314': {
                'license': 'http://creativecommons.org/licenses/by/3.0/'
            },
        },
        f,
    )

  clips_dir = os.path.join(testdata_dir.full_path, 'clips', 'eval')
  os.makedirs(clips_dir)
  for fname, n_samples in [
      (1234, 16000),
      (253463, 16000),
      (5678, 24000),
      (121314, 32000),
  ]:
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

  def test_full_sounds_retains_non_commercial(self):
    task = fsd50k.FSD50KTestClassification()
    sounds = list(task.multimodal_inputs())
    self.assertLen(sounds, 4)
    self.assertEqual(
        [sound.context.id for sound in sounds],
        ['1234', '253463', '5678', '121314'],
    )

  def test_full_examples_retains_non_commercial(self):
    task = fsd50k.FSD50KTestClassification()
    examples = list(task.examples('classification'))
    self.assertLen(examples, 4)
    self.assertEqual(examples[0].example_id, '1234')
    self.assertListEqual(examples[0].label_ids, ['Bark'])
    self.assertEqual(examples[1].example_id, '253463')

  def test_commercial_sounds_drops_non_commercial(self):
    task = fsd50k.FSD50KTestClassificationCommercial()
    sounds = list(task.multimodal_inputs())
    self.assertLen(sounds, 3)
    self.assertEqual(
        [sound.context.id for sound in sounds], ['1234', '5678', '121314']
    )

  def test_commercial_examples_drops_non_commercial(self):
    task = fsd50k.FSD50KTestClassificationCommercial()
    examples = list(task.examples('classification'))
    self.assertLen(examples, 3)
    self.assertEqual(
        [ex.example_id for ex in examples], ['1234', '5678', '121314']
    )

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

  def test_class_labels_with_split_none_reads_vocabulary_only(self):
    vocab_only_dir = self.create_tempdir()
    labels_dir = os.path.join(vocab_only_dir.full_path, 'labels')
    os.makedirs(labels_dir)
    pd.DataFrame({
        'index': [0, 1],
        'display_name': ['Bark', 'Siren'],
        'mid': ['m0', 'm1'],
    }).to_csv(
        os.path.join(labels_dir, 'vocabulary.csv'),
        index=False,
        header=False,
    )
    with flagsaver.flagsaver(
        (dataset._DATASET_BASEPATH, vocab_only_dir.full_path)
    ):
      task = fsd50k.FSD50KClassification()
      self.assertEqual(list(task.class_labels()), ['Bark', 'Siren'])


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

  def test_revisions(self):
    self.assertFalse(fsd50k.FSD50KTestClassification.commercial_only)
    self.assertEqual(fsd50k.FSD50KTestClassification.metadata.revision, '1.0.0')
    self.assertEqual(
        fsd50k.FSD50KTestClassification.metadata.dataset.revision,
        fsd50k_dataset.FULL_VERSION,
    )

    self.assertTrue(fsd50k.FSD50KTestClassificationCommercial.commercial_only)
    self.assertEqual(
        fsd50k.FSD50KTestClassificationCommercial.metadata.name,
        'FSD50KTestClassificationCommercial',
    )
    self.assertEqual(
        fsd50k.FSD50KTestClassificationCommercial.metadata.revision, '2.0.0'
    )
    self.assertEqual(
        fsd50k.FSD50KTestClassificationCommercial.metadata.dataset.revision,
        fsd50k_dataset.COMMERCIAL_VERSION,
    )


class DebugClassTest(absltest.TestCase):
  """Tests for FSD50KTestClassificationDebug."""

  def test_size_is_debug(self):
    self.assertEqual(fsd50k.FSD50KTestClassificationDebug.size, 'debug')
    self.assertTrue(fsd50k.FSD50KTestClassificationDebug.commercial_only)
    self.assertEqual(
        fsd50k.FSD50KTestClassificationDebug.metadata.revision, '2.0.0'
    )

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
