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

"""Tests for the Doppelganger dataset loader."""

from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from mseb import types
from mseb.datasets import doppelganger
from mseb.datasets.doppelganger_testutil import create_mock_dataset


class DoppelgangerDatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.base_dir = self.create_tempdir().full_path
    create_mock_dataset(self.base_dir)
    self.dataset = doppelganger.DoppelgangerDataset(
        split='test', base_path=self.base_dir
    )

  def test_metadata_loading(self):
    self.assertLen(self.dataset, 3)
    self.assertEqual(
        self.dataset.get_task_data()['pair_id'].tolist(),
        [
            '1',
            '2',
            '3',
        ],
    )

  def test_domain_sounds(self):
    record = self.dataset.get_task_data().iloc[0].to_dict()
    real = self.dataset.get_real_sound(record)
    synthetic = self.dataset.get_synthetic_sound(record)

    self.assertIsInstance(real, types.Sound)
    self.assertEqual(real.context.id, 'real:1')
    self.assertLen(real.waveform, 8000)
    self.assertEqual(synthetic.context.id, 'synthetic:1')
    self.assertLen(synthetic.waveform, 4000)

  def test_get_sound_requires_domain(self):
    record = self.dataset.get_task_data().iloc[0].to_dict()
    with self.assertRaisesRegex(ValueError, 'must include a domain'):
      self.dataset.get_sound(record)

  def test_split_validation(self):
    with self.assertRaisesRegex(ValueError, 'Split must be test'):
      doppelganger.DoppelgangerDataset(split='train', base_path=self.base_dir)

  def test_filter_fn(self):
    filtered = doppelganger.DoppelgangerDataset(
        split='test',
        base_path=self.base_dir,
        filter_fn=lambda pair_id: str(pair_id) in {'1', '3'},
    )
    self.assertEqual(filtered.get_task_data()['pair_id'].tolist(), ['1', '3'])

  def test_default_base_path_resolution(self):
    with flagsaver.flagsaver(dataset_basepath=None):
      with mock.patch.object(
          doppelganger, 'DEFAULT_BASE_PATH', self.base_dir
      ):
        ds = doppelganger.DoppelgangerDataset(split='test')
        self.assertEqual(ds.base_path, self.base_dir)
        self.assertLen(ds, 3)

  def test_default_base_path_missing_files_raises(self):
    empty_dir = self.create_tempdir().full_path
    with flagsaver.flagsaver(dataset_basepath=None):
      with mock.patch.object(doppelganger, 'DEFAULT_BASE_PATH', empty_dir):
        with self.assertRaisesRegex(
            FileNotFoundError, 'Doppelganger metadata not found'
        ):
          doppelganger.DoppelgangerDataset(split='test')

  def test_is_member_of_debug_true(self):
    for pair_id in (
        '168765981935',
        '204643997502',
        '218906114721',
        '224092971726',
        '292264958284',
    ):
      with self.subTest(pair_id=pair_id):
        self.assertTrue(doppelganger.is_member_of_debug(pair_id))
        self.assertTrue(doppelganger.is_member_of_debug(int(pair_id)))

  def test_is_member_of_debug_false(self):
    self.assertFalse(doppelganger.is_member_of_debug('non_existent_id'))
    self.assertFalse(doppelganger.is_member_of_debug(''))


if __name__ == '__main__':
  absltest.main()
