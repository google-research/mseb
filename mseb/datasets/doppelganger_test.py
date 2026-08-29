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

from absl.testing import absltest
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


if __name__ == '__main__':
  absltest.main()
