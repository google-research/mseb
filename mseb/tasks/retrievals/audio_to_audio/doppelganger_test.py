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

"""Tests for Doppelganger audio-to-audio retrieval tasks."""

from absl.testing import absltest
from mseb import types
from mseb.datasets import doppelganger as doppelganger_dataset
from mseb.datasets.doppelganger_test import create_mock_dataset
from mseb.tasks.retrievals.audio_to_audio import doppelganger


class DoppelgangerRetrievalTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    base_dir = self.create_tempdir().full_path
    create_mock_dataset(base_dir)
    self.dataset = doppelganger_dataset.DoppelgangerDataset(
        split='test', base_path=base_dir
    )

  def _task(self, task_class):
    task = task_class()
    task._dataset = self.dataset  # pylint: disable=protected-access
    return task

  def test_tasks_are_registered_with_expected_metadata(self):
    synthetic_to_real = self._task(
        doppelganger.DoppelgangerSyntheticToRealRetrieval
    )
    real_to_synthetic = self._task(
        doppelganger.DoppelgangerRealToSyntheticRetrieval
    )
    self.assertEqual(synthetic_to_real.metadata.main_score, 'MAP')
    self.assertEqual(real_to_synthetic.metadata.main_score, 'MAP')
    self.assertEqual(
        synthetic_to_real.sub_tasks,
        [
            'exact_source',
            'category',
        ],
    )

  def test_synthetic_to_real_inputs_and_documents(self):
    task = self._task(doppelganger.DoppelgangerSyntheticToRealRetrieval)
    inputs = list(task.multimodal_inputs())
    documents = list(task.documents())
    self.assertTrue(all(isinstance(sound, types.Sound) for sound in inputs))
    self.assertEqual(
        {sound.context.id for sound in inputs},
        {'synthetic:1', 'synthetic:2', 'synthetic:3'},
    )
    self.assertEqual(
        {sound.context.id for sound in documents},
        {'real:1', 'real:2', 'real:3'},
    )

  def test_real_to_synthetic_inputs_and_documents(self):
    task = self._task(doppelganger.DoppelgangerRealToSyntheticRetrieval)
    self.assertEqual(
        {sound.context.id for sound in task.multimodal_inputs()},
        {'real:1', 'real:2', 'real:3'},
    )
    self.assertEqual(
        {sound.context.id for sound in task.documents()},
        {'synthetic:1', 'synthetic:2', 'synthetic:3'},
    )

  def test_exact_source_relevance(self):
    task = self._task(doppelganger.DoppelgangerSyntheticToRealRetrieval)
    references = {
        example.sound_id: example.reference_id
        for example in task.examples('exact_source')
    }
    self.assertEqual(references['synthetic:1'], 'real:1')
    self.assertEqual(references['synthetic:3'], 'real:3')

  def test_category_relevance(self):
    task = self._task(doppelganger.DoppelgangerRealToSyntheticRetrieval)
    references = {
        example.sound_id: set(example.reference_id)
        for example in task.examples('category')
    }
    self.assertEqual(references['real:1'], {'synthetic:1', 'synthetic:2'})
    self.assertEqual(references['real:2'], {'synthetic:1', 'synthetic:2'})
    self.assertEqual(references['real:3'], {'synthetic:3'})

  def test_unknown_sub_task(self):
    task = self._task(doppelganger.DoppelgangerSyntheticToRealRetrieval)
    with self.assertRaisesRegex(ValueError, 'Unknown Doppelganger sub-task'):
      list(task.examples('unknown'))


if __name__ == '__main__':
  absltest.main()
