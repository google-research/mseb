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

"""Tests for the HuthLab brain encoding task."""

import os

from absl.testing import absltest
import joblib
from mseb import types
from mseb.datasets import huthlab_fmri
from mseb.evaluators import brain_encoding_evaluator
from mseb.tasks.brain_encodings import huthlab
import numpy as np


def _create_mock_data(base_path: str, subject: str = 'S03'):
  """Creates a minimal mock data directory for testing.

  Sets up mock fMRI responses as a dict keyed by story name, matching the
  format expected by HuthLabDataset.load_responses().

  Args:
    base_path: Root path to create mock data.
    subject: Subject identifier.

  Returns:
    A dictionary containing mock data parameters and generated stories.
  """
  os.makedirs(os.path.join(base_path, 'responses'), exist_ok=True)
  os.makedirs(os.path.join(base_path, 'stimuli'), exist_ok=True)

  n_voxels = 5
  n_trs_per_story = 100

  train_stories = list(huthlab_fmri.TRAIN_STORIES[subject])
  test_stories = list(huthlab_fmri.TEST_STORIES[subject])
  all_stories = train_stories + test_stories

  # Response file is a dict: story_name -> (T, V) array.
  np.random.seed(42)
  resp_dict = {}
  for story in all_stories:
    resp_dict[story] = np.random.randn(n_trs_per_story, n_voxels).astype(
        np.float32
    )

  # File is named UT{subject}_responses.jbl.
  joblib.dump(
      resp_dict,
      os.path.join(base_path, 'responses', f'UT{subject}_responses.jbl'),
  )

  return {
      'n_voxels': n_voxels,
      'n_trs_per_story': n_trs_per_story,
      'train_stories': train_stories,
      'test_stories': test_stories,
      'resp_dict': resp_dict,
  }


class HuthLabBrainEncodingTest(absltest.TestCase):
  """Tests for the HuthLabBrainEncoding task."""

  def setUp(self):
    super().setUp()
    self.testdata_dir = self.create_tempdir()
    self.base_path = self.testdata_dir.full_path
    self.mock_data = _create_mock_data(self.base_path)

  def _create_task(self, **kwargs):
    """Creates a task with only the test subject."""
    return huthlab.HuthLabBrainEncoding(
        base_path=self.base_path,
        subjects=('S03',),
        **kwargs,
    )

  # --- Metadata tests ---

  def test_metadata(self):
    self.assertIsNotNone(huthlab.HuthLabBrainEncoding.metadata)
    self.assertEqual(
        huthlab.HuthLabBrainEncoding.metadata.name, 'HuthLabBrainEncoding'
    )
    self.assertEqual(
        huthlab.HuthLabBrainEncoding.metadata.type, 'BrainEncoding'
    )
    self.assertEqual(
        huthlab.HuthLabBrainEncoding.metadata.main_score, 'MeanCorrelation'
    )

  def test_metadata_scores(self):
    scores = huthlab.HuthLabBrainEncoding.metadata.scores
    metric_names = {s.metric for s in scores}
    self.assertIn('MeanCorrelation', metric_names)
    self.assertIn('MedianCorrelation', metric_names)
    self.assertIn('FractionSignificant', metric_names)

  def test_metadata_dataset(self):
    ds = huthlab.HuthLabBrainEncoding.metadata.dataset
    self.assertEqual(ds.name, 'HuthLabScalingLaws')

  # --- Sub-tasks tests ---

  def test_sub_tasks(self):
    task = self._create_task()
    self.assertEqual(task.sub_tasks, ['S03'])

  def test_sub_tasks_multiple_subjects(self):
    task = huthlab.HuthLabBrainEncoding(
        base_path=self.base_path,
        subjects=('S01', 'S02', 'S03'),
    )
    self.assertEqual(task.sub_tasks, ['S01', 'S02', 'S03'])

  # --- multimodal_inputs tests ---

  def test_multimodal_inputs_covers_all_stories(self):
    task = self._create_task()
    inputs = list(task.multimodal_inputs())
    input_ids = [inp.context.id for inp in inputs]
    for story in self.mock_data['train_stories']:
      self.assertIn(story, input_ids)
    for story in self.mock_data['test_stories']:
      self.assertIn(story, input_ids)

  def test_multimodal_inputs_are_text(self):
    task = self._create_task()
    inputs = list(task.multimodal_inputs())
    for inp in inputs:
      self.assertIsInstance(inp, types.Text)
      self.assertEqual(inp.text, inp.context.id)
      self.assertEqual(inp.context.language_name, 'English')

  def test_multimodal_inputs_no_duplicates(self):
    task = self._create_task()
    inputs = list(task.multimodal_inputs())
    input_ids = [inp.context.id for inp in inputs]
    self.assertEqual(len(input_ids), len(set(input_ids)))

  # --- train/test examples tests ---

  def test_train_examples_count(self):
    task = self._create_task()
    examples = task.train_examples('S03')
    self.assertLen(examples, len(self.mock_data['train_stories']))

  def test_test_examples_count(self):
    task = self._create_task()
    examples = task.test_examples('S03')
    self.assertLen(examples, len(self.mock_data['test_stories']))

  def test_train_examples_sound_ids(self):
    task = self._create_task()
    examples = task.train_examples('S03')
    sound_ids = [ex.sound_id for ex in examples]
    self.assertEqual(sound_ids, self.mock_data['train_stories'])

  def test_test_examples_sound_ids(self):
    task = self._create_task()
    examples = task.test_examples('S03')
    sound_ids = [ex.sound_id for ex in examples]
    self.assertEqual(sound_ids, list(self.mock_data['test_stories']))

  def test_train_examples_are_brain_encoding_examples(self):
    task = self._create_task()

    examples = task.train_examples('S03')
    for ex in examples:
      self.assertIsInstance(ex, brain_encoding_evaluator.BrainEncodingExample)

  def test_train_examples_trim_values(self):
    """train_examples uses trim_start=10, trim_end=-5."""
    task = self._create_task()
    examples = task.train_examples('S03')
    for ex in examples:
      self.assertEqual(ex.tr_index_start, 10)
      self.assertEqual(ex.tr_index_end, -5)

  def test_test_examples_trim_values(self):
    """test_examples uses trim_start=50, trim_end=-5."""
    task = self._create_task()
    examples = task.test_examples('S03')
    for ex in examples:
      self.assertEqual(ex.tr_index_start, 50)
      self.assertEqual(ex.tr_index_end, -5)

  # --- fMRI data tests ---

  def test_fmri_train_shape(self):
    task = self._create_task()
    resp = task.fmri_train('S03')
    n_train = len(self.mock_data['train_stories'])
    expected_trs = n_train * self.mock_data['n_trs_per_story']
    self.assertEqual(resp.shape, (expected_trs, self.mock_data['n_voxels']))

  def test_fmri_test_shape(self):
    task = self._create_task()
    resp = task.fmri_test('S03')
    n_test = len(self.mock_data['test_stories'])
    expected_trs = n_test * (self.mock_data['n_trs_per_story'] - 40)
    self.assertEqual(resp.shape, (expected_trs, self.mock_data['n_voxels']))

  def test_fmri_train_content(self):
    task = self._create_task()
    resp = task.fmri_train('S03')
    expected = np.vstack([
        self.mock_data['resp_dict'][s] for s in self.mock_data['train_stories']
    ])
    np.testing.assert_array_equal(resp, expected)

  def test_fmri_test_content(self):
    task = self._create_task()
    resp = task.fmri_test('S03')
    expected = np.vstack([
        self.mock_data['resp_dict'][s][40:]
        for s in self.mock_data['test_stories']
    ])
    np.testing.assert_array_equal(resp, expected)


if __name__ == '__main__':
  absltest.main()
