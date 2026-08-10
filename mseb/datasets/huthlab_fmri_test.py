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

"""Tests for huthlab_fmri dataset loader."""

import os

from absl.testing import absltest
import joblib
from mseb.datasets import huthlab_fmri
import numpy as np


class HuthLabDatasetTest(absltest.TestCase):
  """Tests for the HuthLabDataset class."""

  def setUp(self):
    super().setUp()
    self.testdata_dir = self.create_tempdir()
    self.base_path = self.testdata_dir.full_path

    # Create directory structure.
    os.makedirs(os.path.join(self.base_path, 'responses'))
    os.makedirs(os.path.join(self.base_path, 'stimuli'))

    self.n_voxels = 10
    self.n_trs_per_story = 50

    # The on-disk .jbl file is a dict mapping story_name -> (T, V) array.
    # File is named UT{subject}_responses.jbl (e.g. UTS03_responses.jbl).
    np.random.seed(42)
    self.mock_resp = {}
    for story in list(huthlab_fmri.TRAIN_STORIES['S03']) + list(
        huthlab_fmri.TEST_STORIES['S03']
    ):
      self.mock_resp[story] = np.random.randn(
          self.n_trs_per_story, self.n_voxels
      ).astype(np.float32)

    joblib.dump(
        self.mock_resp,
        os.path.join(self.base_path, 'responses', 'UTS03_responses.jbl'),
    )

  def _make_dataset(self, **kwargs):
    return huthlab_fmri.HuthLabDataset(base_path=self.base_path, **kwargs)

  # --- Init tests ---

  def test_init_with_base_path(self):
    dataset = self._make_dataset()
    self.assertEqual(dataset.base_path, self.base_path)
    self.assertEqual(dataset.subject, 'S03')

  def test_init_with_custom_subject(self):
    dataset = self._make_dataset(subject='S01')
    self.assertEqual(dataset.subject, 'S01')

  def test_init_default_tr_duration(self):
    dataset = self._make_dataset()
    self.assertEqual(dataset.tr_duration, huthlab_fmri.DEFAULT_TR_DURATION)

  def test_init_custom_tr_duration(self):
    dataset = self._make_dataset(tr_duration=1.5)
    self.assertEqual(dataset.tr_duration, 1.5)

  def test_init_requires_base_path(self):
    with self.assertRaises(ValueError):
      huthlab_fmri.HuthLabDataset(base_path=None)

  # --- load_responses tests ---

  def test_load_responses_returns_dict(self):
    dataset = self._make_dataset()
    resp = dataset.load_responses()
    self.assertIsInstance(resp, dict)

  def test_load_responses_contains_all_stories(self):
    dataset = self._make_dataset()
    resp = dataset.load_responses()
    all_stories = list(huthlab_fmri.TRAIN_STORIES['S03']) + list(
        huthlab_fmri.TEST_STORIES['S03']
    )
    for story in all_stories:
      self.assertIn(story, resp)

  def test_load_responses_values_are_arrays(self):
    dataset = self._make_dataset()
    resp = dataset.load_responses()
    for _, arr in resp.items():
      self.assertIsInstance(arr, np.ndarray)
      self.assertEqual(arr.shape, (self.n_trs_per_story, self.n_voxels))

  def test_load_responses_matches_mock(self):
    dataset = self._make_dataset()
    resp = dataset.load_responses()
    for story in self.mock_resp:
      np.testing.assert_array_equal(resp[story], self.mock_resp[story])

  def test_load_responses_cached(self):
    """The underlying data is loaded only once (self._resp is cached)."""
    dataset = self._make_dataset()
    resp1 = dataset.load_responses()
    resp2 = dataset.load_responses()
    self.assertIs(resp1, resp2)

  # --- load_train_responses / load_test_responses tests ---

  def test_load_train_responses_shape(self):
    dataset = self._make_dataset()
    resp = dataset.load_train_responses()
    train_stories = dataset.get_train_stories()
    expected_trs = len(train_stories) * self.n_trs_per_story
    self.assertEqual(resp.shape, (expected_trs, self.n_voxels))

  def test_load_train_responses_values(self):
    dataset = self._make_dataset()
    resp = dataset.load_train_responses()
    train_stories = dataset.get_train_stories()
    expected = np.vstack([self.mock_resp[s] for s in train_stories])
    np.testing.assert_array_equal(resp, expected)

  def test_load_test_responses_shape(self):
    dataset = self._make_dataset()
    resp = dataset.load_test_responses()
    test_stories = dataset.get_test_stories()
    # load_test_responses slices [40:] from each story.
    trs_per_test_story = self.n_trs_per_story - 40
    expected_trs = len(test_stories) * trs_per_test_story
    self.assertEqual(resp.shape, (expected_trs, self.n_voxels))

  def test_load_test_responses_values(self):
    dataset = self._make_dataset()
    resp = dataset.load_test_responses()
    test_stories = dataset.get_test_stories()
    # load_test_responses slices [40:] from each story.
    expected = np.vstack([self.mock_resp[s][40:] for s in test_stories])
    np.testing.assert_array_equal(resp, expected)

  # --- Story metadata tests ---

  def test_get_train_stories_s03(self):
    dataset = self._make_dataset()
    train_stories = dataset.get_train_stories()
    self.assertIs(train_stories, huthlab_fmri.TRAIN_STORIES['S03'])
    self.assertIn('adollshouse', train_stories)

  def test_get_train_stories_s01(self):
    dataset = self._make_dataset(subject='S01')
    train_stories = dataset.get_train_stories()
    self.assertIs(train_stories, huthlab_fmri.TRAIN_STORIES['S01'])

  def test_get_train_stories_per_subject(self):
    """TRAIN_STORIES is keyed by subject; S02 and S03 share the same list."""
    self.assertIs(
        huthlab_fmri.TRAIN_STORIES['S02'],
        huthlab_fmri.TRAIN_STORIES['S03'],
    )
    self.assertIsNot(
        huthlab_fmri.TRAIN_STORIES['S01'],
        huthlab_fmri.TRAIN_STORIES['S02'],
    )

  def test_s02_has_extra_stories(self):
    """S02/S03 have stories not in S01."""
    s02_only = set(huthlab_fmri.TRAIN_STORIES['S02']) - set(
        huthlab_fmri.TRAIN_STORIES['S01']
    )
    self.assertNotEmpty(s02_only)

  def test_get_test_stories_s03(self):
    dataset = self._make_dataset()
    test_stories = dataset.get_test_stories()
    self.assertIs(test_stories, huthlab_fmri.TEST_STORIES['S03'])
    self.assertIn('wheretheressmoke', test_stories)

  def test_get_test_stories_all_subjects_same(self):
    """All subjects share the same test stories."""
    for subject in huthlab_fmri.SUBJECTS:
      self.assertEqual(
          huthlab_fmri.TEST_STORIES[subject],
          ('wheretheressmoke', 'onapproachtopluto', 'fromboyhoodtofatherhood'),
      )

  def test_train_test_no_overlap(self):
    """Train and test stories should not overlap for any subject."""
    for subject in huthlab_fmri.SUBJECTS:
      train = set(huthlab_fmri.TRAIN_STORIES[subject])
      test = set(huthlab_fmri.TEST_STORIES[subject])
      self.assertEmpty(train & test)

  # --- Audio path tests ---

  def test_get_audio_path(self):
    dataset = self._make_dataset()
    path = dataset.get_audio_path('wheretheressmoke')
    expected = os.path.join(self.base_path, 'stimuli', 'wheretheressmoke.wav')
    self.assertEqual(path, expected)

  # --- Constants tests ---

  def test_default_parameters(self):
    self.assertEqual(huthlab_fmri.DEFAULT_TR_DURATION, 2.0)
    self.assertEqual(huthlab_fmri.DEFAULT_DELAYS, (1, 2, 3, 4))

  def test_subjects(self):
    self.assertEqual(huthlab_fmri.SUBJECTS, ('S01', 'S02', 'S03'))

  def test_train_stories_is_dict(self):
    self.assertIsInstance(huthlab_fmri.TRAIN_STORIES, dict)
    for subject in huthlab_fmri.SUBJECTS:
      self.assertIn(subject, huthlab_fmri.TRAIN_STORIES)

  def test_test_stories_is_dict(self):
    self.assertIsInstance(huthlab_fmri.TEST_STORIES, dict)
    for subject in huthlab_fmri.SUBJECTS:
      self.assertIn(subject, huthlab_fmri.TEST_STORIES)


if __name__ == '__main__':
  absltest.main()
