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

from absl.testing import absltest
from absl.testing import parameterized
from mseb import metrics
import numpy as np


class MetricsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='reference_found',
          reference='reference',
          predicted_neighbors=['reference', 'neighbor1'],
          expected_reciprocal_rank=1.0,
      ),
      dict(
          testcase_name='reference_not_found',
          reference='reference',
          predicted_neighbors=['neighbor1', 'neighbor2'],
          expected_reciprocal_rank=0.0,
      ),
      dict(
          testcase_name='reference_at_2',
          reference='reference',
          predicted_neighbors=['neighbor1', 'reference'],
          expected_reciprocal_rank=1 / 2,
      ),
  )
  def test_compute_reciprocal_rank(
      self, reference, predicted_neighbors, expected_reciprocal_rank
  ):
    reciprocal_rank = metrics.compute_reciprocal_rank(
        reference=reference, predicted_neighbors=predicted_neighbors
    )
    self.assertEqual(reciprocal_rank, expected_reciprocal_rank)

  @parameterized.named_parameters(
      dict(
          testcase_name='exact_match',
          reference='reference',
          predicted_neighbors=['reference', 'neighbor1'],
          expected_exact_match=1.0,
      ),
      dict(
          testcase_name='not_exact_match_but_in_neighbors',
          reference='reference',
          predicted_neighbors=['neighbor1', 'reference'],
          expected_exact_match=0.0,
      ),
      dict(
          testcase_name='not_exact_match_and_not_in_neighbors',
          reference='reference',
          predicted_neighbors=['neighbor1', 'neighbor2'],
          expected_exact_match=0.0,
      ),
      dict(
          testcase_name='no_neighbors',
          reference='reference',
          predicted_neighbors=[],
          expected_exact_match=0.0,
      ),
  )
  def test_compute_exact_match(
      self, reference, predicted_neighbors, expected_exact_match
  ):
    exact_match = metrics.compute_exact_match(
        reference=reference, predicted_neighbors=predicted_neighbors
    )
    self.assertEqual(exact_match, expected_exact_match)

  def test_compute_word_errors_correct(self):
    word_errors, word_errors_weight = metrics.compute_word_errors(
        truth='This is a test.',
        hypothesis='This is a test.',
    )
    self.assertEqual(word_errors, 0.0)
    self.assertEqual(word_errors_weight, 4.0)

  def test_compute_word_errors_incorrect(self):
    word_errors, word_errors_weight = metrics.compute_word_errors(
        truth='This is a test.',
        hypothesis='This is another test.',
    )
    self.assertEqual(word_errors, 1.0)
    self.assertEqual(word_errors_weight, 4.0)

  def test_compute_word_errors_empty_transcript(self):
    word_errors, word_errors_weight = metrics.compute_word_errors(
        truth='This is a test.',
        hypothesis='',
    )
    self.assertEqual(word_errors, 4.0)
    self.assertEqual(word_errors_weight, 4.0)

  def test_compute_unit_edit_distance(self):
    truth = [1, 2, 3, 4]
    hypo = [1, 5, 3]  # 1 substitution (2->5), 1 deletion (4)
    res = metrics.compute_unit_edit_distance(truth, hypo)

    self.assertEqual(res['raw_distance'], 2.0)
    self.assertEqual(res['substitutions'], 1.0)
    self.assertEqual(res['deletions'], 1.0)
    self.assertEqual(res['insertions'], 0.0)
    self.assertEqual(res['normalized_distance'], 0.5)
    self.assertEqual(res['reference_length'], 4.0)

  def test_compute_lp_norm_valid(self):
    z1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    z2 = np.array([[1.0, 0.0], [0.0, 2.0]])
    res = metrics.compute_lp_norm(z1, z2, p=2)
    # Check both the value and the metadata
    self.assertEqual(res['raw_distance'], 1.0)
    self.assertEqual(res['reference_length'], 2.0)

  def test_compute_lp_norm_mismatch(self):
    z1 = np.ones((5, 10))
    z2 = np.ones((6, 10))
    with self.assertRaises(ValueError):
      metrics.compute_lp_norm(z1, z2)

  def test_compute_dynamic_time_warping_distance_alignment(self):
    z1 = np.array([[1.0, 0.0], [0.0, 0.0]])
    z2 = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    # Should match the repeated first frame with 0 cost
    res = metrics.compute_dynamic_time_warping_distance(z1, z2)
    self.assertEqual(res['raw_distance'], 0.0)
    self.assertEqual(res['reference_length'], 2.0)

  def test_compute_continuous_edit_distance_scaling_max_dist(self):
    # Vectors on opposite sides of the origin
    z1 = np.array([[50.0, 0.0]])
    z2 = np.array([[-50.0, 0.0]])
    # Scaled to [1,0] and [-1,0] -> Euclidean dist is 2.0
    res = metrics.compute_continuous_edit_distance(
        z1,
        z2,
        unit_sphere_scaling=True
    )
    self.assertAlmostEqual(res['raw_distance'], 2.0)
    self.assertEqual(res['reference_length'], 1.0)
    # Normalized by 2*L: 2.0 / (2*1) = 1.0
    self.assertAlmostEqual(res['normalized_distance'], 1.0)

  def test_compute_continuous_edit_distance_temporal_penalty(self):
    z1 = np.array([[1.0, 0.0]])
    z2 = np.array([[1.0, 0.0], [1.0, 0.0]])
    # Match first frame (0 cost) + Insert second frame (w_ins cost)
    res = metrics.compute_continuous_edit_distance(z1, z2, w_ins=2.0)
    self.assertAlmostEqual(res['raw_distance'], 2.0)
    self.assertEqual(res['reference_length'], 1.0)

  def test_compute_continuous_edit_distance_empty_input(self):
    z1 = np.zeros((0, 2))
    z2 = np.zeros((0, 2))
    res = metrics.compute_continuous_edit_distance(z1, z2)
    self.assertEqual(res['raw_distance'], 0.0)
    self.assertEqual(res['reference_length'], 0.0)


if __name__ == '__main__':
  absltest.main()
