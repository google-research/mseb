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

"""Regression tests for pointwise latent-sequence distances."""

from absl.testing import absltest
from absl.testing import parameterized
from mseb import metrics
import numpy as np


class LpDistanceTest(parameterized.TestCase):

  @parameterized.parameters(1, 2, 3, np.inf)
  def test_norm_uses_every_element_not_matrix_operator_norm(self, order):
    a = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    b = np.array([[0.0, 1.0, 1.0], [2.0, 3.0, 4.0]])
    errors = np.abs(a - b)
    expected = (
        errors.max()
        if np.isinf(order)
        else (errors**order).sum() ** (1 / order)
    )
    before = a.copy(), b.copy()
    actual = metrics.compute_lp_norm(a, b, p=order)
    self.assertAlmostEqual(actual['raw_distance'], expected)
    self.assertEqual(actual['reference_length'], 2.0)
    np.testing.assert_array_equal(a, before[0])
    np.testing.assert_array_equal(b, before[1])

  @parameterized.parameters(1, 2, 3)
  def test_temporal_padding_preserves_the_pointwise_definition(self, order):
    short = np.array([[1.0, 2.0], [3.0, 4.0]])
    long = np.array([[0.0, 3.0], [2.0, 1.0], [3.0, 5.0]])
    padded = np.concatenate([short, np.zeros((1, 2))])
    expected = (np.abs(padded - long) ** order).sum() ** (1 / order)
    for a, b in [(short, long), (long, short)]:
      actual = metrics.compute_lp_norm(a, b, p=order)
      self.assertAlmostEqual(actual['raw_distance'], expected)
      self.assertEqual(actual['reference_length'], len(a))

  def test_orthogonal_errors_are_accumulated(self):
    actual = metrics.compute_lp_norm(np.eye(3), np.zeros((3, 3)))
    self.assertAlmostEqual(actual['raw_distance'], np.sqrt(3.0))

  @parameterized.parameters(1, 2, 3)
  def test_empty_sequences_and_identity(self, order):
    empty = np.empty((0, 2))
    self.assertEqual(
        metrics.compute_lp_norm(empty, empty, p=order)['raw_distance'], 0.0
    )
    sequence = np.array([[1.0, 2.0], [3.0, 4.0]])
    self.assertEqual(
        metrics.compute_lp_norm(sequence, sequence, p=order)['raw_distance'],
        0.0,
    )

  def test_embedding_dimensions_must_still_agree(self):
    with self.assertRaises(ValueError):
      metrics.compute_lp_norm(np.zeros((2, 2)), np.zeros((2, 3)))


if __name__ == '__main__':
  absltest.main()
