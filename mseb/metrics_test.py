# Copyright 2025 The MSEB Authors.
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


class MetricsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="reference_found",
          reference="reference",
          predicted_neighbors=["reference", "neighbor1"],
          expected_reciprocal_rank=1.0,
      ),
      dict(
          testcase_name="reference_not_found",
          reference="reference",
          predicted_neighbors=["neighbor1", "neighbor2"],
          expected_reciprocal_rank=0.0,
      ),
      dict(
          testcase_name="reference_at_2",
          reference="reference",
          predicted_neighbors=["neighbor1", "reference"],
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
          testcase_name="exact_match",
          reference="reference",
          predicted_neighbors=["reference", "neighbor1"],
          expected_exact_match=1.0,
      ),
      dict(
          testcase_name="not_exact_match_but_in_neighbors",
          reference="reference",
          predicted_neighbors=["neighbor1", "reference"],
          expected_exact_match=0.0,
      ),
      dict(
          testcase_name="not_exact_match_and_not_in_neighbors",
          reference="reference",
          predicted_neighbors=["neighbor1", "neighbor2"],
          expected_exact_match=0.0,
      ),
      dict(
          testcase_name="no_neighbors",
          reference="reference",
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
        truth="This is a test.",
        hypothesis="This is a test.",
    )
    self.assertEqual(word_errors, 0.0)
    self.assertEqual(word_errors_weight, 4.0)

  def test_compute_word_errors_incorrect(self):
    word_errors, word_errors_weight = metrics.compute_word_errors(
        truth="This is a test.",
        hypothesis="This is another test.",
    )
    self.assertEqual(word_errors, 1.0)
    self.assertEqual(word_errors_weight, 4.0)

  def test_compute_word_errors_empty_transcript(self):
    word_errors, word_errors_weight = metrics.compute_word_errors(
        truth="This is a test.",
        hypothesis="",
    )
    self.assertEqual(word_errors, 4.0)
    self.assertEqual(word_errors_weight, 4.0)


if __name__ == "__main__":
  absltest.main()
