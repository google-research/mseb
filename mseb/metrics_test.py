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


if __name__ == "__main__":
  absltest.main()
