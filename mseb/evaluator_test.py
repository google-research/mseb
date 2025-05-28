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
from mseb import evaluator
import numpy.testing as npt


class EvaluatorTest(absltest.TestCase):

  def test_compute_weighted_average_and_std(self):
    metrics = evaluator.compute_weighted_average_and_std(
        scores=[
            {
                'height_example': 1.0,
                'width_example': 3.0,
                'width_example_weight': 1.0,
            },
            {
                'height_example': 5.0,
                'width_example': 2.0,
                'width_example_weight': 2.0,
            },
            {
                'height_example': 3.0,
                'width_example': 1.0,
                'width_example_weight': 3.0,
            },
        ],
        statistic_metric_pairs=[
            ('height_example', 'height'),
            ('width_example', 'width'),
        ],
    )
    npt.assert_equal(len(metrics), 4)
    npt.assert_equal(metrics['height'], 3.0)
    npt.assert_equal(metrics['height_std']**2, 8 / 3)
    npt.assert_equal(metrics['width'], 5 / 3)
    npt.assert_almost_equal(metrics['width_std']**2, 5 / 9)


if __name__ == '__main__':
  absltest.main()
