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
from mseb.evaluators import clustering_evaluator
import numpy as np


class ClusterKmeansTest(absltest.TestCase):

  def test_is_deterministic(self):
    data = np.random.default_rng(seed=0).normal(size=(100, 8))
    labels = clustering_evaluator.cluster_kmeans(
        data, nlabels=5, batch_size=32
    )
    for _ in range(3):
      np.testing.assert_array_equal(
          clustering_evaluator.cluster_kmeans(data, nlabels=5, batch_size=32),
          labels,
      )


if __name__ == '__main__':
  absltest.main()
