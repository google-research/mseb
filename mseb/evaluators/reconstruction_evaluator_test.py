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

import math

from absl.testing import absltest
from mseb import types
from mseb.evaluators import reconstruction_evaluator
import numpy as np


class ReconstructionEvaluatorTest(absltest.TestCase):
  """Tests for the ReconstructionEvaluator class."""

  def setUp(self):
    super().setUp()
    self.evaluator = reconstruction_evaluator.ReconstructionEvaluator()
    self.context = types.SoundContextParams(
        id='test_id',
        sample_rate=16000,
        length=16000
    )

    # Simple embedding data (2 examples, 3 frames each, 4 features)
    self.ref_emb1 = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32
    )
    self.pred_emb1 = np.array(
        [[1.1, 2.1, 3.1, 4.1], [5.1, 6.1, 7.1, 8.1], [9.1, 10.1, 11.1, 12.1]],
        dtype=np.float32,
    )

    self.ref_emb2 = np.array(
        [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]],
        dtype=np.float32,
    )
    self.pred_emb2 = np.array(
        [[15, 25, 35, 45], [55, 65, 75, 85], [95, 105, 115, 125]],
        dtype=np.float32,
    )

    self.references = {
        'ex1': types.SoundEmbedding(
            embedding=self.ref_emb1,
            timestamps=np.zeros((3, 2)),
            context=self.context,
        ),
        'ex2': types.SoundEmbedding(
            embedding=self.ref_emb2,
            timestamps=np.zeros((3, 2)),
            context=self.context,
        ),
    }
    self.predictions = {
        'ex1': types.SoundEmbedding(
            embedding=self.pred_emb1,
            timestamps=np.zeros((3, 2)),
            context=self.context,
        ),
        'ex2': types.SoundEmbedding(
            embedding=self.pred_emb2,
            timestamps=np.zeros((3, 2)),
            context=self.context,
        ),
    }

  def _assert_scores(
      self, scores: list[types.Score], expected_metrics: list[str]
  ):
    """Helper to check if the correct metrics are present."""
    self.assertNotEmpty(scores)
    found_metrics = {s.metric for s in scores}
    self.assertCountEqual(
        found_metrics,
        expected_metrics,
        msg=f'Expected {expected_metrics}, but got {found_metrics}',
    )
    print(f'SCORES ARE {scores}')
    for score in scores:
      self.assertIsInstance(score.value, float)
      is_valid = math.isnan(score.value) or score.value >= 0.0
      self.assertTrue(
          is_valid,
          msg=f'Score {score.metric} was negative: {score.value}'
      )

  def test_compute_metrics_calculates_all_scores(self):
    results = self.evaluator.compute_metrics(
        self.predictions,
        self.references
    )
    self._assert_scores(results, ['FAD', 'KAD', 'Embedding MSE'])

    # Verify MSE calculation manually
    # MSE ex1 = mean((0.1)^2 * 4 features * 3 frames) = mean(0.01 * 12) = 0.01
    # MSE ex2 = mean((5)^2 * 4 features * 3 frames) = mean(25 * 12) = 25
    # Avg MSE = (0.01 + 25) / 2 = 12.505
    mse_score = next(s for s in results if s.metric == 'Embedding MSE')
    self.assertAlmostEqual(mse_score.value, 12.505, places=5)

  def test_compute_metrics_handles_shape_mismatch_for_mse(self):
    # Modify one prediction to have a different number of frames
    pred_emb1_bad_shape = np.vstack([self.pred_emb1, self.pred_emb1])
    bad_predictions = {
        'ex1': types.SoundEmbedding(
            embedding=pred_emb1_bad_shape,
            timestamps=np.zeros((6, 2)),
            context=self.context,
        ),
        'ex2': self.predictions['ex2'],
    }

    with self.assertLogs(level='WARNING') as log:
      results = self.evaluator.compute_metrics(bad_predictions, self.references)

    # Check that the warning was logged
    self.assertIn('Skipping MSE calculation for example ex1', log.output[0])

    # FAD and KAD should still be calculated
    self._assert_scores(results, ['FAD', 'KAD', 'Embedding MSE'])

    # MSE should only be calculated based on the valid example (ex2)
    # MSE ex2 = 25.0
    mse_score = next(s for s in results if s.metric == 'Embedding MSE')
    self.assertAlmostEqual(mse_score.value, 25.0, places=5)

  def test_compute_metrics_handles_missing_predictions(self):
    predictions_missing = {'ex1': self.predictions['ex1']}  # Missing ex2

    with self.assertLogs(level='WARNING') as log:
      results = self.evaluator.compute_metrics(
          predictions_missing, self.references
      )

    self.assertIn('Missing prediction for example_id ex2', log.output[0])

    # Metrics should be calculated based only on the available example (ex1)
    self._assert_scores(results, ['FAD', 'KAD', 'Embedding MSE'])
    mse_score = next(s for s in results if s.metric == 'Embedding MSE')
    self.assertAlmostEqual(mse_score.value, 0.01, places=5)  # Only MSE for ex1

  def test_compute_metrics_returns_empty_on_no_matches(self):
    predictions_none = {'non_matching_id': self.predictions['ex1']}

    with self.assertLogs(level='ERROR') as log:
      results = self.evaluator.compute_metrics(
          predictions_none, self.references
      )

    self.assertIn('No matching predictions and references found', log.output[0])
    self.assertEqual(results, [])


if __name__ == '__main__':
  absltest.main()
