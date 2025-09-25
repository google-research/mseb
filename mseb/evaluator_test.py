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

from unittest import mock

from absl.testing import absltest
from mseb import evaluator
from mseb import types
import numpy as np
import numpy.testing as npt


class MockEvaluator(evaluator.SoundEmbeddingEvaluator):
  """A concrete implementation of the evaluator for testing purposes."""

  def evaluate(
      self,
      waveform_embeddings: np.ndarray,
      embedding_timestamps: np.ndarray,
      params: types.SoundContextParams,
      **kwargs,
  ) -> list[types.Score]:
    """A mock implementation of the evaluate method."""
    return [
        types.Score(
            metric="mock_metric",
            value=0.5,
            description="A mock score.",
            min=0,
            max=1,
        )
    ]

  def combine_scores(
      self, scores_per_example: list[list[types.Score]]
  ) -> list[types.Score]:
    """A mock implementation of the score combination method."""
    if not scores_per_example:
      return []
    return scores_per_example[0]


class SoundEmbeddingEvaluatorTest(absltest.TestCase):

  def test_init_stores_kwargs(self):
    mock_evaluator = MockEvaluator(metric_param="test_value")
    self.assertIn("metric_param", mock_evaluator._kwargs)
    self.assertEqual(mock_evaluator._kwargs["metric_param"], "test_value")

  def test_abstract_class_cannot_be_instantiated(self):
    with self.assertRaises(TypeError):
      evaluator.SoundEmbeddingEvaluator()

  def test_concrete_class_can_be_instantiated(self):
    try:
      _ = MockEvaluator()
    except TypeError:
      self.fail("MockEvaluator failed to instantiate.")

  def test_default_evaluate_batch_calls_evaluate_serially(self):
    mock_evaluator = MockEvaluator()
    batch_size = 3

    waveform_embeddings = np.zeros((10, 16))
    embedding_timestamps = np.zeros((10, 2))
    params = types.SoundContextParams(
        sample_rate=16000, length=80000, id="test"
    )

    encoder_outputs_batch = [
        (waveform_embeddings, embedding_timestamps)
    ] * batch_size
    params_batch = [params] * batch_size

    with mock.patch.object(
        mock_evaluator, "evaluate", wraps=mock_evaluator.evaluate
    ) as spy_evaluate:
      results = mock_evaluator.evaluate_batch(
          encoder_outputs_batch,
          params_batch
      )

      self.assertIsInstance(results, list)
      self.assertLen(results, batch_size)
      self.assertIsInstance(results[0], list)
      self.assertIsInstance(results[0][0], types.Score)

      self.assertEqual(spy_evaluate.call_count, batch_size)

      spy_evaluate.assert_any_call(
          waveform_embeddings=waveform_embeddings,
          embedding_timestamps=embedding_timestamps,
          params=params,
      )


class EvaluatorTest(absltest.TestCase):

  def test_compute_weighted_average_and_std(self):
    mean, std = evaluator.compute_weighted_average_and_std(
        values=[
            types.WeightedValue(value=1.0),
            types.WeightedValue(value=5.0),
            types.WeightedValue(value=3.0),
        ],
    )
    npt.assert_almost_equal(mean, 3.0)
    npt.assert_almost_equal(std**2, 8 / 3)

  def test_compute_weighted_average_and_std_with_weights(self):
    mean, std = evaluator.compute_weighted_average_and_std(
        values=[
            types.WeightedValue(value=3.0),
            types.WeightedValue(value=2.0, weight=2.0),
            types.WeightedValue(value=1.0, weight=3.0),
        ],
    )
    npt.assert_almost_equal(mean, 5 / 3)
    npt.assert_almost_equal(std**2, 5 / 9)

  def test_argmax(self):
    scores = np.array([2, 8, 5, 0])
    top_score, top_id = evaluator.argmax(scores)
    npt.assert_almost_equal(top_score, np.array([8]))
    npt.assert_equal(top_id, np.array([1]))

  def test_top_k(self):
    scores = np.array([2, 8, 5, 0])
    top_k_scores, top_k_ids = evaluator.top_k(scores, k=3)
    npt.assert_almost_equal(top_k_scores, np.array([8, 5, 2]))
    npt.assert_equal(top_k_ids, np.array([1, 2, 0]))

  def test_top_k_not_enough_scores(self):
    scores = np.array([2, 8, 5, 0])
    top_k_scores, top_k_ids = evaluator.top_k(scores, k=10)
    npt.assert_almost_equal(top_k_scores, np.array([8, 5, 2, 0]))
    npt.assert_equal(top_k_ids, np.array([1, 2, 0, 3]))


if __name__ == "__main__":
  absltest.main()
