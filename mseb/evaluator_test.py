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


if __name__ == "__main__":
  absltest.main()
