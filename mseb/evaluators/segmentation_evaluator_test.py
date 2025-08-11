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
from mseb import types
from mseb.evaluators import segmentation_evaluator
import numpy as np


class SegmentationEvaluatorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.evaluator = segmentation_evaluator.SegmentationEvaluator(tau=0.1)
    self.sample_rate = 16000
    self.params = types.SoundContextParams(
        sample_rate=self.sample_rate,
        length=self.sample_rate * 5,
        sound_id="test",
    )

  def _assert_scores(
      self,
      scores: list[types.Score],
      expected_scores: list[types.Score]
  ):
    self.assertLen(scores, len(expected_scores))
    scores_map = {s.metric: s for s in scores}
    expected_scores_map = {s.metric: s for s in expected_scores}
    self.assertCountEqual(scores_map.keys(), expected_scores_map.keys())
    for metric, expected_score in expected_scores_map.items():
      actual_score = scores_map[metric]
      self.assertEqual(actual_score.metric, expected_score.metric)
      self.assertEqual(actual_score.description, expected_score.description)
      self.assertAlmostEqual(actual_score.value, expected_score.value)
      self.assertEqual(actual_score.min, expected_score.min)
      self.assertEqual(actual_score.max, expected_score.max)

  def test_evaluate_perfect_match(self):
    ref_ts = np.array([[0.5, 1.0], [2.0, 2.5]])
    ref_emb = np.array(["dog", "cat"])
    scores = self.evaluator.evaluate(
        waveform_embeddings=ref_emb,
        embedding_timestamps=ref_ts,
        params=self.params,
        reference_waveform_embeddings=ref_emb,
        reference_embedding_timestamps=ref_ts,
    )
    expected_scores = [
        types.Score(
            metric="TimestampsHits",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS["TimestampsHits"]
            ),
            value=2.0,
            min=0.0,
            max=2.0
        ),
        types.Score(
            metric="EmbeddingsHits",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS["EmbeddingsHits"]
            ),
            value=2.0,
            min=0.0,
            max=2.0
        ),
        types.Score(
            metric="TimestampsAndEmbeddingsHits",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS[
                    "TimestampsAndEmbeddingsHits"]
            ),
            value=2.0,
            min=0.0,
            max=2.0
        ),
        types.Score(
            metric="NumSegments",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS["NumSegments"]
            ),
            value=2.0,
            min=0.0,
            max=2.0
        ),
    ]
    self._assert_scores(scores, expected_scores)

  def test_evaluate_no_match(self):
    ref_ts = np.array([[0.5, 1.0]])
    ref_emb = np.array(["dog"])
    pred_ts = np.array([[3.0, 3.5]])
    pred_emb = np.array(["cat"])
    scores = self.evaluator.evaluate(
        waveform_embeddings=pred_emb,
        embedding_timestamps=pred_ts,
        params=self.params,
        reference_waveform_embeddings=ref_emb,
        reference_embedding_timestamps=ref_ts,
    )
    expected_scores = [
        types.Score(
            metric="TimestampsHits",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS["TimestampsHits"]
            ),
            value=0.0,
            min=0.0,
            max=1.0
        ),
        types.Score(
            metric="EmbeddingsHits",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS["EmbeddingsHits"]
            ),
            value=0.0,
            min=0.0,
            max=1.0
        ),
        types.Score(
            metric="TimestampsAndEmbeddingsHits",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS[
                    "TimestampsAndEmbeddingsHits"]
            ),
            value=0.0,
            min=0.0,
            max=1.0
        ),
        types.Score(
            metric="NumSegments",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS["NumSegments"]
            ),
            value=1.0,
            min=0.0,
            max=1.0
        ),
    ]
    self._assert_scores(scores, expected_scores)

  def test_evaluate_timestamp_hit_with_tolerance(self):
    ref_ts = np.array([[1.0, 2.0]])
    ref_emb = np.array(["sound"])
    pred_ts = np.array([[1.05, 1.95]])
    pred_emb = np.array(["different_sound"])
    scores = self.evaluator.evaluate(
        waveform_embeddings=pred_emb,
        embedding_timestamps=pred_ts,
        params=self.params,
        reference_waveform_embeddings=ref_emb,
        reference_embedding_timestamps=ref_ts,
        tau=0.1,
    )
    expected_scores = [
        types.Score(
            metric="TimestampsHits",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS["TimestampsHits"]
            ),
            value=1.0,
            min=0.0,
            max=1.0
        ),
        types.Score(
            metric="EmbeddingsHits",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS["EmbeddingsHits"]
            ),
            value=0.0,
            min=0.0,
            max=1.0
        ),
        types.Score(
            metric="TimestampsAndEmbeddingsHits",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS[
                    "TimestampsAndEmbeddingsHits"]
            ),
            value=0.0,
            min=0.0,
            max=1.0
        ),
        types.Score(
            metric="NumSegments",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS["NumSegments"]
            ),
            value=1.0,
            min=0.0,
            max=1.0
        ),
    ]
    self._assert_scores(scores, expected_scores)

  def test_evaluate_embedding_hit_only(self):
    ref_ts = np.array([[1.0, 2.0]])
    ref_emb = np.array(["sound"])
    pred_ts = np.array([[4.0, 4.5]])
    pred_emb = np.array(["sound"])
    scores = self.evaluator.evaluate(
        waveform_embeddings=pred_emb,
        embedding_timestamps=pred_ts,
        params=self.params,
        reference_waveform_embeddings=ref_emb,
        reference_embedding_timestamps=ref_ts,
    )
    expected_scores = [
        types.Score(
            metric="TimestampsHits",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS["TimestampsHits"]
            ),
            value=0.0,
            min=0.0,
            max=1.0
        ),
        types.Score(
            metric="EmbeddingsHits",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS["EmbeddingsHits"]
            ),
            value=1.0,
            min=0.0,
            max=1.0
        ),
        types.Score(
            metric="TimestampsAndEmbeddingsHits",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS[
                    "TimestampsAndEmbeddingsHits"]
            ),
            value=0.0,
            min=0.0,
            max=1.0
        ),
        types.Score(
            metric="NumSegments",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS["NumSegments"]
            ),
            value=1.0,
            min=0.0,
            max=1.0
        ),
    ]
    self._assert_scores(scores, expected_scores)

  def test_evaluate_fails_on_out_of_bounds_prediction(self):
    ref_ts = np.array([[1.0, 2.0]])
    ref_emb = np.array(["sound"])
    invalid_pred_ts = np.array([[4.0, 5.1]])
    with self.assertRaisesRegex(
        ValueError,
        "exceeds the total waveform length"
    ):
      self.evaluator.evaluate(
          waveform_embeddings=ref_emb,
          embedding_timestamps=invalid_pred_ts,
          params=self.params,
          reference_waveform_embeddings=ref_emb,
          reference_embedding_timestamps=ref_ts,
      )

  def test_evaluate_fails_on_out_of_bounds_reference(self):
    ref_ts = np.array([[6.0, 7.0]])
    ref_emb = np.array(["sound"])
    pred_ts = np.array([[1.0, 2.0]])
    with self.assertRaisesRegex(
        ValueError,
        "exceeds the total waveform length"
    ):
      self.evaluator.evaluate(
          waveform_embeddings=ref_emb,
          embedding_timestamps=pred_ts,
          params=self.params,
          reference_waveform_embeddings=ref_emb,
          reference_embedding_timestamps=ref_ts,
      )

  def test_evaluate_fails_on_negative_timestamp(self):
    ref_ts = np.array([[1.0, 2.0]])
    ref_emb = np.array(["sound"])
    invalid_pred_ts = np.array([[-0.1, 1.0]])
    with self.assertRaisesRegex(
        ValueError,
        "contains negative timestamps"
    ):
      self.evaluator.evaluate(
          waveform_embeddings=ref_emb,
          embedding_timestamps=invalid_pred_ts,
          params=self.params,
          reference_waveform_embeddings=ref_emb,
          reference_embedding_timestamps=ref_ts,
      )

  def test_evaluate_fails_on_invalid_start_end_order(self):
    ref_ts = np.array([[1.0, 2.0]])
    ref_emb = np.array(["sound"])
    invalid_pred_ts = np.array([[2.0, 1.0]])
    with self.assertRaisesRegex(
        ValueError,
        "start_time > end_time"
    ):
      self.evaluator.evaluate(
          waveform_embeddings=ref_emb,
          embedding_timestamps=invalid_pred_ts,
          params=self.params,
          reference_waveform_embeddings=ref_emb,
          reference_embedding_timestamps=ref_ts,
      )

  def test_combine_scores_aggregation(self):
    scores_example_1 = [
        types.Score(
            metric="TimestampsHits",
            description="desc",
            value=1.0,
            min=0.0,
            max=2.0
        ),
        types.Score(
            metric="EmbeddingsHits",
            description="desc",
            value=1.0,
            min=0.0,
            max=2.0
        ),
        types.Score(
            metric="TimestampsAndEmbeddingsHits",
            description="desc",
            value=0.0,
            min=0.0,
            max=2.0
        ),
        types.Score(
            metric="NumSegments",
            description="desc",
            value=2.0,
            min=0.0,
            max=2.0
        ),
    ]
    scores_example_2 = [
        types.Score(
            metric="TimestampsHits",
            description="desc",
            value=1.0,
            min=0.0,
            max=1.0
        ),
        types.Score(
            metric="EmbeddingsHits",
            description="desc",
            value=0.0,
            min=0.0,
            max=1.0
        ),
        types.Score(
            metric="TimestampsAndEmbeddingsHits",
            description="desc",
            value=0.0,
            min=0.0,
            max=1.0
        ),
        types.Score(
            metric="NumSegments",
            description="desc",
            value=1.0,
            min=0.0,
            max=1.0
        ),
    ]

    final_scores = self.evaluator.combine_scores(
        [scores_example_1, scores_example_2]
    )

    expected_scores = [
        types.Score(
            metric="TimestampsHits",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS["TimestampsHits"]
            ),
            value=2.0,
            min=0.0,
            max=3.0
        ),
        types.Score(
            metric="EmbeddingsHits",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS["EmbeddingsHits"]
            ),
            value=1.0,
            min=0.0,
            max=3.0
        ),
        types.Score(
            metric="TimestampsAndEmbeddingsHits",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS[
                    "TimestampsAndEmbeddingsHits"]
            ),
            value=0.0,
            min=0.0,
            max=3.0
        ),
        types.Score(
            metric="NumSegments",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS["NumSegments"]
            ),
            value=3.0,
            min=0.0,
            max=3.0
        ),
        types.Score(
            metric="TimestampsAccuracy",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS[
                    "TimestampsAccuracy"
                ]
            ),
            value=2.0 / 3.0,
            min=0.0,
            max=1.0
        ),
        types.Score(
            metric="EmbeddingsAccuracy",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS[
                    "EmbeddingsAccuracy"
                ]
            ),
            value=1.0 / 3.0,
            min=0.0,
            max=1.0
        ),
        types.Score(
            metric="TimestampsAndEmbeddingAccuracy",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS[
                    "TimestampsAndEmbeddingAccuracy"]
            ),
            value=0.0,
            min=0.0,
            max=1.0
        ),
    ]
    self._assert_scores(final_scores, expected_scores)

  def test_combine_scores_empty_input(self):
    final_scores = self.evaluator.combine_scores([])
    self.assertEqual(final_scores, [])

  def test_combine_scores_no_segments(self):
    scores_no_segments = [[
        types.Score(
            metric="NumSegments",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS["NumSegments"]
            ),
            value=0.0,
            min=0.0,
            max=0.0
        )
    ]]
    final_scores = self.evaluator.combine_scores(scores_no_segments)
    expected_scores = [
        types.Score(
            metric="TimestampsHits",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS["TimestampsHits"]
            ),
            value=0.0,
            min=0.0,
            max=0.0
        ),
        types.Score(
            metric="EmbeddingsHits",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS["EmbeddingsHits"]
            ),
            value=0.0,
            min=0.0,
            max=0.0
        ),
        types.Score(
            metric="TimestampsAndEmbeddingsHits",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS[
                    "TimestampsAndEmbeddingsHits"
                ]
            ),
            value=0.0,
            min=0.0,
            max=0.0
        ),
        types.Score(
            metric="NumSegments",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS["NumSegments"]
            ),
            value=0.0,
            min=0.0,
            max=0.0
        ),
        types.Score(
            metric="TimestampsAccuracy",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS[
                    "TimestampsAccuracy"
                ]
            ),
            value=0.0,
            min=0.0,
            max=1.0
        ),
        types.Score(
            metric="EmbeddingsAccuracy",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS[
                    "EmbeddingsAccuracy"
                ]
            ),
            value=0.0,
            min=0.0,
            max=1.0
        ),
        types.Score(
            metric="TimestampsAndEmbeddingAccuracy",
            description=(
                segmentation_evaluator._METRIC_DESCRIPTIONS[
                    "TimestampsAndEmbeddingAccuracy"
                ]
            ),
            value=0.0,
            min=0.0,
            max=1.0
        ),
    ]
    self._assert_scores(final_scores, expected_scores)


if __name__ == "__main__":
  absltest.main()
