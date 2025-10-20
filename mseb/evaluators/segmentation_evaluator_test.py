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
    self.context = types.SoundContextParams(
        id="test_id", sample_rate=16000, length=16000 * 10
    )

  def _assert_scores(
      self, scores: list[types.Score],
      expected_scores: list[types.Score]
  ):
    """Helper to compare the final list of Score objects."""
    self.assertLen(scores, len(expected_scores))
    scores_map = {s.metric: s for s in scores}
    expected_scores_map = {s.metric: s for s in expected_scores}
    self.assertCountEqual(scores_map.keys(), expected_scores_map.keys())
    for metric, expected_score in expected_scores_map.items():
      actual_score = scores_map[metric]
      self.assertEqual(actual_score.metric, expected_score.metric)
      self.assertAlmostEqual(
          actual_score.value,
          expected_score.value,
          places=6,
          msg=f"Failed on metric: {metric}",
      )
      self.assertEqual(actual_score.min, expected_score.min)
      self.assertEqual(actual_score.max, expected_score.max)

  def test_compute_scores_perfect_match(self):
    ref_segments = [
        segmentation_evaluator.Segment("dog", 0.5, 1.0),
        segmentation_evaluator.Segment("cat", 2.0, 2.5),
    ]
    references = [
        segmentation_evaluator.SegmentationReference("ex1", ref_segments)
    ]
    predictions = {
        "ex1": types.SoundEmbedding(
            embedding=np.array(["dog", "cat"]),
            timestamps=np.array([[0.5, 1.0], [2.0, 2.5]]),
            scores=np.array([0.9, 0.8]),
            context=self.context,
        )
    }

    result = self.evaluator.compute_scores(predictions, references)
    scores = result.per_example_scores[0]

    self.assertEqual(scores.num_reference_segments, 2)
    self.assertEqual(scores.timestamps_and_embeddings_hits, 2)
    self.assertEqual(scores.timestamps_hits, 2)
    self.assertEqual(scores.embeddings_hits, 2)
    self.assertEqual(scores.ndcg, 1.0)
    self.assertEqual(scores.edit_distance, 0.0)
    self.assertEqual(scores.num_reference_words, 2)
    self.assertLen(result.all_predictions_for_map, 2)

  def test_compute_scores_imperfect_sequence_and_match(self):
    ref_segments = [
        segmentation_evaluator.Segment("dog", 0.5, 1.0),
        segmentation_evaluator.Segment("cat", 2.0, 2.5),
    ]
    references = [
        segmentation_evaluator.SegmentationReference("ex1", ref_segments)
    ]
    predictions = {
        "ex1": types.SoundEmbedding(
            embedding=np.array(["cat", "bird", "dog"]),
            timestamps=np.array([[2.05, 2.55], [3.0, 3.5], [5.0, 5.5]]),
            scores=np.array([0.9, 0.8, 0.7]),
            context=self.context,
        )
    }

    result = self.evaluator.compute_scores(predictions, references)
    scores = result.per_example_scores[0]

    self.assertEqual(scores.timestamps_and_embeddings_hits, 1)
    self.assertEqual(scores.timestamps_hits, 1)
    self.assertEqual(scores.embeddings_hits, 1 + 1)
    self.assertLess(scores.ndcg, 1.0)
    self.assertGreater(scores.edit_distance, 0.0)
    self.assertEqual(scores.num_reference_words, 2)

  def test_compute_scores_raises_type_error(self):
    references = [
        segmentation_evaluator.SegmentationReference(
            "ex1", [segmentation_evaluator.Segment("a", 1, 2)]
        )
    ]
    predictions = {
        "ex1": types.TextEmbedding(
            embedding=np.array(["a"]),
            spans=np.array([[0, 1]]),
            context=types.TextContextParams(id="ex1")
        )
    }
    with self.assertRaisesRegex(TypeError, "expected a SoundEmbedding"):
      self.evaluator.compute_scores(predictions, references)

  def test_compute_scores_raises_value_error_for_mismatched_lengths(self):
    references = [
        segmentation_evaluator.SegmentationReference(
            "ex1", [segmentation_evaluator.Segment("a", 1, 2)]
        )
    ]
    predictions = {
        "ex1": types.SoundEmbedding(
            embedding=np.array(["a"]),
            timestamps=np.array([[1.0, 2.0], [3.0, 4.0]]),
            context=self.context,
        )
    }
    with self.assertRaisesRegex(ValueError, "Inconsistent lengths"):
      self.evaluator.compute_scores(predictions, references)

  def test_compute_metrics_full_aggregation(self):
    result = segmentation_evaluator.SegmentationScoringResult(
        per_example_scores=[
            segmentation_evaluator.SegmentationScores(
                timestamps_hits=1,
                embeddings_hits=2,
                timestamps_and_embeddings_hits=1,
                num_reference_segments=2,
                ndcg=0.5,
                edit_distance=2,
                num_reference_words=2,
            ),
            segmentation_evaluator.SegmentationScores(
                timestamps_hits=1,
                embeddings_hits=1,
                timestamps_and_embeddings_hits=1,
                num_reference_segments=1,
                ndcg=1.0,
                edit_distance=0,
                num_reference_words=2,
            ),
        ],
        all_predictions_for_map=[
            ("ex1", segmentation_evaluator.Segment("a", 1, 2, 0.9)),
            ("ex1", segmentation_evaluator.Segment("b", 3, 4, 0.8)),
            ("ex2", segmentation_evaluator.Segment("c", 5, 6, 0.7)),
        ],
        ground_truths_for_map={
            "ex1": [segmentation_evaluator.Segment("a", 1.05, 2.05)],
            "ex2": [segmentation_evaluator.Segment("c", 5.05, 6.05)],
        },
    )

    final_scores = self.evaluator.compute_metrics(result)

    # Expected values
    # Accuracies:
    #   total_refs=3, total_ts_hits=2, total_emb_hits=3, total_ts_emb_hits=2
    # Sequence: mean_ndcg=0.75, mean_edit_distance=1.0
    # mAP: y_true=[1, 0, 1], y_score=[0.9, 0.8, 0.7] ->
    # AP is calculated from this.
    # AP for [1,0,1] at scores [.9,.8,.7] ->
    # P = [1/1, 1/2, 2/3], R = [1/2, 1/2, 2/2] ->
    # (1*1/2) + (2/3*1/2) = 1/2 + 1/3 = 5/6
    expected_scores = [
        segmentation_evaluator.timestamps_and_embeddings_hits(2.0, 3.0),
        segmentation_evaluator.timestamps_hits(2.0, 3.0),
        segmentation_evaluator.embeddings_hits(3.0, 3.0),
        segmentation_evaluator.num_segments(3.0),
        segmentation_evaluator.timestamps_and_embeddings_accuracy(2.0 / 3.0),
        segmentation_evaluator.timestamps_accuracy(2.0 / 3.0),
        segmentation_evaluator.embeddings_accuracy(3.0 / 3.0),
        segmentation_evaluator.normalized_discounted_cumulative_gain(0.75),
        segmentation_evaluator.word_error_rate(0.5),
        segmentation_evaluator.mean_average_precision(5.0 / 6.0),
    ]
    self._assert_scores(final_scores, expected_scores)

  def test_compute_metrics_empty_input(self):
    empty_result = segmentation_evaluator.SegmentationScoringResult(
        per_example_scores=[],
        all_predictions_for_map=[],
        ground_truths_for_map={},
    )
    final_scores = self.evaluator.compute_metrics(empty_result)
    self.assertLen(final_scores, 1)
    self.assertEqual(final_scores[0].metric, "mAP")
    self.assertEqual(final_scores[0].value, 0.0)

  def test_compute_scores_handles_empty_reference_list(self):
    # The ground truth has no segments.
    references = [
        segmentation_evaluator.SegmentationReference("ex1", [])
    ]
    # The model predicts two false positives.
    predictions = {
        "ex1": types.SoundEmbedding(
            embedding=np.array(["false", "positive"]),
            timestamps=np.array([[1.0, 2.0], [3.0, 4.0]]),
            scores=np.array([0.9, 0.8]),
            context=self.context,
        )
    }
    result = self.evaluator.compute_scores(predictions, references)
    self.assertLen(result.per_example_scores, 1)
    scores = result.per_example_scores[0]
    self.assertEqual(scores.num_reference_segments, 0)
    self.assertEqual(scores.timestamps_and_embeddings_hits, 0)
    # Sequence metrics should reflect failure.
    self.assertEqual(scores.ndcg, 0.0)
    # The two predicted words are pure insertion errors.
    self.assertEqual(scores.edit_distance, 2)
    self.assertEqual(scores.num_reference_words, 0)

  def test_compute_scores_handles_reference_with_empty_strings(self):
    # The ground truth has a segment, but its label is an empty string.
    ref_segments = [segmentation_evaluator.Segment("", 1.0, 2.0)]
    references = [
        segmentation_evaluator.SegmentationReference("ex1", ref_segments)
    ]
    # The model predicts one false positive.
    predictions = {
        "ex1": types.SoundEmbedding(
            embedding=np.array(["prediction"]),
            timestamps=np.array([[3.0, 4.0]]),
            scores=np.array([0.9]),
            context=self.context,
        )
    }
    result = self.evaluator.compute_scores(predictions, references)
    self.assertLen(result.per_example_scores, 1)
    scores = result.per_example_scores[0]
    self.assertEqual(scores.num_reference_segments, 1)
    self.assertEqual(scores.timestamps_and_embeddings_hits, 0)
    self.assertEqual(scores.ndcg, 0.0)
    self.assertEqual(scores.edit_distance, 1)
    self.assertEqual(scores.num_reference_words, 0)

if __name__ == "__main__":
  absltest.main()
