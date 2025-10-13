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
from mseb.tasks import segmentation
import numpy as np


class SegmentationTaskTest(absltest.TestCase):

  def _assert_scores(
      self, scores: list[types.Score], expected_scores: list[types.Score]
  ):
    """Helper to compare the final list of Score objects."""
    # This helper is unchanged and is now used in the main test.
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

  def test_setup_initializes_evaluator(self):

    class MockTask(segmentation.SegmentationTask):

      sub_tasks = []

      def examples(self, sub_task):
        return []

      def sounds(self):
        return []

    task = MockTask(tau=0.25)
    task.setup()
    self.assertIsInstance(
        task._evaluator, segmentation_evaluator.SegmentationEvaluator
    )
    self.assertEqual(task._evaluator.tau, 0.25)

  def test_compute_scores_raises_error_if_not_setup(self):

    class MockTask(segmentation.SegmentationTask):

      sub_tasks = []

      def examples(self, sub_task):
        return []

      def sounds(self):
        return []

    task = MockTask()
    with self.assertRaisesRegex(ValueError, "Evaluator is not initialized"):
      task.compute_scores(embeddings={})

  def test_compute_scores_runs_full_pipeline(self):

    class MockTask(segmentation.SegmentationTask):

      @property
      def sub_tasks(self) -> list[str]:
        return ["test"]

      def examples(self, sub_task: str):
        if sub_task == "test":
          return [
              segmentation_evaluator.SegmentationReference(
                  example_id="utt_1",
                  segments=[
                      segmentation_evaluator.Segment("dog", 1.0, 2.0),
                      segmentation_evaluator.Segment("cat", 3.0, 4.0),
                  ],
              )
          ]
        return []

      def sounds(self):
        raise NotImplementedError()

    task = MockTask(tau=0.1)
    task.setup()
    predictions = {
        "utt_1": types.SoundEmbedding(
            embedding=np.array(["dog", "cat"]),
            timestamps=np.array([[1.0, 2.0], [3.0, 4.0]]),
            scores=np.array([0.9, 0.8]),
            context=types.SoundContextParams(
                id="utt_1",
                sample_rate=16000,
                length=1
            ),
        )
    }
    results = task.compute_scores(predictions)
    self.assertIn("test", results)
    expected_scores = [
        # Accuracy metrics (100%)
        segmentation_evaluator.timestamps_and_embeddings_hits(2.0, 2.0),
        segmentation_evaluator.timestamps_hits(2.0, 2.0),
        segmentation_evaluator.embeddings_hits(2.0, 2.0),
        segmentation_evaluator.num_segments(2.0),
        segmentation_evaluator.timestamps_and_embeddings_accuracy(1.0),
        segmentation_evaluator.timestamps_accuracy(1.0),
        segmentation_evaluator.embeddings_accuracy(1.0),
        # Order metrics (perfect order)
        segmentation_evaluator.normalized_discounted_cumulative_gain(1.0),
        segmentation_evaluator.word_error_rate(0.0),
        # Ranking metric (perfect ranking)
        segmentation_evaluator.mean_average_precision(1.0),
    ]

    self._assert_scores(results["test"], expected_scores)


if __name__ == "__main__":
  absltest.main()
