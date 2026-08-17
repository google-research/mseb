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

from unittest import mock
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

      def multimodal_inputs(self):
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

      def multimodal_inputs(self):
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

      def multimodal_inputs(self):
        raise NotImplementedError()

    task = MockTask(tau=0.1)
    task.setup()
    predictions = {
        "utt_1": types.SoundEmbedding(
            embedding=np.array(["dog", "cat"]),
            timestamps=np.array([[1.0, 2.0], [3.0, 4.0]]),
            scores=np.array([0.9, 0.8]),
            context=types.SoundContextParams(
                id="utt_1", sample_rate=16000, length=1
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
        # Invalid and missing result metrics (0%)
        segmentation_evaluator.invalid_result_rate(0.0),
        segmentation_evaluator.missing_result_rate(0.0),
        # Ranking metric (perfect ranking)
        segmentation_evaluator.mean_average_precision(1.0),
    ]

    self._assert_scores(results["test"], expected_scores)


def _make_context(sound_id: str) -> types.SoundContextParams:
  return types.SoundContextParams(id=sound_id, sample_rate=16000, length=16000)


class SegmentationSelectionTaskTest(absltest.TestCase):

  def test_multimodal_objects_for_setup_yields_terms(self):

    class TaskWithTerms(segmentation.SegmentationSelectionTask):

      sub_tasks = []

      def examples(self, sub_task):
        return []

      def multimodal_inputs(self):
        return []

      def salient_term_lists(self):
        return [
            (
                "list1",
                [
                    segmentation_evaluator.Segment("hello", 0.0, 1.0),
                    segmentation_evaluator.Segment("world", 1.0, 2.0),
                ],
            ),
            (
                "list2",
                [
                    segmentation_evaluator.Segment("foo", 0.0, 1.0),
                ],
            ),
        ]

    task = TaskWithTerms()
    objects = list(task.multimodal_objects_for_setup())
    self.assertLen(objects, 3)
    self.assertEqual(objects[0].text, "hello")
    self.assertEqual(objects[1].text, "world")
    self.assertEqual(objects[2].text, "foo")

  def test_setup_with_embeddings_cache(self):
    # When salient_term_lists is non-empty and embeddings_cache is provided,
    # setup should populate term_embeddings_by_sound_id from the cache.

    class TaskWithTerms(segmentation.SegmentationSelectionTask):

      sub_tasks = []

      def examples(self, sub_task):
        return []

      def multimodal_inputs(self):
        return []

      def salient_term_lists(self):
        return [
            (
                "sound_1",
                [
                    segmentation_evaluator.Segment("dog bark", 0.0, 1.0),
                ],
            ),
        ]

    task = TaskWithTerms()
    mock_embedding = types.SoundEmbedding(
        embedding=np.array([[1.0, 0.0]]),
        timestamps=np.array([[0.0, 1.0]]),
        context=_make_context("term_dog"),
    )
    embeddings_cache = {"dog bark": mock_embedding}
    with mock.patch.object(
        type(task),
        "embeddings_dir",
        new_callable=mock.PropertyMock,
        return_value="/tmp/test_cache",
    ), mock.patch(
        "mseb.tasks.segmentation.runner_lib.load_embeddings",
        side_effect=FileNotFoundError,
    ), mock.patch(
        "mseb.tasks.segmentation.runner_lib.save_embeddings",
    ):
      task.setup(embeddings_cache=embeddings_cache)

    self.assertIsNotNone(task._evaluator)

  def test_setup_without_cache_or_runner_logs_warning(self):
    # When salient_term_lists is non-empty but no cache, runner, or file
    # exists, setup should log a warning and set term_embeddings to None.

    class TaskWithTerms(segmentation.SegmentationSelectionTask):

      sub_tasks = []

      def examples(self, sub_task):
        return []

      def multimodal_inputs(self):
        return []

      def salient_term_lists(self):
        return [
            (
                "sound_1",
                [
                    segmentation_evaluator.Segment("dog bark", 0.0, 1.0),
                ],
            ),
        ]

    task = TaskWithTerms()
    with mock.patch.object(
        type(task),
        "embeddings_dir",
        new_callable=mock.PropertyMock,
        return_value="/tmp/test_cache",
    ), mock.patch(
        "mseb.tasks.segmentation.runner_lib.load_embeddings",
        side_effect=FileNotFoundError,
    ):
      task.setup()  # No runner or embeddings_cache provided.

    self.assertIsNotNone(task._evaluator)

  def test_setup_with_embeddings_cache_saves_embeddings(self):
    # Verify that setup saves embeddings when using embeddings_cache.

    class TaskWithTerms(segmentation.SegmentationSelectionTask):

      sub_tasks = []

      def examples(self, sub_task):
        return []

      def multimodal_inputs(self):
        return []

      def salient_term_lists(self):
        return [
            (
                "sound_1",
                [
                    segmentation_evaluator.Segment("dog bark", 0.0, 1.0),
                ],
            ),
        ]

    task = TaskWithTerms()
    mock_embedding = types.SoundEmbedding(
        embedding=np.array([[1.0, 0.0]]),
        timestamps=np.array([[0.0, 1.0]]),
        context=_make_context("term_dog"),
    )
    embeddings_cache = {"dog bark": mock_embedding}
    mock_save = mock.MagicMock()
    with mock.patch.object(
        type(task),
        "embeddings_dir",
        new_callable=mock.PropertyMock,
        return_value="/tmp/test_cache",
    ), mock.patch(
        "mseb.tasks.segmentation.runner_lib.load_embeddings",
        side_effect=FileNotFoundError,
    ), mock.patch(
        "mseb.tasks.segmentation.runner_lib.save_embeddings",
        mock_save,
    ):
      task.setup(embeddings_cache=embeddings_cache)

    mock_save.assert_called_once()


if __name__ == "__main__":
  absltest.main()
