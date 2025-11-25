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

from typing import Iterable
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from mseb import runner as runner_lib
from mseb import types
from mseb.evaluators import classification_evaluator
from mseb.tasks import classification
import numpy as np

NO_RESPONSE_STR = classification_evaluator.NO_RESPONSE_STR
INVALID_ANSWER_STR = classification_evaluator.INVALID_ANSWER_STR


class ClassificationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.cache_dir = self.create_tempdir().full_path
    # Set the TASK_CACHE_BASEPATH flag to use our temporary directory
    self.enter_context(
        flagsaver.flagsaver(
            (classification.task.TASK_CACHE_BASEPATH, self.cache_dir)
        )
    )

  def test_classification_task_setup_with_runner(self):

    class MockTask(classification.ClassificationTask):
      @property
      def task_type(self) -> str:
        return "multi_class"

      def sounds(self) -> Iterable[types.Sound]:
        raise NotImplementedError()

      def class_labels(self) -> Iterable[str]:
        return ["label_1", "label_2", "label_3"]

      def examples(self, sub_task: str) -> Iterable[types.Sound]:
        raise NotImplementedError()

      @property
      def sub_tasks(self) -> list[str]:
        return ["test"]

    # This mock simulates a text encoder that returns fixed-size embeddings
    mock_encoder = mock.Mock()
    mock_encoder.encode.return_value = [
        types.TextEmbedding(
            embedding=np.zeros((1, 10)),
            spans=np.zeros((1, 2)),
            context=types.TextContextParams(id="mock"),
        )
    ] * 3

    task = MockTask()
    task.setup(runner=runner_lib.DirectRunner(encoder=mock_encoder))

    self.assertIsNotNone(task._evaluator)
    self.assertIsInstance(
        task._evaluator, classification_evaluator.ClassificationEvaluator
    )
    self.assertEqual(task._evaluator.top_k_value, 5)
    self.assertIsNotNone(task._evaluator.weights)
    # 3 classes, 10 embedding dims + 1 bias dim
    self.assertEqual(task._evaluator.weights.shape, (3, 11))

  def test_classification_task_compute_scores_multi_class(self):

    class MockMultiClassTask(classification.ClassificationTask):
      @property
      def task_type(self) -> str:
        return "multi_class"

      def sounds(self) -> Iterable[types.Sound]:
        raise NotImplementedError()

      def class_labels(self) -> Iterable[str]:
        return ["label_1", "label_2"]

      def examples(self, sub_task: str):
        return [
            classification_evaluator.ClassificationReference(
                example_id="utt_1", label_id="label_1"
            ),
            classification_evaluator.ClassificationReference(
                example_id="utt_2", label_id="label_2"
            ),
        ]

      @property
      def sub_tasks(self) -> list[str]:
        return ["test"]

    # Manually save weights to the cache for the task to load
    weights = np.array([[1.0, 0.0, 0.1], [0.0, 1.0, 0.1]])
    task = MockMultiClassTask()
    classification_evaluator.save_linear_classifier(
        task.class_labels(), weights, task.weights_dir
    )
    task.setup()  # Should load the weights we just saved

    embeddings = {
        "utt_1": types.SoundEmbedding(
            context=types.SoundContextParams(
                id="utt_1",
                sample_rate=16000,
                length=1
            ),
            embedding=np.array([[1.0, 0.0]]),
            timestamps=np.zeros((1, 2)),
        ),
        "utt_2": types.SoundEmbedding(
            context=types.SoundContextParams(
                id="utt_2",
                sample_rate=16000,
                length=1
            ),
            embedding=np.array([[0.0, 1.0]]),
            timestamps=np.zeros((1, 2)),
        ),
    }

    scores = task.compute_scores(embeddings=embeddings)
    self.assertIn("test", scores)
    accuracy_score = next(s for s in scores["test"] if s.metric == "Accuracy")
    # utt_1:
    # emb[1,0] * w1[1,0] = 1. emb * w2[0,1] = 0. Pred=label_1 (correct)
    # utt_2:
    # emb[0,1] * w2[0,1] = 1. emb * w1[1,0] = 0. Pred=label_2 (correct)
    self.assertAlmostEqual(accuracy_score.value, 1.0)

  def test_classification_task_compute_scores_multi_label(self):

    class MockMultiLabelTask(classification.ClassificationTask):
      @property
      def task_type(self) -> str:
        return "multi_label"

      def sounds(self) -> Iterable[types.Sound]:
        raise NotImplementedError()

      def class_labels(self) -> Iterable[str]:
        return ["label_1", "label_2"]

      def examples(self, sub_task: str):
        return [
            classification_evaluator.MultiLabelClassificationReference(
                example_id="utt_1", label_ids=["label_1", "label_2"]
            )
        ]

      @property
      def sub_tasks(self) -> list[str]:
        return ["test"]

    weights = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    task = MockMultiLabelTask(
        multi_label_threshold=0.4
    )
    classification_evaluator.save_linear_classifier(
        task.class_labels(), weights, task.weights_dir
    )
    task.setup()

    embeddings = {
        "utt_1": types.SoundEmbedding(
            context=types.SoundContextParams(
                id="utt_1",
                sample_rate=16000,
                length=1
            ),
            embedding=np.array([[0.5, 0.5]]),
            timestamps=np.zeros((1, 2)),
        ),
    }

    scores = task.compute_scores(embeddings=embeddings)
    self.assertIn("test", scores)
    map_score = next(s for s in scores["test"] if s.metric == "mAP")
    # Since scores will be high for both correct labels, mAP should be 1.0
    self.assertAlmostEqual(map_score.value, 1.0)

  def test_create_weights_from_runner(self):
    class MockTask(classification.ClassificationTask):
      @property
      def task_type(self) -> str:
        return "multi_class"

      def sounds(self) -> Iterable[types.Sound]:
        raise NotImplementedError()

      def class_labels(self) -> Iterable[str]:
        return ["cat", "dog"]

      def examples(self, sub_task: str):
        raise NotImplementedError()

      @property
      def sub_tasks(self) -> list[str]:
        return ["test"]

    mock_encoder = mock.Mock()

    # Create a side_effect function that returns the correct embedding
    # based on the input text it receives.
    def encode_side_effect(text_inputs: list[types.Text]):
      embedding_map = {
          "cat": types.TextEmbedding(
              embedding=np.array([[0.1, 0.2]]),
              spans=np.zeros((1, 2)),
              context=types.TextContextParams(id="cat"),
          ),
          "dog": types.TextEmbedding(
              embedding=np.array([[0.3, 0.4]]),
              spans=np.zeros((1, 2)),
              context=types.TextContextParams(id="dog"),
          ),
      }
      # This handles batch or single-item calls from the runner.
      return [embedding_map[t.text] for t in text_inputs]

    mock_encoder.encode.side_effect = encode_side_effect

    task = MockTask()
    runner = runner_lib.DirectRunner(encoder=mock_encoder)
    class_labels, weights = task._create_weights_from_runner(runner)

    self.assertEqual(class_labels, ("cat", "dog"))
    # Shape should be (num_classes, embedding_dim + bias_dim)
    self.assertEqual(weights.shape, (2, 3))

    np.testing.assert_allclose(
        weights,
        np.array([[0.1, 0.2, 0.0], [0.3, 0.4, 0.0]])
    )

    # Verify that the weights were saved to the cache.
    loaded_labels, loaded_weights = (
        classification_evaluator.load_linear_classifier(task.weights_dir)
    )
    self.assertEqual(tuple(loaded_labels), class_labels)
    np.testing.assert_allclose(loaded_weights, weights)

  def test_compute_scores(self):

    class MockMultiClassTask(classification.ClassificationTask):
      @property
      def task_type(self) -> str:
        return "multi_class"

      def sounds(self) -> Iterable[types.Sound]:
        raise NotImplementedError()

      def class_labels(self) -> Iterable[str]:
        return ["label_1", "label_2"]

      def examples(self, sub_task: str):
        return [
            classification_evaluator.ClassificationReference(
                example_id="utt_1", label_id="label_1"
            ),
            classification_evaluator.ClassificationReference(
                example_id="utt_2", label_id="label_2"
            ),
        ]

      @property
      def sub_tasks(self) -> list[str]:
        return ["test"]

    # Setup the task with a dummy evaluator.
    task = MockMultiClassTask()
    weights = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    task._evaluator = classification_evaluator.ClassificationEvaluator(
        class_labels=list(task.class_labels()), weights=weights
    )

    # Mock predictions where the correct class has the highest score.
    embeddings = {
        "utt_1": types.SoundEmbedding(
            embedding=np.array([[0.9, 0.1]]),  # Correctly predicts label_1
            context=types.SoundContextParams(
                id="utt_1",
                sample_rate=16000,
                length=1,
            ),
            timestamps=np.zeros((1, 2)),
        ),
        "utt_2": types.SoundEmbedding(
            embedding=np.array([[0.2, 0.8]]),  # Correctly predicts label_2
            context=types.SoundContextParams(
                id="utt_2",
                sample_rate=16000,
                length=1,
            ),
            timestamps=np.zeros((1, 2)),
        ),
    }

    metrics = task.compute_scores(embeddings)
    self.assertIn("test", metrics)
    accuracy_score = next(
        s for s in metrics["test"] if s.metric == "Accuracy"
    )
    self.assertAlmostEqual(accuracy_score.value, 1.0)

  def test_compute_scores_with_prediction_output(self):

    class MockMultiClassTask(classification.ClassificationTask):
      @property
      def task_type(self) -> str:
        return "multi_class"

      def sounds(self) -> Iterable[types.Sound]:
        raise NotImplementedError()

      def class_labels(self) -> Iterable[str]:
        return ["label_1", "label_2"]

      def examples(self, sub_task: str):
        return [
            classification_evaluator.ClassificationReference(
                example_id="utt_1", label_id="label_1"
            ),
            classification_evaluator.ClassificationReference(
                example_id="utt_2", label_id="label_2"
            ),
            classification_evaluator.ClassificationReference(
                example_id="utt_3", label_id="label_1"
            ),
            classification_evaluator.ClassificationReference(
                example_id="utt_4", label_id="label_2"
            ),
        ]

      @property
      def sub_tasks(self) -> list[str]:
        return ["test"]

    # Setup the task with a dummy evaluator.
    task = MockMultiClassTask()
    weights = None
    task._evaluator = classification_evaluator.ClassificationEvaluator(
        class_labels=list(task.class_labels()), weights=weights
    )

    # Mock predictions where the correct class has the highest score.
    embeddings = {
        "utt_1": types.TextPrediction(
            context=types.PredictionContextParams(id="utt_1"),
            prediction="label_1",
        ),
        "utt_2": types.TextPrediction(
            context=types.PredictionContextParams(id="utt_2"),
            prediction="label_2",
        ),
        "utt_3": types.TextPrediction(
            context=types.PredictionContextParams(id="utt_3"),
            prediction=NO_RESPONSE_STR,
        ),
        "utt_4": types.TextPrediction(
            context=types.PredictionContextParams(id="utt_4"),
            prediction=INVALID_ANSWER_STR,
        ),
    }

    metrics = task.compute_scores(embeddings)
    self.assertIn("test", metrics)
    accuracy_score = next(
        s for s in metrics["test"] if s.metric == "Accuracy"
    )
    self.assertAlmostEqual(accuracy_score.value, 0.5)
    invalid_rate_score = next(
        s for s in metrics["test"] if s.metric == "InvalidResultRate"
    )
    self.assertAlmostEqual(invalid_rate_score.value, 0.25)
    no_result_rate_score = next(
        s for s in metrics["test"] if s.metric == "MissingResultRate"
    )
    self.assertAlmostEqual(no_result_rate_score.value, 0.25)

if __name__ == "__main__":
  absltest.main()
