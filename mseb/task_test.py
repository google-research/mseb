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

from typing import Any
from unittest import mock

from absl.testing import absltest
from mseb import encoder
from mseb import evaluator
from mseb import task
from mseb import types
import numpy as np


MOCK_TASK_METADATA = types.TaskMetadata(
    name="mock_task",
    description="A task for testing.",
    reference="http://example.com/mock_task",
    type="SpeechRecognition",
    category="speech",
    main_score="mock_metric",
    revision="1.0.0",
    dataset=types.Dataset(
        path="/path/to/mock_dataset",
        revision="1.0.0"
    ),
    scores=[
        types.Score(
            metric="mock_metric",
            value=0.0,
            description="A mock metric for testing.",
            min=0,
            max=1,
        )
    ],
    eval_splits=["test", "validation"],
    eval_langs=["en-US"],
    domains=["mock_domain"],
    task_subtypes=["testing"],
)


class MockSoundEncoder(encoder.SoundEncoder):
  """A minimal, concrete encoder for instantiating tasks in tests."""

  def _encode(self, audio, params, **kwargs):
    pass

  def setup(self):
    self._model_loaded = True

  def _encode_single(
      self,
      waveform,
      params,
      **kwargs
  ) -> tuple[np.ndarray, np.ndarray]:
    return (np.zeros((5, 16)), np.zeros((5, 2)))


class MockSoundEmbeddingEvaluator(evaluator.SoundEmbeddingEvaluator):
  """A minimal, concrete evaluator for instantiating tasks in tests."""

  def evaluate(
      self,
      waveform_embeddings,
      embedding_timestamps,
      params,
      **kwargs
  ) -> list[types.Score]:
    return [
        types.Score(
            metric="mock_metric",
            value=0.5,
            description="desc",
            min=0,
            max=1
        )
    ]

  def combine_scores(self, scores_per_example) -> list[types.Score]:
    return [
        types.Score(
            metric="aggregated",
            value=0.55,
            description="aggregated desc",
            min=0,
            max=1,
        )
    ]


class MockTask(task.MSEBTask):
  """A concrete task for testing the base class orchestration logic."""

  metadata = MOCK_TASK_METADATA
  evaluator_cls = MockSoundEmbeddingEvaluator

  def __init__(
      self,
      sound_encoder: encoder.SoundEncoder,
      evaluator_kwargs: dict[str, Any] | None = None,
      dataset_size: int = 5,
  ):
    super().__init__(sound_encoder, evaluator_kwargs)
    self._dataset_size = dataset_size

  def load_data(self):
    for _ in range(self._dataset_size):
      yield (np.zeros(16000), types.SoundContextParams(
          sample_rate=16000, length=16000))


class MSEBTaskTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_encoder = MockSoundEncoder("mock/path")

  def test_init_success(self):
    mock_task = MockTask(sound_encoder=self.mock_encoder)
    self.assertIsInstance(mock_task.encoder, MockSoundEncoder)
    self.assertIsInstance(mock_task.evaluator, MockSoundEmbeddingEvaluator)

  def test_init_fails_if_metadata_is_missing(self):

    class BadTask(task.MSEBTask):
      evaluator_cls = MockSoundEmbeddingEvaluator

      def load_data(self):
        yield (np.zeros(1), types.SoundContextParams(
            sample_rate=16000, length=1))

    with self.assertRaisesRegex(NotImplementedError, "metadata"):
      BadTask(self.mock_encoder)

  def test_init_fails_if_evaluator_cls_is_missing(self):

    class BadTask(task.MSEBTask):
      metadata = MOCK_TASK_METADATA

      def load_data(self):
        yield (np.zeros(1), types.SoundContextParams(
            sample_rate=16000, length=1))

    with self.assertRaisesRegex(NotImplementedError, "evaluator_cls"):
      BadTask(self.mock_encoder)

  def test_setup_delegates_to_encoder(self):
    with mock.patch.object(
        self.mock_encoder,
        "setup",
        autospec=True
    ) as spy_setup:
      mock_task = MockTask(sound_encoder=self.mock_encoder)
      mock_task.setup()
      spy_setup.assert_called_once()

  def test_load_batched_data_logic(self):
    mock_task = MockTask(
        sound_encoder=self.mock_encoder,
        dataset_size=10
    )
    batches = list(mock_task.load_batched_data(batch_size=4))
    self.assertLen(batches, 3)  # 10 items, batch size 4 -> [4, 4, 2]
    self.assertLen(batches[0], 4)
    self.assertLen(batches[1], 4)
    self.assertLen(batches[2], 2)

  def test_run_orchestrates_pipeline_correctly(self):
    dataset_size = 5
    batch_size = 2
    num_batches = 3  # 5 items, batch size 2 -> 3 batches
    mock_task = MockTask(
        sound_encoder=self.mock_encoder,
        dataset_size=dataset_size
    )

    mock_task.setup = mock.MagicMock()
    mock_task.encoder.encode_batch = mock.MagicMock(
        return_value=[(np.zeros(1), np.zeros(1))] * 2
    )
    mock_task.evaluator.evaluate_batch = mock.MagicMock(
        return_value=[[types.Score("m", "d", 0.0, 0, 1)]] * 2
    )
    mock_task.evaluator.combine_scores = mock.MagicMock(
        return_value=[types.Score("agg", "ad", 1.0, 0, 1)]
    )

    final_scores = mock_task.run(batch_size=batch_size)

    mock_task.setup.assert_called_once()
    self.assertEqual(mock_task.encoder.encode_batch.call_count, num_batches)
    self.assertEqual(mock_task.evaluator.evaluate_batch.call_count, num_batches)
    mock_task.evaluator.combine_scores.assert_called_once()
    self.assertEqual(final_scores["mock_task"][0].metric, "agg")

  def test_run_with_empty_dataset(self):
    mock_task = MockTask(
        sound_encoder=self.mock_encoder,
        dataset_size=0
    )
    mock_task.evaluator.combine_scores = mock.MagicMock()

    with self.assertLogs(level="WARNING") as cm:
      final_scores = mock_task.run()
      self.assertIn("Warning: No scores were generated", cm.output[0])

    self.assertEqual(final_scores, {"mock_task": []})
    mock_task.evaluator.combine_scores.assert_not_called()

  def test_list_tasks(self):
    tasks = task.get_name_to_task()
    self.assertIn("mock_task", tasks)


if __name__ == "__main__":
  absltest.main()
