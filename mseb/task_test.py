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

  def _encode_batch(self, sound, **kwargs):
    pass

  def setup(self):
    self._model_loaded = True


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
            metric="mock_metric", value=0.5, description="desc", min=0, max=1
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
    for i in range(self._dataset_size):
      yield types.Sound(
          waveform=np.zeros(16000),
          context=types.SoundContextParams(
              sample_rate=16000, length=16000, sound_id=str(i)
          ),
      )


class MSEBTaskTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_encoder = MockSoundEncoder("mock/path")

  def test_list_tasks(self):
    tasks = task.get_name_to_task()
    self.assertIn("mock_task", tasks)


if __name__ == "__main__":
  absltest.main()
