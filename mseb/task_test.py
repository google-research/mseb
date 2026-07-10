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

from absl.testing import absltest
from mseb import encoder
from mseb import evaluator
from mseb import task as task_lib
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
    dataset=types.Dataset(path="/path/to/mock_dataset", revision="1.0.0"),
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


class MockSoundEncoder(encoder.MultiModalEncoder):
  """A minimal, concrete encoder for instantiating tasks in tests."""

  def _check_input_types(self, batch):
    pass

  def _encode(self, sound_batch):
    pass

  def _setup(self):
    pass


class MockSoundEmbeddingEvaluator(evaluator.SoundEmbeddingEvaluator):
  """A minimal, concrete evaluator for instantiating tasks in tests."""

  def evaluate(
      self, waveform_embeddings, embedding_timestamps, params, **kwargs
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


class MockTask(task_lib.MSEBTask):
  """A concrete task for testing the base class orchestration logic."""

  metadata = MOCK_TASK_METADATA
  evaluator_cls = MockSoundEmbeddingEvaluator

  def __init__(
      self,
      dataset_size: int = 5,
  ):
    super().__init__()
    self._dataset_size = dataset_size

  def load_data(self):
    for i in range(self._dataset_size):
      yield types.Sound(
          waveform=np.zeros(16000),
          context=types.SoundContextParams(
              sample_rate=16000, length=16000, id=str(i)
          ),
      )

  def multimodal_inputs(self) -> task_lib.Iterable[types.Sound]:
    return self.load_data()

  def compute_scores(self, cache: types.MultiModalEmbeddingCache):
    raise NotImplementedError()


class MSEBTaskTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_encoder = MockSoundEncoder()

  def test_list_tasks(self):
    tasks = task_lib.get_name_to_task()
    self.assertIn("mock_task", tasks)

  def test_get_task_by_name(self):
    task_cls = task_lib.get_task_by_name("mock_task")
    self.assertEqual(task_cls, MockTask)

  def test_get_task_by_name_not_found(self):
    with self.assertRaises(ValueError):
      task_lib.get_task_by_name("not_found")

  def test_sounds_deprecation_warning(self):
    with self.assertLogs(level="WARNING") as cm:
      sounds = list(MockTask().sounds())
      self.assertLen(sounds[0].waveform, 16000)
    self.assertTrue(any("sounds() is deprecated" in msg for msg in cm.output))

  def test_multimodal_inputs(self):
    self.assertLen(list(MockTask().multimodal_inputs()), 5)

  def test_multimodal_objects_for_setup_default_empty(self):
    self.assertEmpty(list(MockTask().multimodal_objects_for_setup()))

  def test_setup_accepts_embeddings_cache(self):
    # setup() with embeddings_cache=None should not raise.
    task = MockTask()
    task.setup(embeddings_cache=None)


if __name__ == "__main__":
  absltest.main()
