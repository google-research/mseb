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

"""Clustering tasks."""

import abc
from typing import Iterable, Type

from mseb import runner as runner_lib
from mseb import task
from mseb import types
from mseb.datasets import simple_voice_questions as svq
from mseb.evaluators import clustering_evaluator


class ClusteringTask(task.MSEBTask):
  """Clustering task."""

  def __init__(self):
    self._evaluator = clustering_evaluator.ClusteringEvaluator()

  def compute_scores(
      self, embeddings: types.MultiModalEmbeddingCache
  ) -> dict[str, list[types.Score]]:
    sound_embeddings = {}
    for k, v in embeddings.items():
      assert isinstance(v, types.SoundEmbedding)
      sound_embeddings[k] = v
    scores = {}
    for sub_task in self.sub_tasks:
      scores[sub_task] = self._evaluator(
          sound_embeddings, self.examples(sub_task)
      )
    return scores

  @abc.abstractmethod
  def examples(
      self, sub_task: str
  ) -> Iterable[clustering_evaluator.ClusteringExample]:
    """Get (utt_id, label) examples from dataset for a given sub-task."""

  @property
  @abc.abstractmethod
  def sub_tasks(self) -> list[str]:
    """Get the list of sub-tasks for the clustering task."""


class SVQClustering(ClusteringTask):
  """SVQ clustering."""

  metadata = types.TaskMetadata(
      name='SVQClustering',
      description='Clustering task.',
      reference='TODO',
      type='Clustering',
      category='speech',
      main_score='VMeasure',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[clustering_evaluator.vmeasure_score()],
      eval_splits=['test'],
      eval_langs=['en-US'],
      domains=['speech'],
      task_subtypes=['clustering'],
  )

  _svq_dataset: svq.SimpleVoiceQuestionsDataset

  def setup(
      self, runner_cls: Type[runner_lib.EncoderRunner] | None = None, **kwargs
  ):
    self._svq_dataset = svq.SimpleVoiceQuestionsDataset()

  @property
  def sub_tasks(self) -> list[str]:
    return ['speaker_gender', 'speaker_age', 'speaker_id']

  def sounds(self) -> Iterable[types.Sound]:
    for example in self._svq_dataset.get_task_data('utt_index').itertuples():
      yield self._svq_dataset.get_sound_by_id(example.utt_id)

  def examples(
      self, sub_task: str
  ) -> Iterable[clustering_evaluator.ClusteringExample]:
    """Get (utt_id, label) examples from svq dataset."""
    for example in self._svq_dataset.get_task_data('utt_index').itertuples():
      yield clustering_evaluator.ClusteringExample(
          example.utt_id, getattr(example, sub_task)
      )
