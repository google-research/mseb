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
import os
from typing import Iterable

from mseb import svq_data
from mseb import task
from mseb import types
from mseb.evaluators import clustering_evaluator


class ClusteringTask(task.MSEBTask):
  """Clustering task."""

  def __init__(self, base_path: str):
    self._base_path = base_path
    self._evaluator = clustering_evaluator.ClusteringEvaluator()

  def compute_scores(
      self, embeddings: types.SoundEmbeddingCache
  ) -> dict[str, list[types.Score]]:
    scores = {}
    for sub_task in self.sub_tasks:
      scores[sub_task] = self._evaluator(embeddings, self.examples(sub_task))
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
      name='clustering',
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

  @property
  def sub_tasks(self) -> list[str]:
    return ['speaker_gender', 'speaker_age', 'speaker_id']

  def sounds(self) -> Iterable[types.Sound]:
    for example in svq_data.generate_examples(
        os.path.join(self._base_path, 'utt_index.jsonl')
    ):
      yield types.Sound(
          example['waveform'],
          types.SoundContextParams(
              sample_rate=48000,
              length=len(example['waveform']),
              sound_id=example['utt_id'],
              speaker_id=example['speaker_id'],
              speaker_age=example['speaker_age'],
              speaker_gender=example['speaker_gender'],
          ),
      )

  def examples(
      self, sub_task: str
  ) -> Iterable[clustering_evaluator.ClusteringExample]:
    """Get (utt_id, label) examples from svq dataset."""
    for ex in svq_data.generate_examples(
        os.path.join(self._base_path, 'utt_index.jsonl')
    ):
      yield clustering_evaluator.ClusteringExample(ex['utt_id'], ex[sub_task])
