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

"""Clustering tasks."""

import abc
from typing import Iterable

from mseb import task
from mseb import types
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
