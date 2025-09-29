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

"""Reranking super task."""

import abc
import logging
import os
from typing import Iterable, Sequence

from mseb import runner as runner_lib
from mseb import task
from mseb import types
from mseb.evaluators import reranking_evaluator


logger = logging.getLogger(__name__)


class RerankingTask(task.MSEBTask):
  """Reranking task."""

  def __init__(self):
    super().__init__()
    self._evaluator = None

  @property
  def embeddings_dir(self) -> str:
    """The directory where the candidate embeddings cache is stored."""
    return os.path.join(task.CACHE_BASEPATH.value, 'rerankings')

  def setup(self, runner: runner_lib.EncoderRunner | None = None):
    """Create the candidate embeddings cache."""
    if runner is not None:
      assert hasattr(
          runner, '_output_path'
      ), 'Runner must have an _output_path attribute.'
      runner._output_path = self.embeddings_dir  # pylint: disable=protected-access
      unique_candidates = {}
      for candidate_list in self.candidate_lists():
        for candidate in candidate_list:
          unique_candidates[candidate.text] = candidate
      embeddings_by_text = runner.run(unique_candidates.values())
    else:
      try:
        logger.info(
            'Loading candidate embeddings cache from %s', self.embeddings_dir
        )
        embeddings_by_text = runner_lib.load_embeddings(
            os.path.join(self.embeddings_dir, 'embeddings')
        )
      except FileNotFoundError:
        raise ValueError(
            'Candidate embeddings cache not found in cache directory. Did you'
            ' create the cache by running run_task_setup?'
        ) from FileNotFoundError

    embeddings_by_sound_id = {}
    for sub_task in self.sub_tasks:
      for candidates in self.examples(sub_task):
        embeddings_by_sound_id[candidates.sound_id] = [
            embeddings_by_text[text] for text in candidates.texts
        ]

    self._evaluator = reranking_evaluator.RerankingEvaluator(
        candidate_embeddings_by_sound_id=embeddings_by_sound_id
    )

  def compute_scores(
      self, embeddings: types.MultiModalEmbeddingCache
  ) -> dict[str, list[types.Score]]:
    if self._evaluator is None:
      raise ValueError('Evaluator is not initialized. Did you call setup?')

    scores = {}
    for sub_task in self.sub_tasks:
      scores[sub_task] = self._evaluator.compute_metrics(
          self._evaluator.compute_predictions(embeddings),
          tuple(self.examples(sub_task)),
      )
    return scores

  @abc.abstractmethod
  def examples(
      self, sub_task: str
  ) -> Iterable[reranking_evaluator.RerankingCandidates]:
    """Get (utt_id, candidates) examples from dataset for a given sub-task."""

  @property
  @abc.abstractmethod
  def sub_tasks(self) -> list[str]:
    """Get the list of sub-tasks for the reranking task."""

  @abc.abstractmethod
  def candidate_lists(self) -> Iterable[Sequence[types.Text]]:
    """Get the list of candidates for the reranking task."""
