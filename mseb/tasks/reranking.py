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
import tempfile
from typing import Iterable, Sequence, Type

from mseb import runner as runner_lib
from mseb import task
from mseb import types
from mseb.encoders import encoder_registry
from mseb.evaluators import reranking_evaluator


logger = logging.getLogger(__name__)


class RerankingTask(task.MSEBTask):
  """Reranking task."""

  def __init__(
      self,
      cache_dir: str | None = None,
      text_encoder_name: str | None = None,
  ):
    super().__init__(
        cache_dir=cache_dir or os.path.join(tempfile.gettempdir(), 'mseb_cache')
    )
    self.text_encoder_name = text_encoder_name
    self._evaluator = None

  def setup(
      self, runner_cls: Type[runner_lib.EncoderRunner] | None = None, **kwargs
  ):
    """Create the candidate embeddings cache."""
    if runner_cls is not None:
      if self.text_encoder_name is None:
        raise ValueError('Text encoder name is not set.')
      text_encoder = encoder_registry.get_encoder_metadata(
          self.text_encoder_name
      ).load()
      runner = runner_cls(encoder=text_encoder, **kwargs)
      unique_candidates = {}
      for candidate_list in self.candidate_lists():
        for candidate in candidate_list:
          unique_candidates[candidate.text] = candidate
      embeddings = runner.run(unique_candidates.values())
    else:
      try:
        logger.info(
            'Loading candidate embeddings cache from %s', self.cache_dir
        )
        embeddings = runner_lib.load_embeddings(
            os.path.join(self.cache_dir, 'embeddings')
        )
      except FileNotFoundError:
        raise ValueError(
            'Candidate embeddings cache not found in cache directory. Did you'
            ' create the cache by running run_task_setup?'
        ) from FileNotFoundError

    self._evaluator = reranking_evaluator.RerankingEvaluatorV2(
        candidate_embeddings_by_text=embeddings
    )

  def compute_scores(
      self, embeddings: types.SoundEmbeddingCache
  ) -> dict[str, list[types.Score]]:
    if self._evaluator is None:
      raise ValueError('Evaluator is not initialized. Did you call setup?')
    scores = {}
    for sub_task in self.sub_tasks:
      scores[sub_task] = self._evaluator(
          embeddings, tuple(self.examples(sub_task))
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
