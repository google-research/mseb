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

"""Retrieval super task."""

import abc
import logging
import os
import tempfile
from typing import Any, Iterable, Type

from mseb import encoder as encoder_lib
from mseb import runner as runner_lib
from mseb import task
from mseb import types
from mseb.evaluators import retrieval_evaluator


logger = logging.getLogger(__name__)


class RetrievalTask(task.MSEBTask):
  """Retrieval task."""

  def __init__(
      self,
      cache_dir: str | None = None,
      text_encoder_cls: type[encoder_lib.TextEncoder] | None = None,
      text_encoder_kwargs: dict[str, Any] | None = None,
      id_by_index_id_filepath: str = 'ids.txt',
  ):
    super().__init__(
        cache_dir=cache_dir or os.path.join(tempfile.gettempdir(), 'mseb_cache')
    )
    self.text_encoder_cls = text_encoder_cls
    self.text_encoder_kwargs = text_encoder_kwargs
    self.id_by_index_id_filepath = id_by_index_id_filepath
    self._evaluator = None

  def setup(
      self, runner_cls: Type[runner_lib.EncoderRunner] | None = None, **kwargs
  ):
    """Create the index."""
    if runner_cls is not None:
      if self.text_encoder_cls is None:
        raise ValueError('Text encoder class is not set.')
      if self.text_encoder_kwargs is None:
        raise ValueError('Text encoder kwargs are not set.')
      encoder = self.text_encoder_cls(**self.text_encoder_kwargs)
      runner = runner_cls(encoder=encoder, **kwargs)
      embeddings = runner.run(self.documents())
      logger.info('Building ScaNN index...')
      searcher, id_by_index_id = retrieval_evaluator.build_index(embeddings)
      logger.info('Built ScaNN index with %d documents.', len(id_by_index_id))
      logger.info('Saving ScaNN index to %s', self.cache_dir)
      retrieval_evaluator.save_index(
          searcher,
          id_by_index_id,
          self.cache_dir,
          self.id_by_index_id_filepath,
      )
    else:
      try:
        logger.info('Loading ScaNN index from %s', self.cache_dir)
        searcher, id_by_index_id = retrieval_evaluator.load_index(
            self.cache_dir, self.id_by_index_id_filepath
        )
      except FileNotFoundError:
        raise ValueError(
            'Index not found in cache directory. Did you create the index by'
            ' running run_task_setup?'
        ) from FileNotFoundError

    self._evaluator = retrieval_evaluator.RetrievalEvaluatorV2(
        searcher=searcher, id_by_index_id=id_by_index_id
    )

  def compute_scores(
      self, embeddings: types.SoundEmbeddingCache
  ) -> dict[str, list[types.Score]]:
    if self._evaluator is None:
      raise ValueError('Evaluator is not initialized. Did you call setup?')
    scores = {}
    for sub_task in self.sub_tasks:
      scores[sub_task] = self._evaluator(embeddings, self.examples(sub_task))
    return scores

  @abc.abstractmethod
  def examples(
      self, sub_task: str
  ) -> Iterable[retrieval_evaluator.RetrievalReferenceId]:
    """Get (utt_id, reference_id) examples from dataset for a given sub-task."""

  @property
  @abc.abstractmethod
  def sub_tasks(self) -> list[str]:
    """Get the list of sub-tasks for the retrieval task."""

  @abc.abstractmethod
  def documents(self) -> Iterable[types.Text]:
    """Get the list of documents for the retrieval task."""
