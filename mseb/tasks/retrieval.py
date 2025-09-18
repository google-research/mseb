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
import itertools
import logging
import os
import tempfile
from typing import Any, Iterable, Type

from mseb import runner as runner_lib
from mseb import task
from mseb import types
from mseb.encoders import encoder_registry
from mseb.evaluators import retrieval_evaluator


logger = logging.getLogger(__name__)


class RetrievalTask(task.MSEBTask):
  """Retrieval task."""

  def __init__(
      self,
      cache_dir: str | None = None,
      text_encoder_name: str | None = None,
      id_by_index_id_filepath: str = 'ids.txt',
      num_partitions: int = 1,
  ):
    """Initializes the retrieval task.

    Args:
      cache_dir: The cache directory to store the document embeddings / index.
      text_encoder_name: The name of the text encoder to build the index.
      id_by_index_id_filepath: The filepath to save the id by index id mapping.
      num_partitions: The number of index partitions to use.
    """
    super().__init__(
        cache_dir=cache_dir or os.path.join(tempfile.gettempdir(), 'mseb_cache')
    )
    self.text_encoder_name = text_encoder_name
    self.id_by_index_id_filepath = id_by_index_id_filepath
    self.num_partitions = num_partitions
    self._evaluator = None

  @property
  def index_dir(self) -> str:
    """The directory where the index is stored."""
    return os.path.join(self.cache_dir, 'retrievals')

  def setup(
      self, runner_cls: Type[runner_lib.EncoderRunner] | None = None, **kwargs
  ):
    """Create the index."""
    if self.num_partitions > 1:
      self.setup_partitioned(runner_cls, **kwargs)
    else:
      self.setup_unpartitioned(runner_cls, **kwargs)

  def setup_unpartitioned(
      self, runner_cls: Type[runner_lib.EncoderRunner] | None = None, **kwargs
  ):
    assert self.num_partitions == 1, (
        'setup_unpartitioned requires num_partitions == 1.'
    )
    if runner_cls is not None:
      if self.text_encoder_name is None:
        raise ValueError('Text encoder name is not set.')
      text_encoder = encoder_registry.get_encoder_metadata(
          self.text_encoder_name
      ).load()
      kwargs: dict[str, Any] = {'output_path': self.index_dir, **kwargs}
      runner = runner_cls(encoder=text_encoder, **kwargs)
      embeddings = runner.run(self.documents())
      searcher, id_by_index_id = retrieval_evaluator.build_index(embeddings)
      retrieval_evaluator.save_index(
          searcher,
          id_by_index_id,
          self.index_dir,
          self.id_by_index_id_filepath,
      )
    else:
      try:
        searcher, id_by_index_id = retrieval_evaluator.load_index(
            self.index_dir, self.id_by_index_id_filepath
        )
      except FileNotFoundError:
        raise ValueError(
            'Index not found in cache directory. Did you create the index by'
            ' running run_task_setup?'
        ) from FileNotFoundError

    self._evaluator = retrieval_evaluator.RetrievalEvaluatorV2(
        searcher=searcher, id_by_index_id=id_by_index_id
    )

  def setup_partitioned(
      self, runner_cls: Type[runner_lib.EncoderRunner] | None = None, **kwargs
  ):
    if runner_cls is not None:
      if self.text_encoder_name is None:
        raise ValueError('Text encoder name is not set.')
      text_encoder = encoder_registry.get_encoder_metadata(
          self.text_encoder_name
      ).load()
      for partition_id in range(self.num_partitions):
        logger.info(
            'Setting up partition %d/%d', partition_id, self.num_partitions
        )
        index_dir = kwargs.pop('output_path', self.index_dir)
        kwargs: dict[str, Any] = {
            'output_path': os.path.join(index_dir, str(partition_id)),
            **kwargs,
        }
        runner = runner_cls(encoder=text_encoder, **kwargs)
        embeddings = runner.run(
            itertools.islice(
                self.documents(), partition_id, None, self.num_partitions
            )
        )
        searcher, id_by_index_id = retrieval_evaluator.build_index(embeddings)
        retrieval_evaluator.save_index(
            searcher,
            id_by_index_id,
            os.path.join(self.index_dir, str(partition_id)),
            self.id_by_index_id_filepath,
        )

    self._evaluator = retrieval_evaluator.RetrievalEvaluatorPartitioned(
        index_dir=self.index_dir
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
  ) -> Iterable[retrieval_evaluator.RetrievalReferenceId]:
    """Get (utt_id, reference_id) examples from dataset for a given sub-task."""

  @property
  @abc.abstractmethod
  def sub_tasks(self) -> list[str]:
    """Get the list of sub-tasks for the retrieval task."""

  @abc.abstractmethod
  def documents(self) -> Iterable[types.Text]:
    """Get the list of documents for the retrieval task."""
