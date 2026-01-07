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
import json
import logging
import os
from typing import Iterable

from absl import flags
from etils import epath
from mseb import encoder as encoder_lib
from mseb import runner as runner_lib
from mseb import task
from mseb import types
from mseb.evaluators import retrieval_evaluator


_NUM_PARTITIONS = flags.DEFINE_integer(
    'num_partitions',
    1,
    'Number of partitions to use for the retrieval task.',
)

RETRIEVED_ITEMS_KEY = flags.DEFINE_string(
    'retrieved_items_key',
    None,
    'Key to use for the retrieved items.',
)

NO_RESPONSE_STR = encoder_lib.NO_RESPONSE_STR
INVALID_ANSWER_STR = encoder_lib.INVALID_ANSWER_STR

logger = logging.getLogger(__name__)


class RetrievalTask(task.MSEBTask):
  """Retrieval task."""

  def __init__(
      self,
      id_by_index_id_filepath: str = 'ids.txt',
  ):
    """Initializes the retrieval task.

    Args:
      id_by_index_id_filepath: The filepath to save the id by index id mapping.
    """
    super().__init__()
    self.id_by_index_id_filepath = id_by_index_id_filepath
    self._evaluator = None

  @property
  def evaluator(self) -> retrieval_evaluator.RetrievalEvaluator:
    if self._evaluator is None:
      raise ValueError('Evaluator is not initialized. Did you call setup?')
    return self._evaluator

  @property
  def index_dir(self) -> str:
    """The directory where the index is stored."""
    return os.path.join(task.TASK_CACHE_BASEPATH.value, 'retrievals')

  def setup(self, runner: runner_lib.EncoderRunner | None = None):
    """Create the index."""
    if _NUM_PARTITIONS.value > 1:
      self.setup_partitioned(_NUM_PARTITIONS.value, runner)
    else:
      self.setup_unpartitioned(runner)

  def setup_unpartitioned(self, runner: runner_lib.EncoderRunner | None = None):
    try:
      searcher, id_by_index_id = retrieval_evaluator.load_index(
          self.index_dir, self.id_by_index_id_filepath
      )
    except FileNotFoundError:
      if runner is not None:
        embeddings = runner.run(self.documents(), output_path=self.index_dir)
        searcher, id_by_index_id = retrieval_evaluator.build_index(embeddings)
        retrieval_evaluator.save_index(
            searcher,
            id_by_index_id,
            self.index_dir,
            self.id_by_index_id_filepath,
        )
      else:
        raise ValueError(
            'Index not found in cache directory. Did you create the index by'
            ' running run_task_setup?'
        ) from FileNotFoundError

    self._evaluator = retrieval_evaluator.RetrievalEvaluator(
        searcher=searcher, id_by_index_id=id_by_index_id
    )

  def setup_partitioned(
      self, num_partitions: int, runner: runner_lib.EncoderRunner | None = None
  ):
    for partition_id in range(num_partitions):
      logger.info('Setting up partition %d/%d', partition_id, num_partitions)
      if epath.Path(os.path.join(self.index_dir, str(partition_id))).exists():
        logger.info(
            'Index partition %d/%d already exists at %s',
            partition_id,
            num_partitions,
            os.path.join(self.index_dir, str(partition_id)),
        )
        continue
      elif runner is not None:
        embeddings = runner.run(
            itertools.islice(
                self.documents(), partition_id, None, num_partitions
            ),
            output_path=os.path.join(self.index_dir, str(partition_id)),
        )
        searcher, id_by_index_id = retrieval_evaluator.build_index(embeddings)
        retrieval_evaluator.save_index(
            searcher,
            id_by_index_id,
            os.path.join(self.index_dir, str(partition_id)),
            self.id_by_index_id_filepath,
        )
      else:
        raise ValueError(
            'Index partition %d/%d not found in cache directory. Did you create'
            ' the index by running run_task_setup?'
            % (partition_id, num_partitions)
        ) from FileNotFoundError

    self._evaluator = retrieval_evaluator.RetrievalEvaluatorPartitioned(
        index_dir=self.index_dir
    )

  def compute_scores(
      self, embeddings: types.MultiModalEmbeddingCache
  ) -> dict[str, list[types.Score]]:
    if self._evaluator is None:
      raise ValueError('Evaluator is not initialized. Did you call setup?')

    if not isinstance(next(iter(embeddings.values())), types.TextPrediction):
      predictions = self._evaluator.compute_predictions(embeddings)
    else:
      predictions = {}
      for sound_id, example in embeddings.items():
        assert isinstance(example, types.TextPrediction)
        if example.prediction == NO_RESPONSE_STR:
          predictions[sound_id] = NO_RESPONSE_STR
        elif example.prediction == INVALID_ANSWER_STR:
          predictions[sound_id] = INVALID_ANSWER_STR
        else:
          predictions[sound_id] = [
              # Assign pseudo-scores to preserve the predicted ranking.
              (item.get('score', 1 / n), item['id'])
              for n, item in enumerate(json.loads(example.prediction), 1)
          ]

    scores = {}
    for sub_task in self.sub_tasks:
      scores[sub_task] = self._evaluator.compute_metrics(
          predictions, tuple(self.examples(sub_task))
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
