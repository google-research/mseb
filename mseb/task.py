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

"""MSEB Task Class."""

import abc
import itertools
import logging
from typing import Any, Iterable, Sequence

from mseb import encoder
from mseb import evaluator
from mseb import types

logger = logging.getLogger(__name__)


class MSEBTask(abc.ABC):
  """Abstract base class for MSEB tasks.

  This class orchestrates the evaluation pipeline for a specific task.
  """
  metadata: types.TaskMetadata = None
  # A concrete task must define the specific evaluator class it uses.
  evaluator_cls: type[evaluator.SoundEmbeddingEvaluator] = None

  def __init__(self,
               sound_encoder: encoder.SoundEncoder,
               evaluator_kwargs: dict[str, Any] | None = None,
               ):
    """Initializes the task with fully-formed dependency instances.

    Args:
      sound_encoder: An instantiated object that inherits from
        `encoder.SoundEncoder`. This object is responsible for its own model
        loading via its `setup()` method.
      evaluator_kwargs: A dictionary of keyword arguments to be passed to the
        evaluator's constructor.
    """
    if self.metadata is None:
      raise NotImplementedError(
          "A concrete task must define its `metadata` class attribute."
      )
    if self.evaluator_cls is None:
      raise NotImplementedError(
          "A concrete task must specify its `evaluator_cls` class attribute."
      )
    self.encoder = sound_encoder
    self.evaluator = self.evaluator_cls(**(evaluator_kwargs or {}))

  def setup(self):
    """Delegates the setup call to the injected encoder.

    This is designed to be called once on each worker in a distributed
    setting to handle the loading of the actual model into memory.
    """
    self.encoder.setup()

  @abc.abstractmethod
  def load_data(
      self
    ) -> Iterable[tuple[Sequence[float], types.SoundContextParams]]:
    """Loads and yields dataset examples one by one.

    This method must return an iterable (like a generator) that yields
    tuples, where each tuple contains the sound sequence and its
    corresponding sound context parameters.

    Returns:
      An iterable of (waveform, params) tuples.
    """
    ...

  def load_batched_data(
      self, batch_size: int
    ) -> Iterable[list[tuple[Sequence[float], types.SoundContextParams]]]:
    """Yields batches of data from the `load_data` iterable.

    Subclasses CAN override this method for a more performant, framework-native
    batching implementation (e.g., using tf.data or PyTorch DataLoader).

    Args:
      batch_size: The number of examples per batch.

    Yields:
      A list of (waveform, params) tuples, where the list size is `batch_size`.
    """
    iterator = self.load_data()
    while True:
      batch = list(itertools.islice(iterator, batch_size))
      if not batch:
        break
      yield batch

  def run(self, batch_size: int = 1) -> dict[str, list[types.Score]]:
    """Default "simple runner" for local, single-machine, batched execution.

    This runner orchestrates the entire process: it ensures the model is
    loaded, iterates through batches of data, passes each batch to the encoder,
    passes the results to the evaluator, and finally aggregates the scores.

    Args:
      batch_size: The number of examples to process in each batch.

    Returns:
      A dictionary of lists of aggregated Score object keyed by sub-task name.
      Sub-tasks are useful when a single dataset contains examples for multiple
      related tasks that re-use the same encoder outputs.
    """
    # 1. Ensure the model is loaded.
    self.setup()

    logger.info("--- Starting evaluation for %s ---", self.metadata.name)
    dataset_iterable = self.load_batched_data(batch_size=batch_size)

    # 2. Orchestrate the batch encode -> batch evaluate pipeline.
    all_scores_per_example: list[list[types.Score]] = []
    for batch in dataset_iterable:
      waveforms, params = zip(*batch)
      encoder_outputs = self.encoder.encode_batch(waveforms, params)
      scores = self.evaluator.evaluate_batch(encoder_outputs, params)
      all_scores_per_example.extend(scores)

    if not all_scores_per_example:
      logger.warning(
          "Warning: No scores were generated for task %s. "
          "The dataset might be empty.", self.metadata.name
      )
      return {self.metadata.name: []}

    # 3. Combine the final scores using the evaluator's aggregation logic.
    logger.info(
        "--- Aggregating scores from %d examples... ---",
        len(all_scores_per_example),
    )
    final_scores = self.evaluator.combine_scores(all_scores_per_example)

    logger.info("--- Evaluation for %s complete ---", self.metadata.name)

    return {self.metadata.name: final_scores}


def get_task_list() -> list[type[MSEBTask]]:
  return list(MSEBTask.__subclasses__())


def get_name_to_task() -> dict[str, type[MSEBTask]]:
  tasks = get_task_list()
  return {cls.metadata.name: cls for cls in tasks if cls.metadata}

