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

"""Segmentation super task."""

import abc
from typing import Iterable

from mseb import task
from mseb import types
from mseb.evaluators import segmentation_evaluator


class SegmentationTask(task.MSEBTask):
  """Segmentation super task.

  This task class orchestrates the evaluation pipeline for segmentation.
  The `setup` method initializes the `SegmentationEvaluator`, and the
  `compute_scores` method uses it to run the full evaluation, including
  accuracy, ranking (mAP), and order-based (NDCG, Edit Distance) metrics.

  Concrete subclasses must implement `sub_tasks`, `examples`, and `sounds`
  to provide the specific data for a given dataset.
  """

  def __init__(self, tau: float = 0.05):
    """Initializes the SegmentationTask.

    Args:
      tau: The acceptable time tolerance in seconds for a segment match,
        to be passed to the evaluator.
    """
    super().__init__()
    self._evaluator: segmentation_evaluator.SegmentationEvaluator | None = None
    self.tau = tau

  @property
  @abc.abstractmethod
  def sub_tasks(self) -> list[str]:
    """Get the list of sub-tasks, e.g., evaluation splits like ['test']."""
    ...

  @abc.abstractmethod
  def examples(
      self, sub_task: str
  ) -> Iterable[segmentation_evaluator.SegmentationReference]:
    """Get all reference examples for a given sub-task."""
    ...

  @abc.abstractmethod
  def sounds(self) -> Iterable[types.Sound]:
    """Iterate all sounds in the corpus for this task."""
    ...

  def setup(self, runner=None):
    """Initializes the SegmentationEvaluator."""
    self._evaluator = segmentation_evaluator.SegmentationEvaluator(tau=self.tau)

  def compute_scores(
      self, embeddings: types.MultiModalEmbeddingCache
  ) -> dict[str, list[types.Score]]:
    """Runs the full segmentation evaluation pipeline.

    For each sub-task, this method calculates intermediate scores and then
    computes the final, comprehensive set of metrics.

    Args:
      embeddings: A cache of `SoundEmbedding` objects from the model, keyed by
        example ID.

    Returns:
      A dictionary mapping each sub-task name to a list of its computed
      `types.Score` objects.

    Raises:
      ValueError: If the evaluator has not been initialized via `setup()`.
    """
    if self._evaluator is None:
      raise ValueError("Evaluator is not initialized. Did you call setup()?")

    results = {}
    for sub_task in self.sub_tasks:
      references = list(self.examples(sub_task))
      if not references:
        results[sub_task] = []
        continue
      scoring_result = self._evaluator.compute_scores(embeddings, references)
      final_scores = self._evaluator.compute_metrics(scoring_result)
      results[sub_task] = final_scores

    return results
