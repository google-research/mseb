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

"""Segmentation super task."""

import abc
import os
from typing import Iterable, Sequence

from absl import logging as logger
from mseb import runner as runner_lib
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
      tau: The acceptable time tolerance in seconds for a segment match, to be
        passed to the evaluator.
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
  def multimodal_inputs(self) -> Iterable[types.Sound]:
    """Iterate all sounds in the corpus for this task."""
    ...

  def setup(self, runner=None, embeddings_cache=None):
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
      raise ValueError('Evaluator is not initialized. Did you call setup()?')

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


class SegmentationSelectionTask(SegmentationTask):
  """Segmentation selection task."""

  def salient_term_lists(
      self,
  ) -> Iterable[tuple[str, Sequence[segmentation_evaluator.Segment]]]:
    """Iterate all salient term lists in the corpus for this task."""
    return []

  def multimodal_objects_for_setup(self) -> Iterable[types.MultiModalObject]:
    """Get all salient term needed for setting up the task."""
    for _, term_list in self.salient_term_lists():
      for term in term_list:
        yield types.Text(
            text=term.embedding,
            context=types.TextContextParams(id=term.embedding),
        )

  @property
  def embeddings_dir(self) -> str:
    """The directory where the salient term embeddings cache is stored."""
    return os.path.join(  # pyrefly: ignore[no-matching-overload]
        task.TASK_CACHE_BASEPATH.value, 'segmentation_selection'
    )

  def setup(self, runner=None, embeddings_cache=None):
    """Initializes the SegmentationEvaluator."""
    super().setup(runner=runner, embeddings_cache=embeddings_cache)
    if self.salient_term_lists():
      try:
        embeddings_path_prefix = os.path.join(self.embeddings_dir, 'embeddings')
        logger.info(
            'Loading salient term embeddings cache from %s',
            embeddings_path_prefix,
        )
        _ = runner_lib.load_embeddings(embeddings_path_prefix)
      except FileNotFoundError:
        if embeddings_cache is not None:
          term_embeddings = {}
          for _, stl in self.salient_term_lists():
            for st in stl:
              term_embeddings[st.embedding] = embeddings_cache[st.embedding]
          runner_lib.save_embeddings(
              os.path.join(self.embeddings_dir, 'embeddings'), term_embeddings
          )
        elif runner is not None:
          unique_terms = {}
          for _, term_list in self.salient_term_lists():
            for term in term_list:
              unique_terms[term.embedding] = term.embedding
          runner.run(unique_terms.values(), output_path=self.embeddings_dir)
        else:
          logger.warning(
              'Salient term embeddings cache not found in cache directory. Did'
              ' you create the cache by running run_task_setup?'
          )
