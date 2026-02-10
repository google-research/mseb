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

"""Classification super task."""

import abc
import dataclasses
import logging
import os
from typing import Iterable, Mapping, Sequence, Union

from mseb import runner as runner_lib
from mseb import task
from mseb import types
from mseb.evaluators import classification_evaluator
import numpy as np


logger = logging.getLogger(__name__)


ReferenceType = Union[
    classification_evaluator.ClassificationReference,
    classification_evaluator.MultiLabelClassificationReference,
]


class ClassificationTask(task.MSEBTask):
  """Multi-class or Multi-label Classification task.

  This task can handle both single-label (multi-class) and multi-label
  classification by selecting the appropriate evaluator based on the `task_type`
  property.

  The `setup` method defines the classifier's weights in one of two modes:
  1.  **Zero-Shot Initialization:** When an `EncoderRunner` is provided, the
      weights are initialized directly from the embeddings of the class labels
      (i.e., the class names are encoded as text to form the weight vectors).
  2.  **Loading from Cache:** When no runner is provided, the task loads a
      pre-trained linear classifier (weights and biases) from a cached
      directory.

  A bias term is handled automatically by padding a '1.0' to the input
  embeddings during evaluation and expecting a corresponding extra dimension
  in the weight matrix.
  """

  def __init__(
      self,
      top_k_value: int = 5,
      multi_label_threshold: float = 0.5,
  ):
    """Initializes the ClassificationTask.

    Args:
      top_k_value: The 'k' for top-k accuracy (for multi_class).
      multi_label_threshold: Decision threshold for multi-label metrics.
    """
    super().__init__()
    self._evaluator = None
    self.top_k_value = top_k_value
    self.multi_label_threshold = multi_label_threshold

  @property
  @abc.abstractmethod
  def task_type(self) -> str:
    """Get the classification task type: 'multi_class' or 'multi_label'."""
    ...

  @abc.abstractmethod
  def examples(self, sub_task: str) -> Iterable[ReferenceType]:
    """Get examples from dataset for a given sub-task."""
    ...

  @property
  @abc.abstractmethod
  def sub_tasks(self) -> list[str]:
    """Get the list of sub-tasks for the classification task."""
    ...

  @abc.abstractmethod
  def class_labels(self) -> Iterable[str]:
    """Get the list of class labels for the classification task."""
    ...

  @property
  def weights_dir(self) -> str:
    """The directory where the weights of the linear classifier are stored."""
    return os.path.join(task.TASK_CACHE_BASEPATH.value, "classifications")

  def setup(self, runner: runner_lib.EncoderRunner | None = None):
    """Creates/loads weights and instantiates the correct evaluator."""
    try:
      class_labels, weights = classification_evaluator.load_linear_classifier(
          self.weights_dir
      )
    except FileNotFoundError:
      if runner is not None:
        if runner.encoder_output_type() is not types.TextPrediction:
          class_labels, weights = self._create_weights_from_runner(runner)
        else:
          class_labels = list(self.class_labels())
          weights = None
      else:
        class_labels = list(self.class_labels())
        weights = None
        logger.error("Weights not found in cache. Did you run run_task_setup?")

    if self.task_type == "multi_class":
      self._evaluator = classification_evaluator.ClassificationEvaluator(
          class_labels=class_labels,
          weights=weights,
          top_k_value=self.top_k_value,
      )
    elif self.task_type == "multi_label":
      self._evaluator = (
          classification_evaluator.MultiLabelClassificationEvaluator(
              id_by_class_index=class_labels,
              weights=weights,
          )
      )
    else:
      raise ValueError(f"Unknown task_type: '{self.task_type}'")

  def compute_scores(
      self, embeddings: types.MultiModalEmbeddingCache
  ) -> dict[str, list[types.Score]]:
    """Runs the full evaluation pipeline (embeddings -> predictions -> metrics)."""
    if self._evaluator is None:
      raise ValueError("Evaluator is not initialized. Did you call setup?")

    if isinstance(next(iter(embeddings.values())), types.TextPrediction):
      predictions = {}
      for k, v in embeddings.items():
        assert isinstance(v, types.TextPrediction)
        predicted_labels = set(v.prediction.split("\n"))
        predictions[k] = np.array(
            object=[
                1.0 if x in predicted_labels else 0.0
                for x in self._evaluator.get_extended_class_labels()
            ]
        )
    else:
      embeddings_with_bias = {}
      for k, v in embeddings.items():
        embedding_array = classification_evaluator.get_embedding_array(v)
        embeddings_with_bias[k] = dataclasses.replace(
            v,
            embedding=np.pad(
                embedding_array,
                ((0, 0), (0, 1)),
                "constant",
                constant_values=1.0,
            ),
        )
      predictions = self._evaluator.compute_predictions(embeddings_with_bias)

    return self._compute_metrics(predictions)

  def _compute_metrics(
      self, scores: Mapping[str, np.ndarray]
  ) -> dict[str, list[types.Score]]:
    """Computes metrics from a pre-computed dictionary of scores."""
    if self._evaluator is None:
      raise ValueError("Evaluator is not initialized. Did you call setup?")
    results = {}
    for sub_task in self.sub_tasks:
      kwargs = {}
      if self.task_type == "multi_label":
        kwargs["threshold"] = self.multi_label_threshold

      results[sub_task] = self._evaluator.compute_metrics(
          scores,
          tuple(self.examples(sub_task)),
          **kwargs,
      )
    return results

  def _create_weights_from_runner(
      self, runner: runner_lib.EncoderRunner
  ) -> tuple[Sequence[str], np.ndarray]:
    """Generate classifier weights by running a text encoder over class labels."""
    class_labels = tuple(self.class_labels())
    class_label_embeddings = runner.run([
        types.Text(
            text=class_label, context=types.TextContextParams(id=class_label)
        )
        for class_label in class_labels
    ], output_name="label_embeddings", output_path=self.weights_dir)
    weights = []
    for class_label in class_labels:
      embedding = class_label_embeddings[class_label]
      assert isinstance(embedding, types.TextEmbedding)
      weight_array = classification_evaluator.get_embedding_array(embedding)
      # Pad with a zero for the bias term dimension
      weight = np.pad(
          weight_array, ((0, 0), (0, 1)), "constant", constant_values=0.0
      )
      weights.append(weight)

    weights = np.squeeze(np.array(weights, dtype=np.float32), axis=1)
    classification_evaluator.save_linear_classifier(
        class_labels, weights, self.weights_dir
    )
    return class_labels, weights
