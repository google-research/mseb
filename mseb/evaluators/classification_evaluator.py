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

"""Evaluator for classification tasks."""

from __future__ import annotations

import dataclasses
import logging
from typing import Callable, Mapping, Optional, Sequence

from mseb import types
import numpy as np
from sklearn import metrics


logger = logging.getLogger(__name__)


def accuracy(value: float = 0.0) -> types.Score:
  return types.Score(
      metric='Accuracy',
      description='Overall classification accuracy',
      value=value,
      min=0.0,
      max=1.0,
  )


def top_k_accuracy(value: float = 0.0, k: int = 1) -> types.Score:
  return types.Score(
      metric=f'Top-{k} Accuracy',
      description=f'Accuracy considering the top {k} predictions',
      value=value,
      min=0.0,
      max=1.0,
  )


def balanced_accuracy(value: float = 0.0) -> types.Score:
  return types.Score(
      metric='Balanced Accuracy',
      description='Accuracy adjusted for class imbalance',
      value=value,
      min=0.0,
      max=1.0,
  )


def weighted_precision(value: float = 0.0) -> types.Score:
  return types.Score(
      metric='Weighted Precision',
      description='Precision weighted by class support',
      value=value,
      min=0.0,
      max=1.0,
  )


def weighted_recall(value: float = 0.0) -> types.Score:
  return types.Score(
      metric='Weighted Recall',
      description='Recall weighted by class support',
      value=value,
      min=0.0,
      max=1.0,
  )


def weighted_f1(value: float = 0.0) -> types.Score:
  return types.Score(
      metric='Weighted F1-Score',
      description='F1-Score weighted by class support',
      value=value,
      min=0.0,
      max=1.0,
  )


@dataclasses.dataclass
class ClassificationReference:
  """A generic ground-truth reference for a classification task."""
  example_id: str
  label_id: str


def _get_embedding_array(
        embedding_obj: types.MultiModalEmbedding
) -> np.ndarray:
  """Extracts the embedding numpy array from a MultiModalEmbedding object."""
  if isinstance(embedding_obj, types.SoundEmbedding):
    return embedding_obj.embedding
  if isinstance(embedding_obj, types.TextEmbedding):
    return embedding_obj.embedding
  raise TypeError(f'Unsupported embedding type: {type(embedding_obj)}')


class ClassificationEvaluator:
  """Generic evaluator for classification tasks on any modality."""

  def __init__(
      self,
      embedding_table: Optional[np.ndarray],
      id_by_class_index: Sequence[str],
      distance_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = np.dot,
      top_k_value: int = 5,
  ):
    """Initializes the ClassificationEvaluator.

    This evaluator can be configured for a "full pipeline" (generating scores
    from embeddings and then computing metrics) or "metrics-only" (if you
    already have scores).

    Args:
      embedding_table: An optional [num_classes, embedding_dim] numpy array
        where each row is the vector representation for a class. This is
        required only when using the `compute_predictions` method. For
        metrics-only evaluation, this can be `None`.
      id_by_class_index: A required sequence of strings representing the class
        labels. The order of labels must correspond to the row order of the
        `embedding_table`. This is needed to map score vector indices to
        human-readable labels.
      distance_fn: A callable function used to compute scores between an example
        embedding and the class embedding table. It should accept a (D,) array
        and a (D, V) array and return a (V,) array of scores. Defaults to
        `numpy.dot`.
      top_k_value: The integer `k` to use for calculating Top-K accuracy.
        Defaults to 5.

    Raises:
      ValueError: If `top_k_value` is not a positive integer.
    """
    num_classes = len(id_by_class_index)
    if top_k_value <= 0:
      raise ValueError(
          f'top_k_value must be positive, but got {top_k_value}.'
      )
    if top_k_value >= num_classes:
      logger.warning(
          'top_k_value (%d) is >= to the number of classes (%d). '
          'Top-%d Accuracy will always be 100%%.',
          top_k_value, num_classes, top_k_value
      )
    self.embedding_table = embedding_table
    self.id_by_class_index = id_by_class_index
    self.index_by_id = {
        label_id: i for i, label_id in enumerate(id_by_class_index)
    }
    self.distance_fn = distance_fn
    self.top_k_value = top_k_value

  def compute_predictions(
      self,
      embeddings_by_id: types.MultiModalEmbeddingCache
  ) -> Mapping[str, np.ndarray]:
    """Computes prediction scores for the given embeddings."""
    if self.embedding_table is None:
      raise ValueError(
          'An embedding_table must be provided during initialization to '
          'use compute_predictions.'
      )

    classification_scores = {}
    for example_id, embedding_obj in embeddings_by_id.items():
      embedding_array = _get_embedding_array(embedding_obj)

      if embedding_array.ndim != 2 or embedding_array.shape[0] == 0:
        raise ValueError(
            'Found missing or malformed embeddings '
            f'for example_id {example_id}. '
            f'Expected shape (N, D) with N > 0, but got shape '
            f'{embedding_array.shape}.'
        )
      example_embedding = embedding_array[0]
      classification_scores[example_id] = self.distance_fn(
          example_embedding,
          self.embedding_table.T
      )
    return classification_scores

  def compute_metrics(
      self,
      scores: Mapping[str, np.ndarray],
      references: Sequence[ClassificationReference],
  ) -> list[types.Score]:
    """Computes a suite of classification metrics from prediction scores.

    This method takes raw score vectors for a set of examples and compares them
    against ground-truth references to calculate performance.

    The following metrics are computed:
      - Accuracy
      - Top-K Accuracy
      - Balanced Accuracy
      - Weighted Precision
      - Weighted Recall
      - Weighted F1-Score

    Args:
      scores: A mapping from a unique example ID to its corresponding raw
        prediction score vector. Each score vector should be a 1D numpy array
        of shape `[num_classes]`.
      references: A sequence of `ClassificationReference` objects, where each
        object contains the ground-truth `example_id` and `label_id`.

    Returns:
      A list of `types.Score` dataclass objects, with each object representing
      one of the computed metrics. Returns an empty list if no matching
      predictions are found for the provided references.
    """
    y_true_labels, y_pred_labels, y_scores_list = [], [], []

    for ref in references:
      if ref.example_id in scores:
        y_true_labels.append(ref.label_id)
        pred_index = np.argmax(scores[ref.example_id])
        y_pred_labels.append(self.id_by_class_index[pred_index])
        y_scores_list.append(scores[ref.example_id])

    if not y_true_labels:
      logger.error('No matching predictions found for the given references.')
      return []

    y_scores = np.array(y_scores_list)
    acc = metrics.accuracy_score(y_true_labels, y_pred_labels)
    true_indices = np.array(
        [self.index_by_id[label] for label in y_true_labels]
    )
    top_k_indices = np.argsort(y_scores, axis=1)[:, -self.top_k_value:]
    top_k_hits = np.any(top_k_indices == true_indices[:, np.newaxis], axis=1)
    top_k_acc = np.mean(top_k_hits)
    bal_acc = metrics.balanced_accuracy_score(y_true_labels, y_pred_labels)
    class_labels = list(self.id_by_class_index)
    prec, rec, f1, _ = metrics.precision_recall_fscore_support(
        y_true_labels,
        y_pred_labels,
        average='weighted',
        labels=class_labels,
        zero_division=0
    )

    return [
        accuracy(acc),
        top_k_accuracy(top_k_acc, k=self.top_k_value),
        balanced_accuracy(bal_acc),
        weighted_precision(prec),
        weighted_recall(rec),
        weighted_f1(f1),
    ]
