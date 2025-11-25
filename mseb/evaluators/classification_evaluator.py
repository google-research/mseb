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
import functools
import logging
import os
from typing import Mapping, Sequence

from etils import epath
import jaxtyping
from mseb import evaluator as evaluator_lib
from mseb import types
from mseb.evaluators import reasoning_evaluator
import numpy as np
from sklearn import metrics


logger = logging.getLogger(__name__)

INVALID_ANSWER_STR = reasoning_evaluator.INVALID_ANSWER_STR
NO_RESPONSE_STR = reasoning_evaluator.NO_RESPONSE_STR


def accuracy(value: float = 0.0) -> types.Score:
  """Creates a Score object for accuracy."""
  return types.Score(
      metric='Accuracy',
      description='Overall classification accuracy',
      value=value,
      min=0.0,
      max=1.0,
  )


def top_k_accuracy(value: float = 0.0, k: int = 1) -> types.Score:
  """Creates a Score object for top-k accuracy."""
  return types.Score(
      metric=f'Top-{k} Accuracy',
      description=f'Accuracy considering the top {k} predictions',
      value=value,
      min=0.0,
      max=1.0,
  )


def balanced_accuracy(value: float = 0.0) -> types.Score:
  """Creates a Score object for balanced accuracy."""
  return types.Score(
      metric='Balanced Accuracy',
      description='Accuracy adjusted for class imbalance',
      value=value,
      min=0.0,
      max=1.0,
  )


def weighted_precision(value: float = 0.0) -> types.Score:
  """Creates a Score object for weighted precision."""
  return types.Score(
      metric='Weighted Precision',
      description='Precision weighted by class support',
      value=value,
      min=0.0,
      max=1.0,
  )


def weighted_recall(value: float = 0.0) -> types.Score:
  """Creates a Score object for weighted recall."""
  return types.Score(
      metric='Weighted Recall',
      description='Recall weighted by class support',
      value=value,
      min=0.0,
      max=1.0,
  )


def weighted_f1(value: float = 0.0) -> types.Score:
  """Creates a Score object for weighted F1-score."""
  return types.Score(
      metric='Weighted F1-Score',
      description='F1-Score weighted by class support',
      value=value,
      min=0.0,
      max=1.0,
  )


# Multi-label classification metrics.
def mean_average_precision(value: float = 0.0) -> types.Score:
  """Creates a Score object for mean Average Precision."""
  return types.Score(
      metric='mAP',
      description='Mean Average Precision (Macro)',
      value=value,
      min=0.0,
      max=1.0,
  )


def micro_f1(value: float = 0.0) -> types.Score:
  """Creates a Score object for micro-averaged F1-score."""
  return types.Score(
      metric='Micro F1',
      description='F1-score calculated globally across all labels',
      value=value,
      min=0.0,
      max=1.0,
  )


def macro_f1(value: float = 0.0) -> types.Score:
  """Creates a Score object for macro-averaged F1-score."""
  return types.Score(
      metric='Macro F1',
      description='F1-score averaged per class, treating all classes equally',
      value=value,
      min=0.0,
      max=1.0,
  )


def hamming_loss(value: float = 0.0) -> types.Score:
  """Creates a Score object for Hamming loss."""
  return types.Score(
      metric='Hamming Loss',
      description=(
          'Fraction of incorrectly predicted labels to '
          'the total number of labels'
      ),
      value=value,
      min=0.0,
      max=1.0,
  )


def subset_accuracy(value: float = 0.0) -> types.Score:
  """Creates a Score object for subset accuracy."""
  return types.Score(
      metric='Subset Accuracy',
      description=(
          'Fraction of samples where the predicted '
          'label set is an exact match'
      ),
      value=value,
      min=0.0,
      max=1.0,
  )


def invalid_result_rate(value: float = 0.0) -> types.Score:
  """Creates a Score object for invalid result rate."""
  return types.Score(
      metric='InvalidResultRate',
      description='Invalid result rate',
      value=value,
      min=0.0,
      max=1.0,
  )


def no_result_rate(value: float = 0.0) -> types.Score:
  """Creates a Score object for no result rate."""
  return types.Score(
      metric='MissingResultRate',
      description='Missing result rate',
      value=value,
      min=0.0,
      max=1.0,
  )


@dataclasses.dataclass
class ClassificationReference:
  """A generic ground-truth reference for a classification task."""
  example_id: str
  label_id: str


@dataclasses.dataclass
class MultiLabelClassificationReference:
  """A ground-truth reference for a multi-label classification task."""

  example_id: str
  label_ids: list[str]


def get_embedding_array(
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
      class_labels: Sequence[str],
      weights: jaxtyping.Float[jaxtyping.Array, 'C D'] | None,
      distance_fn: evaluator_lib.DistanceFn = evaluator_lib.dot_product,
      predict_fn: evaluator_lib.PredictFn = functools.partial(
          evaluator_lib.top_k, k=5
      ),
      top_k_value: int = 5,
  ):
    """Initializes the ClassificationEvaluator.

    This evaluator can be configured for a "full pipeline" (generating scores
    from a linear classifier with weights (aka class label embeddings) and then
    computing metrics) or "metrics-only" (if you already have scores).

    Args:
      class_labels: A required sequence of strings representing the class
        labels. The order of labels must correspond to the row order of
        `weights`. This is needed to map score vector indices to human-readable
        labels.
      weights: An optional [num_classes, embedding_dim+1] numpy array where each
        row is the vector representation for a class, including the bias term.
        This is required only when using the `compute_predictions` method. For
        metrics-only evaluation, this can be `None`.
      distance_fn: A callable function used to compute scores between an example
        embedding and the class embedding table. It should accept a (D,) array
        and a (D, V) array and return a (V,) array of scores. Defaults to
        `numpy.dot`.
      predict_fn: A callable function used to compute predictions from the
        scores. It should accept a 1D array of scores and return a 1D array of
        predictions. Defaults to `evaluator_lib.top_k` with `k=5`.
      top_k_value: The integer `k` to use for calculating Top-K accuracy.
        Defaults to 5.

    Raises:
      ValueError: If `top_k_value` is not a positive integer.
    """
    num_classes = len(class_labels)
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
    self.weights = weights
    self.extended_class_labels = list(class_labels)
    self.extended_class_labels.extend([
        INVALID_ANSWER_STR,
        NO_RESPONSE_STR,
    ])
    self.id_by_label = {
        label: label_id
        for label_id, label in enumerate(self.extended_class_labels)
    }
    self.distance_fn = distance_fn
    self.predict_fn = predict_fn
    self.top_k_value = top_k_value

  def get_extended_class_labels(self) -> list[str]:
    return self.extended_class_labels

  def compute_predictions(
      self,
      embeddings_by_id: types.MultiModalEmbeddingCache
  ) -> Mapping[str, np.ndarray]:
    """Computes prediction scores for the given embeddings."""
    if self.weights is None:
      raise ValueError(
          'Weights must be provided during initialization to use'
          ' compute_predictions.'
      )

    classification_scores = {}
    for example_id, embedding_obj in embeddings_by_id.items():
      embedding_array = get_embedding_array(embedding_obj)

      if embedding_array.ndim != 2 or embedding_array.shape[0] == 0:
        raise ValueError(
            'Found missing or malformed embeddings '
            f'for example_id {example_id}. '
            f'Expected shape (N, D) with N > 0, but got shape '
            f'{embedding_array.shape}.'
        )
      example_embedding = embedding_array[0]
      classification_scores[example_id] = self.distance_fn(
          example_embedding, self.weights
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
        of shape `[num_classes]` (or `[num_classes + 2]' when handling invalid
        and missing results).
      references: A sequence of `ClassificationReference` objects, where each
        object contains the ground-truth `example_id` and `label_id`.

    Returns:
      A list of `types.Score` dataclass objects, with each object representing
      one of the computed metrics. Returns an empty list if no matching
      predictions are found for the provided references.
    """
    y_true_labels, y_pred_labels, y_top_k_labels = [], [], []

    for ref in references:
      if ref.example_id in scores:
        y_true_labels.append(self.id_by_label[ref.label_id])
        _, top_k_indices = self.predict_fn(scores[ref.example_id])
        y_pred_labels.append(top_k_indices[0])
        y_top_k_labels.append(top_k_indices[:self.top_k_value])

    if not y_true_labels:
      logger.error('No matching predictions found for the given references.')
      return []

    acc = metrics.accuracy_score(y_true_labels, y_pred_labels)
    top_k_hits = np.any(
        np.array(y_top_k_labels) == np.array(y_true_labels)[:, np.newaxis],
        axis=1,
    )
    top_k_acc = np.mean(top_k_hits)
    bal_acc = metrics.balanced_accuracy_score(y_true_labels, y_pred_labels)
    prec, rec, f1, _ = metrics.precision_recall_fscore_support(
        y_true_labels,
        y_pred_labels,
        average='weighted',
        labels=list(self.id_by_label.values()),
        zero_division=0
    )
    invalid_result = np.mean(
        np.array(y_pred_labels) == self.id_by_label[INVALID_ANSWER_STR]
    )
    no_result = np.mean(
        np.array(y_pred_labels) == self.id_by_label[NO_RESPONSE_STR]
    )

    return [
        accuracy(acc),
        top_k_accuracy(float(top_k_acc), k=self.top_k_value),
        balanced_accuracy(bal_acc),
        weighted_precision(prec),
        weighted_recall(rec),
        weighted_f1(f1),
        invalid_result_rate(invalid_result),
        no_result_rate(no_result),
    ]


class MultiLabelClassificationEvaluator:
  """Generic evaluator for multi-label classification tasks."""

  def __init__(
      self,
      weights: jaxtyping.Float[jaxtyping.Array, 'C D'] | None,
      id_by_class_index: Sequence[str],
      distance_fn: evaluator_lib.DistanceFn = evaluator_lib.dot_product,
  ):
    """Initializes the MultiLabelClassificationEvaluator.

    Args:
      weights: An optional [num_classes, embedding_dim] numpy array
        where each row is the vector representation for a class. This is
        required only when using the `compute_predictions` method.
      id_by_class_index: A required sequence of strings representing all
        possible class labels. The order must correspond to the row order of
        the `embedding_table`.
      distance_fn: A callable function used to compute scores between an example
        embedding and the class embedding table. It should accept a (D,) array
        and a (D, V) array and return a (V,) array of scores. Defaults to
        `numpy.dot`.
    """
    self.weights = weights
    self.id_by_class_index = id_by_class_index
    self.index_by_id = {
        label_id: i for i, label_id in enumerate(id_by_class_index)
    }
    self.distance_fn = distance_fn

  def get_extended_class_labels(self) -> list[str]:
    raise NotImplementedError

  def compute_predictions(
      self, embeddings_by_id: types.MultiModalEmbeddingCache
  ) -> Mapping[str, np.ndarray]:
    """Computes prediction scores for the given embeddings."""
    if self.weights is None:
      raise ValueError(
          'An embedding_table must be provided during initialization to '
          'use compute_predictions.'
      )

    classification_scores = {}
    for example_id, embedding_obj in embeddings_by_id.items():
      embedding_array = get_embedding_array(embedding_obj)

      if embedding_array.ndim != 2 or embedding_array.shape[0] == 0:
        raise ValueError(
            'Found missing or malformed embeddings '
            f'for example_id {example_id}. '
            f'Expected shape (N, D) with N > 0, but got shape '
            f'{embedding_array.shape}.'
        )
      example_embedding = embedding_array[0]
      classification_scores[example_id] = self.distance_fn(
          example_embedding, self.weights
      )
    return classification_scores

  def compute_metrics(
      self,
      scores: Mapping[str, np.ndarray],
      references: Sequence[MultiLabelClassificationReference],
      threshold: float = 0.5,
  ) -> list[types.Score]:
    """Computes a suite of multi-label classification metrics.

    This method calculates ranking-based (mAP) and threshold-based
    (F1, Hamming Loss, etc.) metrics.

    Args:
      scores: A mapping from an example ID to its raw prediction score vector.
      references: A sequence of `MultiLabelClassificationReference` objects.
      threshold: The decision threshold (0.0 to 1.0) to convert scores into
        binary predictions for threshold-based metrics.

    Returns:
      A list of `types.Score` objects representing the computed metrics.
    """
    num_classes = len(self.id_by_class_index)
    y_true = []
    y_scores = []

    # Align ground truth and predictions into binary matrices
    for ref in references:
      if ref.example_id in scores:
        true_labels_binary = np.zeros(num_classes, dtype=int)
        for label_id in ref.label_ids:
          if label_id in self.index_by_id:
            true_labels_binary[self.index_by_id[label_id]] = 1
        y_true.append(true_labels_binary)
        y_scores.append(scores[ref.example_id])

    if not y_true:
      logger.error('No matching predictions found for the given references.')
      return []

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # 1. Ranking-based metric (threshold-independent)
    map_score = metrics.average_precision_score(
        y_true,
        y_scores,
        average='macro'
    )

    # 2. Threshold-based metrics
    y_pred = (y_scores >= threshold).astype(int)

    micro_f1_score = metrics.f1_score(
        y_true,
        y_pred,
        average='micro',
        zero_division=0
    )
    macro_f1_score = metrics.f1_score(
        y_true,
        y_pred,
        average='macro',
        zero_division=0
    )
    h_loss = metrics.hamming_loss(y_true, y_pred)
    sub_acc = metrics.accuracy_score(y_true, y_pred)

    return [
        mean_average_precision(map_score),
        micro_f1(micro_f1_score),
        macro_f1(macro_f1_score),
        hamming_loss(h_loss),
        subset_accuracy(sub_acc),
    ]


def load_linear_classifier(base_dir: str) -> tuple[
    Sequence[str],
    jaxtyping.Float[jaxtyping.Array, 'C D'],
]:
  """Loads the linear classifier from a directory."""
  logger.info('Loading weights and bias from %s', base_dir)

  weights_path = epath.Path(os.path.join(base_dir, 'weights.npy'))
  class_labels_path = epath.Path(os.path.join(base_dir, 'class_labels.txt'))
  if not weights_path.exists() or not class_labels_path.exists():
    raise FileNotFoundError(
        'Weights or class labels not found in directory: %s' % base_dir
    )
  with weights_path.open('rb') as f:
    weights = np.load(f)
  with class_labels_path.open('r') as f:
    class_labels = f.read().splitlines()
  return class_labels, weights


def save_linear_classifier(
    class_labels: Sequence[str],
    weights: jaxtyping.Float[jaxtyping.Array, 'C D'],
    base_dir: str,
):
  """Saves the linear classifier to a directory."""
  logger.info('Saving weights and bias to %s', base_dir)
  epath.Path(base_dir).mkdir(parents=True, exist_ok=True)
  with epath.Path(os.path.join(base_dir, 'weights.npy')).open('wb') as f:
    np.save(f, weights)
  with epath.Path(os.path.join(base_dir, 'class_labels.txt')).open('w') as f:
    f.write('\n'.join(class_labels))
