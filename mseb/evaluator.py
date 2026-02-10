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

"""MSEB Evaluator base class."""

from __future__ import annotations

import abc
from typing import Any, Callable, Sequence

import jaxtyping
from mseb import types
import numpy as np


class SoundEmbeddingEvaluator(abc.ABC):
  """The base class for all MSEB evaluators.

  This component is responsible for calculating metrics from an encoder's
  output. It is designed to be stateless and has no direct knowledge of the
  encoder that produced the embeddings.
  """

  def __init__(self, **kwargs: Any):
    """Initializes the evaluator with metric-specific parameters.

    Args:
      **kwargs: Keyword arguments for the specific evaluation metric.
    """
    self._kwargs = kwargs

  @abc.abstractmethod
  def evaluate(
      self,
      waveform_embeddings: np.ndarray,
      embedding_timestamps: np.ndarray,
      params: types.SoundContextParams,
      **kwargs: Any,
  ) -> list[types.Score]:
    """Evaluates the quality of embeddings for a single example.

    Subclasses MUST implement this method.

    Args:
      waveform_embeddings: A 2D array of shape (n, embedding_dim) from the
        encoder.
      embedding_timestamps: A 2D array of shape (m, 2) from the encoder, where
        each row is a [start, end] pair by waveform index.
      params: The sound context parameters for the waveform embeddings. These
        parameters can be used as a source of ground truth labels for scoring.
      **kwargs: Additional runtime arguments to pass to the evaluator.

    Returns:
      A list of Score objects for the single example. There can be multiple
      scores if the evaluator computes multiple related metrics.
    """
    ...

  def evaluate_batch(
      self,
      encoder_outputs_batch: Sequence[tuple[np.ndarray, np.ndarray]],
      params_batch: Sequence[types.SoundContextParams],
      **kwargs: Any,
  ) -> list[list[types.Score]]:
    """Evaluates a batch of examples.

    This is a default, non-performant implementation that processes items
    serially. For optimal performance, subclasses SHOULD override this method
    with a truly batched or vectorized implementation if possible for the given
    metric.

    Args:
      encoder_outputs_batch: A sequence of
        (waveform_embeddings, embedding_timestamps) tuples, one for each example
        in the batch.
      params_batch: A sequence of `SoundContextParams` objects, each
        corresponding to an example.
      **kwargs: Additional runtime arguments.

    Returns:
      A list of score lists, where each inner list contains the Score
      objects from a single evaluated example.
    """
    return [
        self.evaluate(
            waveform_embeddings=encoder_outputs[0],
            embedding_timestamps=encoder_outputs[1],
            params=params,
            **kwargs,
        )
        for encoder_outputs, params in zip(encoder_outputs_batch, params_batch)
    ]

  @abc.abstractmethod
  def combine_scores(
      self, scores_per_example: list[list[types.Score]]
  ) -> list[types.Score]:
    """Combines scores from all examples into a final aggregated list of Scores.

    Subclasses MUST implement this method to define how the scores for their
    specific metric should be aggregated (e.g., by arithmetic or geometric
    averaging).

    Args:
      scores_per_example: A sequence of lists, where each inner list contains
        the Score objects from a single evaluated example.

    Returns:
      A list of Score objects containing the final, aggregated scores.
    """
    ...


def compute_weighted_average_and_std(
    values: list[types.WeightedValue],
) -> tuple[float, float]:
  """Computes weighted average and standard deviation for a list of values."""

  weights = np.array([x.weight for x in values])
  mean = np.average(
      np.array([x.value for x in values]),
      weights=weights,
  )
  std = (
      np.average(
          (np.array([x.value for x in values]) - mean) ** 2,
          weights=weights,
      )
      ** 0.5
  )
  return mean, std


DistanceFn = Callable[
    [
        jaxtyping.Float[jaxtyping.Array, '*B D'],
        jaxtyping.Float[jaxtyping.Array, 'N D'],
    ],
    jaxtyping.Float[jaxtyping.Array, '*B N'],
]

dot_product = lambda x, y: np.dot(x, y.T)


PredictFn = Callable[
    [jaxtyping.Float[jaxtyping.Array, '*B N']],
    jaxtyping.Int[jaxtyping.Array, '*B k'],
]


def top_1(
    scores: jaxtyping.Float[jaxtyping.Array, '*B N'],
) -> tuple[
    jaxtyping.Float[jaxtyping.Array, '*B 1'],
    jaxtyping.Int[jaxtyping.Array, '*B 1'],
]:
  """Returns the top-1 value and its index of scores."""
  top_id = np.argmax(scores)
  return np.array([scores[top_id]]), np.array([top_id])


def top_k(scores: jaxtyping.Float[jaxtyping.Array, '*B N'], k: int) -> tuple[
    jaxtyping.Float[jaxtyping.Array, '*B k'],
    jaxtyping.Int[jaxtyping.Array, '*B k'],
]:
  """Returns top k values and their indices of scores."""
  k = min(k, len(scores))
  ids_k = np.argpartition(scores, -k, axis=-1)[-k:]
  ids = np.argsort(scores[ids_k], axis=-1)[::-1]
  ids_k = ids_k[ids]
  return scores[ids_k], ids_k


def top_inf(scores: jaxtyping.Float[jaxtyping.Array, '*B N']) -> tuple[
    jaxtyping.Float[jaxtyping.Array, '*B N'],
    jaxtyping.Int[jaxtyping.Array, '*B N'],
]:
  """Returns the values and their indices of scores in descending order."""
  ids = np.argsort(-scores, axis=-1)
  return scores[ids], ids
