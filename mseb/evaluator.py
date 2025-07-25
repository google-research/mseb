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

"""MSEB Evaluator base class."""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Sequence, Union

from mseb import encoder
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


class Evaluator(abc.ABC):
  """The MSEB evaluator's base class.

  Each task should implement an instance of this class.
  """

  def __init__(self,
               sound_encoder: encoder.Encoder,
               encode_kwargs: dict[str, Any]):
    self.sound_encoder = sound_encoder
    self.encode_kwargs = encode_kwargs

  @abc.abstractmethod
  def __call__(self,
               sequence: Union[str, Sequence[float]],
               context: encoder.ContextParams,
               **kwargs: Any,
               ) -> Dict[str, float]:
    """Evaluates quality of the encoder for input sequence and return metrics.

    Args:
      sequence: Input sound sequence to encode. String-type sequences are
        interpreted as sound file paths.
      context: Encoder input context parameters.
      **kwargs: Additional arguments to pass to the evaluator.

    Returns:
      Dictionary of metrics.
    """
    ...

  def evaluate(
      self,
      sequence: Union[str, Sequence[float]],
      context: encoder.ContextParams,
      **kwargs: Any,
  ) -> Dict[str, float]:
    """Evaluates quality of the encoder for input sequence and return metrics."""
    return self.__call__(sequence, context, **kwargs)

  def evaluate_batch(
      self,
      sequences: Sequence[Union[str, Sequence[float]]],
      contexts: Sequence[encoder.ContextParams],
      **kwargs: Any,
  ) -> Sequence[Dict[str, float]]:
    """Evaluates quality of the encoder for a batch of input sequences.

    Args:
      sequences: Input sound sequences to encode. String-type sequences are
        interpreted as sound file paths.
      contexts: Encoder input context parameters.
      **kwargs: Additional arguments to pass to the evaluator.

    Returns:
      List of dictionaries of metrics.
    """
    return [self.evaluate(sequence, context, **kwargs)
            for sequence, context in zip(sequences, contexts)]

  @abc.abstractmethod
  def combine_scores(self, scores: List[Dict[str, float]]) -> Dict[str, float]:
    """Combines individual example scores into a merged score dictionary.

    This abstract method must be implemented by subclasses to define
    how the scores from multiple examples are aggregated (e.g., averaged,
    summed, or based on specific metrics).

    Args:
      scores: A list of dictionaries, where each dictionary represents
              the metrics and their values for a single example (e.g.,
              `[{'accuracy': 0.9, 'f1': 0.8}, {'accuracy': 0.85, 'f1': 0.82}]`).

    Returns:
      A single dictionary containing the combined/aggregated scores.
      The keys represent metric names and values are their combined results.
    """
    ...


def compute_weighted_average_and_std(
    scores: List[Dict[str, float]],
    statistic_metric_pairs: Sequence[tuple[str, str]],
    weight_suffix: str = '_weight',
) -> Dict[str, float]:
  """Computes weighted average and standard deviation for a list of scores.

  Args:
    scores: A list of dictionaries, where each dictionary represents the
      statistics and possibly weights for a single example.
    statistic_metric_pairs: A list of tuples, where each tuple represents a
      statistic name and its corresponding metric name.
    weight_suffix: The suffix to use for the weight key.

  Returns:
    A dictionary containing the weighted average and standard deviation
    for each metric. The keys represent metric names and values are
    their combined results.
  """
  statistics_by_name = {}
  weights_by_name = {}
  for s, _ in statistic_metric_pairs:
    statistics_by_name[s] = [score[s] for score in scores]
    weights_by_name[s] = [
        score.get(f'{s}{weight_suffix}', 1) for score in scores
    ]

  combined_scores = {}
  for s, m in statistic_metric_pairs:
    mean = np.average(
        np.array(statistics_by_name[s]),
        weights=np.array(weights_by_name[s]),
    )
    std = (
        np.average(
            (np.array(statistics_by_name[s]) - mean) ** 2,
            weights=weights_by_name[s],
        )
        ** 0.5
    )
    combined_scores[m] = float(mean)
    combined_scores[m + '_std'] = float(std)

  return combined_scores


def compute_weighted_average_and_std_v2(
    scores_list: list[list[types.Score]],
    statistic_metric_pairs: Sequence[tuple[str, str]],
) -> list[types.Score]:
  """Computes weighted average and stddev for a list of list of scores.

  Args:
    scores_list: A list of lists of Score objects, where each inner list
      contains the scores for a single evaluated example.
    statistic_metric_pairs: A list of tuples, where each tuple represents a
      statistic name and its corresponding metric name.

  Returns:
    A list of Score objects containing the weighted average and standard
    deviation for each metric.
  """
  values_by_metric = {k: [] for k, _ in statistic_metric_pairs}
  weights_by_metric = {k: [] for k, _ in statistic_metric_pairs}
  mins_by_metric = {k: [] for k, _ in statistic_metric_pairs}
  maxs_by_metric = {k: [] for k, _ in statistic_metric_pairs}
  description_by_metric = {}
  for scores in scores_list:
    for score in scores:
      if score.metric in values_by_metric:
        values_by_metric[score.metric].append(score.value)
        weights_by_metric[score.metric].append(score.weight)
        mins_by_metric[score.metric].append(score.min)
        maxs_by_metric[score.metric].append(score.max)
        description_by_metric[score.metric] = score.description

  final_scores = []
  for s, m in statistic_metric_pairs:
    mean = np.average(
        np.array(values_by_metric[s]),
        weights=np.array(weights_by_metric[s]),
    )
    std = (
        np.average(
            (np.array(values_by_metric[s]) - mean) ** 2,
            weights=weights_by_metric[s],
        )
        ** 0.5
    )
    min_value = float(np.min(mins_by_metric[s]))
    max_value = float(np.max(maxs_by_metric[s]))
    final_scores.append(
        types.Score(
            metric=m,
            description=description_by_metric[s],
            value=float(mean),
            min=min_value,
            max=max_value,
        )
    )
    final_scores.append(
        types.Score(
            metric=m + '_std',
            description=description_by_metric[s],
            value=float(std),
            min=0.0,
            max=(max_value - min_value) / 2,
        )
    )

  return final_scores
