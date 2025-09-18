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

"""Evaluator for reasoning tasks."""

from __future__ import annotations

import collections
import dataclasses
import re
import string
from typing import Mapping, Sequence

import jaxtyping
from mseb import evaluator
from mseb import types
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs


def f1(value: float = 0.0, std: float | None = None):
  return types.Score(
      metric='F1',
      description='F1 score',
      value=value,
      min=0,
      max=1,
      std=std,
  )


def _normalize_answer(text: str, punc_chars: str, punc_repl: str) -> str:
  """Lower text and remove punctuation, articles and extra whitespace."""

  def remove_articles(s):
    return re.sub(r'\b(a|an|the)\b', ' ', s)

  def replace_punctuation(s):
    to_replace = set(punc_chars)
    return ''.join(punc_repl if ch in to_replace else ch for ch in s)

  def white_space_fix(s):
    return ' '.join(s.split())

  text = text.lower()
  text = replace_punctuation(text)
  text = remove_articles(text)
  text = white_space_fix(text)

  return text


def normalize_squad(answer: str) -> str:
  """Normalization used in official SQuAD evaluation script."""
  return _normalize_answer(answer, punc_chars=string.punctuation, punc_repl='')


def compute_f1_score(target: str, prediction: str) -> float:
  """Token-based F1 score used XTREME-UP."""
  prediction_tokens = prediction.split()
  target_tokens = target.split()
  common = (collections.Counter(prediction_tokens) &
            collections.Counter(target_tokens))
  num_same = sum(common.values())
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(target_tokens)
  f1_score = (2 * precision * recall) / (precision + recall)
  return f1_score


@dataclasses.dataclass
class ReasoningSpans:
  sound_id: str
  reference_answer: str
  texts: Sequence[str]


ReasoningPredictionsCache = Mapping[str, str]


class ReasoningEvaluator:
  """Evaluator for reasoning tasks."""

  def __init__(
      self,
      span_embeddings_by_text: types.TextEmbeddingCache,
      no_answer_threshold: float,
  ):
    self.span_embeddings_by_text = span_embeddings_by_text
    self.no_answer_threshold = no_answer_threshold

  def __call__(
      self,
      embeddings: types.SoundEmbeddingCache,
      spans_batch: Sequence[ReasoningSpans],
  ) -> list[types.Score]:
    """Evaluates reasoning quality for a batch of examples.

    Args:
      embeddings: The embeddings to evaluate.
      spans_batch: The reference spans to evaluate.

    Returns:
      A list of Score objects containing the final, aggregated scores. The
      scores include F1 score.
    """
    return self.evaluate_predictions(
        self.compute_predictions(embeddings, spans_batch), spans_batch
    )

  def compute_predictions(
      self,
      embeddings: types.SoundEmbeddingCache,
      spans_batch: Sequence[ReasoningSpans],
  ) -> ReasoningPredictionsCache:
    """Computes the predictions for the given embeddings."""
    sound_embedding = next(iter(embeddings.values()))
    if np.issubdtype(sound_embedding.embedding.dtype, np.floating):
      return self.compute_predictions_float_embedding(embeddings, spans_batch)
    else:
      return self.compute_predictions_string_embedding(embeddings, spans_batch)

  def compute_predictions_float_embedding(
      self,
      embeddings: types.SoundEmbeddingCache,
      spans_batch: Sequence[ReasoningSpans],
  ) -> ReasoningPredictionsCache:
    """Computes the best matching span.

    If the score of the best span exceeds the no_answer_threshold, the text of
    the best span is returned. Otherwise, 'No Answer' is returned.

    Args:
      embeddings: The sound embeddings.
      spans_batch: The reference spans for each sound.

    Returns:
      A mapping from sound_id to the predicted answer string.
    """
    predictions = {}
    for spans in spans_batch:
      embedding: jaxtyping.Float[jaxtyping.Array, '1 D'] = embeddings[
          spans.sound_id
      ].embedding
      searcher = tfrs.layers.factorized_top_k.BruteForce(k=1)
      span_embeddings = [
          self.span_embeddings_by_text[text].embeddings[0]
          for text in spans.texts
      ]
      searcher.index(candidates=tf.constant(span_embeddings, dtype=tf.float32))
      top_span_score, top_span_id = searcher(
          tf.constant(embedding, dtype=tf.float32)
      )
      top_span_score = float(top_span_score[0].numpy())
      top_span_text = spans.texts[int(top_span_id[0].numpy())]
      prediction = (
          'No Answer'
          if top_span_score < self.no_answer_threshold
          else top_span_text
      )
      predictions[spans.sound_id] = prediction
    return predictions

  def compute_predictions_string_embedding(
      self,
      embeddings: types.SoundEmbeddingCache,
      spans_batch: Sequence[ReasoningSpans],
  ) -> ReasoningPredictionsCache:
    """Copies the predictions for the given string embeddings."""
    predictions = {}
    for spans in spans_batch:
      embedding: jaxtyping.Shaped[np.ndarray, '1'] = embeddings[
          spans.sound_id
      ].embedding
      predictions[spans.sound_id] = str(embedding[0])
    return predictions

  def evaluate_predictions(
      self,
      predictions: ReasoningPredictionsCache,
      spans_batch: Sequence[ReasoningSpans],
  ) -> list[types.Score]:
    """Returns quality metrics of the predictions."""
    values_by_metric: dict[str, list[types.WeightedValue]] = {'f1': []}
    for spans in spans_batch:
      values_by_metric['f1'].append(
          types.WeightedValue(
              value=compute_f1_score(
                  spans.reference_answer, predictions[spans.sound_id]
              ),
              weight=1.0,
          )
      )

    f1_score = f1(
        *evaluator.compute_weighted_average_and_std(values_by_metric['f1'])
    )
    return [f1_score]
