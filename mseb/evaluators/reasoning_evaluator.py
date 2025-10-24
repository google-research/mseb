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


NO_ANSWER_STR = 'No Answer'


def f1(value: float = 0.0, std: float | None = None):
  return types.Score(
      metric='F1',
      description='F1 score',
      value=value,
      min=0,
      max=1,
      std=std,
  )


def gmean_f1(value: float = 0.0, std: float | None = None):
  """geometric mean of f1('No Answer's) and f1(real answers).

  Motivation:
    - f1('No Answer's) ~ f1(real answers): same as original f1.
    - trivial solution (all examples are assigned 'No Answer'): gmean-f1=0 vs
      f1=p where p is the proportion of 'No Answer' examples (often ~50%, which
    looks competitive with f1 numbers for gemma).

  Args:
    value: geometric mean of f1('No Answer's) and f1(real answers)
    std: standard deviation of geometric mean of f1('No Answer's) and f1(real
      answers)

  Returns:
    A types.Score object.
  """
  return types.Score(
      metric='GmeanF1',
      description='Geometric mean F1 score',
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
  if target == NO_ANSWER_STR or prediction == NO_ANSWER_STR:
    return float(target == prediction)
  prediction_tokens = prediction.split()
  target_tokens = target.split()
  common = collections.Counter(prediction_tokens) & collections.Counter(
      target_tokens
  )
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


ReasoningPredictionsCache = Mapping[str, types.ReasoningPrediction]


class ReasoningEvaluator:
  """Evaluator for reasoning tasks."""

  def __init__(
      self,
      span_embeddings_by_sound_id: Mapping[
          str, Sequence[types.MultiModalEmbedding]
      ],
      distance_fn: evaluator.DistanceFn = evaluator.dot_product,
      predict_fn: evaluator.PredictFn = evaluator.top_1,
      no_answer_threshold: float = 0.0,
  ):
    self.span_embeddings_by_sound_id = span_embeddings_by_sound_id
    self.distance_fn = distance_fn
    self.predict_fn = predict_fn
    self.no_answer_threshold = no_answer_threshold

  def compute_predictions(
      self,
      embeddings_by_sound_id: types.MultiModalEmbeddingCache,
  ) -> ReasoningPredictionsCache:
    """Computes the best matching span.

    If the score of the best span exceeds the no_answer_threshold, the text of
    the best span is returned. Otherwise, NO_ANSWER_STR is returned.

    Args:
      embeddings_by_sound_id: The sound embeddings.

    Returns:
      A mapping from sound_id to the predicted answer string.
    """
    predictions = {}
    for sound_id, embeddings in embeddings_by_sound_id.items():
      assert hasattr(embeddings, 'embedding')
      embedding: jaxtyping.Float[jaxtyping.Array, '1 D'] = embeddings.embedding
      span_embeddings = self.span_embeddings_by_sound_id[sound_id]
      if span_embeddings:
        embeddings = []
        for embeds in span_embeddings:
          assert hasattr(embeds, 'embedding')
          embed: jaxtyping.Float[jaxtyping.Array, '1 D'] = embeds.embedding
          embeddings.append(embed[0])
        scores = self.distance_fn(embedding[0], np.array(embeddings))
        top_span_score, top_span_id = self.predict_fn(scores)
        texts = [text.context.id for text in span_embeddings]
        prediction = (
            NO_ANSWER_STR
            if top_span_score[0] < self.no_answer_threshold
            else texts[top_span_id[0]]
        )
      else:
        prediction = NO_ANSWER_STR
      predictions[sound_id] = types.ReasoningPrediction(
          answer=prediction,
          context=types.ReasoningContextParams(id=sound_id),
      )
    return predictions

  def compute_metrics(
      self,
      predictions: ReasoningPredictionsCache,
      spans_batch: Sequence[ReasoningSpans],
  ) -> list[types.Score]:
    """Returns quality metrics of the predictions."""
    values_by_metric: dict[str, list[types.WeightedValue]] = {
        'f1': [],
        'f1_no_answer': [],
    }
    for spans in spans_batch:
      f1_value = compute_f1_score(
          spans.reference_answer, predictions[spans.sound_id].answer
      )
      if spans.reference_answer == NO_ANSWER_STR:
        values_by_metric['f1_no_answer'].append(
            types.WeightedValue(value=f1_value, weight=1.0)
        )
      else:
        values_by_metric['f1'].append(
            types.WeightedValue(value=f1_value, weight=1.0)
        )

    f1_score = f1(
        *evaluator.compute_weighted_average_and_std(
            values_by_metric['f1'] + values_by_metric['f1_no_answer']
        )
    )
    weight = len(values_by_metric['f1']) / (
        len(values_by_metric['f1']) + len(values_by_metric['f1_no_answer'])
    )
    weight_no_answer = 1.0 - weight
    if values_by_metric['f1']:
      mean, _ = evaluator.compute_weighted_average_and_std(
          values_by_metric['f1']
      )
    else:
      mean = 0.0
    if values_by_metric['f1_no_answer']:
      mean_no_answer, _ = evaluator.compute_weighted_average_and_std(
          values_by_metric['f1_no_answer']
      )
    else:
      mean_no_answer = 0.0
    gmean_f1_score = gmean_f1(mean**weight * mean_no_answer**weight_no_answer)
    return [gmean_f1_score, f1_score]
