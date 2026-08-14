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

"""Evaluator for reasoning tasks."""

from __future__ import annotations

import collections
import dataclasses
import re
import string
from typing import Mapping, Sequence

from absl import logging
import jaxtyping
from mseb import encoder
from mseb import evaluator
from mseb import types
import numpy as np
from scipy import stats
from sklearn import mixture

GaussianMixture = mixture.GaussianMixture
NO_ANSWER_STR = 'No Answer'
INVALID_ANSWER_STR = encoder.INVALID_ANSWER_STR
NO_RESPONSE_STR = encoder.NO_RESPONSE_STR


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
  if prediction == INVALID_ANSWER_STR:
    return 0.0
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


def _find_decision_boundary(gmm: GaussianMixture) -> float:
  """Finds the decision boundary of a 1D 2-component GaussianMixture model.

  The decision boundary is the point x* such that pdf_1(x*) == pdf_2(x*), where
  pdf_i is the probability density function of the i-th component (cf. Bayes
  decision rule).

  Args:
    gmm: A fitted 1D 2-component scikit-learn GaussianMixture model.

  Returns:
    The decision boundary as a float.
  """
  assert gmm.n_components == 2
  assert gmm.weights_ is not None and gmm.weights_.shape == (2,)
  assert gmm.means_ is not None and gmm.means_.shape == (2, 1)
  assert gmm.covariances_ is not None and gmm.covariances_.shape == (2, 1, 1)

  pi1, pi2 = gmm.weights_
  mu1, mu2 = gmm.means_.flatten()
  var1, var2 = gmm.covariances_.flatten()
  sigma1, sigma2 = np.sqrt(var1), np.sqrt(var2)

  # Quadratic coefficients: A*x^2 + B*x + C = 0
  A = (1.0 / (2 * var1)) - (1.0 / (2 * var2))  # pylint: disable=invalid-name
  B = (mu2 / var2) - (mu1 / var1)  # pylint: disable=invalid-name
  C = (  # pylint: disable=invalid-name
      (mu1**2 / (2 * var1))
      - (mu2**2 / (2 * var2))
      - np.log((pi1 * sigma2) / (pi2 * sigma1))
  )

  # Equal variance case (Linear equation)
  if np.isclose(A, 0):
    x_intersect = -C / B
    return float(x_intersect)

  # Unequal variance case (Quadratic equation)
  discriminant = B**2 - 4 * A * C
  if discriminant < 0:
    return float((mu1 + mu2) / 2)  # Fallback to midpoint.

  x1 = (-B + np.sqrt(discriminant)) / (2 * A)
  x2 = (-B - np.sqrt(discriminant)) / (2 * A)

  y1 = pi1 * stats.norm.pdf(x1, loc=mu1, scale=sigma1)
  y2 = pi2 * stats.norm.pdf(x2, loc=mu2, scale=sigma2)
  return float(x1) if y1 > y2 else float(x2)


def find_no_answer_threshold_by_gmm(scores: Sequence[float]) -> float:
  """Finds the threshold by fitting a 1D 2-component GaussianMixture model.

  One component represents the scores of the no-answer predictions, while the_
  other component represents the scores of the answer predictions.

  Args:
    scores: The scores of the predictions.

  Returns:
    The threshold for the no-answer predictions.
  """
  gmm = GaussianMixture(n_components=2, random_state=42)
  gmm.fit(np.array(scores)[:, np.newaxis])
  threshold = _find_decision_boundary(gmm)
  logging.info('no-answer threshold: %f', threshold)
  return threshold


@dataclasses.dataclass
class ReasoningSpans:
  sound_id: str
  reference_answer: str
  texts: Sequence[str]


ReasoningPredictionsCache = Mapping[str, types.TextPrediction]


class ReasoningEvaluator:
  """Evaluator for reasoning tasks.

  Attributes:
    span_embeddings_by_sound_id: A mapping from sound_id to a sequence of span
      embeddings.
    distance_fn: The distance function to use for the predictions.
    predict_fn: The predict function to use for the predictions.
    no_answer_threshold: The threshold for the no-answer predictions. If None,
      the threshold is determined by a GaussianMixture model fitted to the
      scores of the predictions.
  """

  def __init__(
      self,
      span_embeddings_by_sound_id: Mapping[
          str, Sequence[types.MultiModalEmbedding]
      ],
      distance_fn: evaluator.DistanceFn = evaluator.dot_product,
      predict_fn: evaluator.PredictFn = evaluator.top_1,  # pyrefly: ignore[bad-function-definition]
      no_answer_threshold: float | None = None,
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
    raw_predictions = {}
    for sound_id, embeddings in embeddings_by_sound_id.items():
      assert hasattr(embeddings, 'embedding')
      embedding: jaxtyping.Float[jaxtyping.Array, '1 D'] = embeddings.embedding  # pyrefly: ignore[bad-assignment]
      span_embeddings = self.span_embeddings_by_sound_id[sound_id]
      if span_embeddings:
        embeddings = []
        for embeds in span_embeddings:
          assert hasattr(embeds, 'embedding')
          embed: jaxtyping.Float[jaxtyping.Array, '1 D'] = embeds.embedding  # pyrefly: ignore[bad-assignment]
          embeddings.append(embed[0])
        scores = self.distance_fn(embedding[0], np.array(embeddings))  # pyrefly: ignore[bad-argument-type]
        top_span_score, top_span_id = self.predict_fn(scores)
        texts = [text.context.id for text in span_embeddings]
        prediction = (texts[top_span_id[0]], top_span_score[0])
      else:
        prediction = NO_ANSWER_STR
      raw_predictions[sound_id] = prediction

    no_answer_threshold = self.no_answer_threshold
    if no_answer_threshold is None:
      no_answer_threshold = find_no_answer_threshold_by_gmm([
          pred[1] for pred in raw_predictions.values() if pred != NO_ANSWER_STR
      ])

    predictions = {}
    for sound_id, raw_prediction in raw_predictions.items():
      if raw_prediction == NO_ANSWER_STR:
        prediction = NO_ANSWER_STR
      else:
        prediction, score = raw_prediction
        if score < no_answer_threshold:
          prediction = NO_ANSWER_STR
      predictions[sound_id] = types.TextPrediction(
          prediction=prediction,
          context=types.PredictionContextParams(id=sound_id),
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
          spans.reference_answer, predictions[spans.sound_id].prediction
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
    invalid_result_rate, invalid_result_rate_std = (
        evaluator.compute_weighted_average_and_std([
            types.WeightedValue(
                value=float(
                    predictions[spans.sound_id].prediction == INVALID_ANSWER_STR
                ),
                weight=1.0,
            )
            for spans in spans_batch
        ])
    )
    invalid_result_score = types.Score(
        metric='InvalidResultRate',
        description='Invalid result rate',
        value=invalid_result_rate,
        min=0,
        max=1,
        std=invalid_result_rate_std,
    )
    no_result_rate, no_result_rate_std = (
        evaluator.compute_weighted_average_and_std([
            types.WeightedValue(
                value=float(
                    predictions[spans.sound_id].prediction == NO_RESPONSE_STR
                ),
                weight=1.0,
            )
            for spans in spans_batch
        ])
    )
    no_result_score = types.Score(
        metric='MissingResultRate',
        description='Missing result rate',
        value=no_result_rate,
        min=0,
        max=1,
        std=no_result_rate_std,
    )
    return [gmean_f1_score, f1_score, invalid_result_score, no_result_score]
