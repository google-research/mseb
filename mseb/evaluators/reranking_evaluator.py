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

"""Evaluator for reranking tasks."""

from __future__ import annotations

import dataclasses
from typing import Callable, Mapping, Sequence

import jaxtyping
from mseb import evaluator
from mseb import metrics
from mseb import types
import numpy as np
from sklearn import metrics as sklearn_metrics
from whisper.normalizers import basic
from whisper.normalizers import english


def map(value: float = 0.0, std: float | None = None):  # pylint: disable=redefined-builtin
  return types.Score(
      metric='MAP',
      description='Mean Average Precision',
      value=value,
      min=0,
      max=1,
      std=std,
  )


def mrr(value: float = 0.0, std: float | None = None):
  return types.Score(
      metric='MRR',
      description='Mean Reciprocal Rank',
      value=value,
      min=0,
      max=1,
      std=std,
  )


def wer(value: float = 0.0, std: float | None = None):
  return types.Score(
      metric='WER',
      description='Word Error Rate',
      value=value,
      min=0,
      max=float('inf'),
      std=std,
  )


def cer(value: float = 0.0, std: float | None = None):
  return types.Score(
      metric='CER',
      description='Candidate Error Rate',
      value=value,
      min=0,
      max=float('inf'),
      std=std,
  )


@dataclasses.dataclass
class RerankingCandidates:
  sound_id: str
  texts: Sequence[str]
  language: str  # For text normalization.


RerankingPredictionsCache = Mapping[str, tuple[Sequence[float], Sequence[str]]]


class RerankingEvaluator:
  """Evaluator for reranking tasks."""

  def __init__(
      self,
      candidate_embeddings_by_sound_id: Mapping[
          str, Sequence[types.MultiModalEmbedding]
      ],
      distance_fn: evaluator.DistanceFn = evaluator.dot_product,
      predict_fn: evaluator.PredictFn = evaluator.top_inf,
      mrr_at_k: int = 10,
  ):
    """Initializes the reranking evaluator.

    Args:
      candidate_embeddings_by_sound_id: A dictionary mapping sound_id to a
        sequence of candidate embeddings.
      distance_fn: The distance function to use for computing the scores.
      predict_fn: The function to use for computing the predictions.
      mrr_at_k: Computes MRR @ `mrr_at_k`.
    """
    self.candidate_embeddings_by_sound_id = candidate_embeddings_by_sound_id
    self.distance_fn = distance_fn
    self.predict_fn = predict_fn
    self.mrr_at_k = mrr_at_k

  def compute_predictions(
      self, embeddings_by_sound_id: types.MultiModalEmbeddingCache
  ) -> RerankingPredictionsCache:
    """Evaluates reranking quality for a single example.

    Args:
      embeddings_by_sound_id: The embeddings to evaluate.

    Returns:
      A dictionary mapping sound_id to a tuple of (ranked_candidate_scores,
      ranked_candidate_texts).
    """
    predictions = {}
    for sound_id, embeddings in embeddings_by_sound_id.items():
      assert hasattr(embeddings, 'embedding')
      embedding: jaxtyping.Float[jaxtyping.Array, '1 D'] = embeddings.embedding
      candidate_embeddings = self.candidate_embeddings_by_sound_id[sound_id]
      embeddings = []
      for embeds in candidate_embeddings:
        assert hasattr(embeds, 'embedding')
        embed: jaxtyping.Float[jaxtyping.Array, '1 D'] = embeds.embedding
        embeddings.append(embed[0])
      scores = self.distance_fn(embedding[0], np.array(embeddings))
      ranked_candidate_scores, ranked_candidate_ids = self.predict_fn(scores)
      texts = [text.context.id for text in candidate_embeddings]
      ranked_candidate_texts: Sequence[str] = [
          texts[x] for x in ranked_candidate_ids
      ]
      predictions[sound_id] = (ranked_candidate_scores, ranked_candidate_texts)
    return predictions

  def compute_metrics(
      self,
      predictions: RerankingPredictionsCache,
      candidates_batch: Sequence[RerankingCandidates],
  ) -> list[types.Score]:
    """Returns quality metrics of the predictions."""

    def text_transform(language: str) -> Callable[[str], str]:
      if language.split('_')[0].lower() == 'en':
        return english.EnglishTextNormalizer()
      else:
        return basic.BasicTextNormalizer()

    values_by_metric: dict[str, list[types.WeightedValue]] = {
        'map': [],
        'mrr': [],
        'wer': [],
        'cer': [],
    }
    for candidates in candidates_batch:
      ranked_candidate_scores, ranked_candidate_texts = predictions[
          candidates.sound_id
      ]

      word_errors, word_errors_weight = metrics.compute_word_errors(
          truth=candidates.texts[0],
          hypothesis=ranked_candidate_texts[0],
          text_transform=text_transform(candidates.language),
      )
      values_by_metric['wer'].append(
          types.WeightedValue(
              value=word_errors / word_errors_weight,
              weight=word_errors_weight,
          )
      )
      values_by_metric['cer'].append(
          types.WeightedValue(
              value=float(
                  metrics.compute_word_errors(
                      truth=candidates.texts[0],
                      hypothesis=ranked_candidate_texts[0],
                      text_transform=text_transform(candidates.language),
                  )[0]
                  != 0.0
              )
          )
      )
      values_by_metric['mrr'].append(
          types.WeightedValue(
              value=metrics.compute_reciprocal_rank(
                  candidates.texts[0], ranked_candidate_texts[: self.mrr_at_k]
              ),
          )
      )
      values_by_metric['map'].append(
          types.WeightedValue(
              value=sklearn_metrics.average_precision_score(
                  y_true=[
                      metrics.compute_word_errors(
                          truth=candidates.texts[0],
                          hypothesis=text,
                          text_transform=text_transform(candidates.language),
                      )[0]
                      == 0.0
                      for text in ranked_candidate_texts
                  ],
                  y_score=ranked_candidate_scores,
              ),
          )
      )

    map_score = map(
        *evaluator.compute_weighted_average_and_std(values_by_metric['map'])
    )
    mrr_score = mrr(
        *evaluator.compute_weighted_average_and_std(values_by_metric['mrr'])
    )
    wer_score = wer(
        *evaluator.compute_weighted_average_and_std(values_by_metric['wer'])
    )
    cer_score = cer(
        *evaluator.compute_weighted_average_and_std(values_by_metric['cer'])
    )
    return [map_score, wer_score, cer_score, mrr_score]
