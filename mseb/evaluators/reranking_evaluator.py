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
from typing import Dict, Mapping, Sequence

import jiwer
from mseb import evaluator
from mseb import metrics
from mseb import types
from sklearn import metrics as sklearn_metrics
import tensorflow as tf
import tensorflow_recommenders as tfrs
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


def _compute_levenshtein_stats(truth: str, hypothesis: str) -> Dict[str, float]:
  """Wrapper around jiwer library to compute Levenshtein statistics."""
  try:
    stats = jiwer.compute_measures(truth=[truth], hypothesis=[hypothesis])  # pytype: disable=module-attr
    return {
        'substitutions': stats['substitutions'],
        'deletions': stats['deletions'],
        'insertions': stats['insertions'],
        'hits': stats['hits'],
    }
  except AttributeError:
    stats = jiwer.process_words(reference=[truth], hypothesis=[hypothesis])  # pytype: disable=module-attr
    return {
        'substitutions': stats.substitutions,
        'deletions': stats.deletions,
        'insertions': stats.insertions,
        'hits': stats.hits,
    }


def compute_word_errors(
    truth: str, hypothesis: str, *, is_english: bool
) -> tuple[float, float]:
  """Computes the word error rate (WER)."""
  text_transform = (
      english.EnglishTextNormalizer()
      if is_english
      else basic.BasicTextNormalizer()
  )

  stats = _compute_levenshtein_stats(
      truth=text_transform(truth), hypothesis=text_transform(hypothesis)
  )
  return (
      stats['substitutions'] + stats['deletions'] + stats['insertions'],
      stats['hits'] + stats['substitutions'] + stats['deletions'],
  )


def compute_correct(truth: str, hypothesis: str, *, is_english: bool) -> float:
  """Computes the query error rate (QER)."""
  text_transform = (
      english.EnglishTextNormalizer()
      if is_english
      else basic.BasicTextNormalizer()
  )

  correct = float(text_transform(truth) != text_transform(hypothesis))
  return correct


@dataclasses.dataclass
class RerankingCandidates:
  sound_id: str
  texts: Sequence[str]


RerankingPredictionsCache = Mapping[str, tuple[Sequence[str], Sequence[float]]]


class RerankingEvaluator:
  """Evaluator for reranking tasks."""

  def __init__(
      self,
      candidate_embeddings_by_text: types.TextEmbeddingCache,
      mrr_at_k: int = 10,
  ):
    """Initializes the reranking evaluator.

    Args:
      candidate_embeddings_by_text: A dictionary mapping candidate texts to
        their embeddings.
      mrr_at_k: Computes MRR @ `mrr_at_k`..
    """
    self.candidate_embeddings_by_text = candidate_embeddings_by_text
    self.mrr_at_k = mrr_at_k

  def __call__(
      self,
      embeddings: types.SoundEmbeddingCache,
      candidates_batch: Sequence[RerankingCandidates],
  ) -> list[types.Score]:
    """Evaluates reranking quality for a single example.

    Args:
      embeddings: The embeddings to evaluate.
      candidates_batch: The candidate texts sorted by relevance to evaluate.

    Returns:
      A list of Score objects containing the final, aggregated scores. The
      scores include mean average precision (MAP), mean reciprocal rank (MRR),
      word error rate (WER) and candidate error rate (CER).
    """
    predictions = {}
    is_english = []
    for candidates in candidates_batch:
      embedding = embeddings[candidates.sound_id].embedding
      context = embeddings[candidates.sound_id].context
      if embedding.ndim != 2 or embedding.shape[0] != 1:
        raise ValueError(
            'Embedding must be a 2D array of shape (1, embedding_dim),'
            f' but got a {embedding.shape} array.'
        )
      searcher = tfrs.layers.factorized_top_k.BruteForce(
          k=len(candidates.texts)
      )
      candidate_embeddings = [
          self.candidate_embeddings_by_text[text].embeddings[0]
          for text in candidates.texts
      ]
      searcher.index(
          candidates=tf.constant(candidate_embeddings, dtype=tf.float32)
      )
      ranked_candidate_scores, ranked_candidate_ids = searcher(
          tf.constant(embedding, dtype=tf.float32)
      )
      ranked_candidate_texts = [  # pylint: disable=g-complex-comprehension
          [candidates.texts[int(x.numpy())] for x in ids]
          for ids in ranked_candidate_ids
      ]
      predictions[candidates.sound_id] = (
          ranked_candidate_texts[0],
          ranked_candidate_scores[0],
      )
      if context.language is None:
        raise ValueError('Language is required for reranking evaluation.')
      is_english.append(context.language.lower() == 'en')
    if sum(is_english) == len(is_english):
      is_english = True
    elif sum(is_english) == 0:
      is_english = False
    else:
      raise ValueError(
          'Language must be consistent for all examples in a batch.'
      )
    return self.evaluate_predictions(predictions, candidates_batch, is_english)

  def evaluate_predictions(
      self,
      predictions: RerankingPredictionsCache,
      candidates_batch: Sequence[RerankingCandidates],
      is_english: bool,  # For text normalization.
  ) -> list[types.Score]:
    """Returns quality metrics of the predictions."""
    values_by_metric: dict[str, list[types.WeightedValue]] = {
        'map': [],
        'mrr': [],
        'wer': [],
        'cer': [],
    }
    for candidates in candidates_batch:
      ranked_candidate_texts, ranked_candidate_scores = predictions[
          candidates.sound_id
      ]

      word_errors, word_errors_weight = compute_word_errors(
          truth=candidates.texts[0],
          hypothesis=ranked_candidate_texts[0],
          is_english=is_english,
      )
      values_by_metric['wer'].append(
          types.WeightedValue(
              value=word_errors / word_errors_weight,
              weight=word_errors_weight,
          )
      )
      values_by_metric['cer'].append(
          types.WeightedValue(
              value=compute_correct(
                  truth=candidates.texts[0],
                  hypothesis=ranked_candidate_texts[0],
                  is_english=is_english,
              )
          )
      )
      values_by_metric['mrr'].append(
          types.WeightedValue(
              value=metrics.compute_reciprocal_rank(
                  candidates.texts[0], ranked_candidate_texts[:self.mrr_at_k]
              ),
          )
      )
      values_by_metric['map'].append(
          types.WeightedValue(
              value=sklearn_metrics.average_precision_score(
                  y_true=[True] + [False] * (len(ranked_candidate_scores) - 1),
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
