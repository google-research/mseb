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
from typing import Dict, List, Mapping, Sequence, Union

import jiwer
from mseb import encoder
from mseb import evaluator
from mseb import types
from mseb.evaluators import retrieval_evaluator
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from whisper.normalizers import basic
from whisper.normalizers import english


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


compute_reciprocal_rank = retrieval_evaluator.compute_reciprocal_rank


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


class RerankingEvaluator(evaluator.Evaluator):
  """Evaluator for reranking tasks."""

  def __call__(
      self,
      sequence: Union[str, Sequence[float]],
      context: encoder.ContextParams,
      candidate_texts: Sequence[str] = tuple(),
      candidate_embeddings: np.ndarray = np.empty((0, 0)),
      document_top_k: int = 10,
  ) -> dict[str, float]:
    """Evaluates quality of the encoder for input sequence and return metrics.

    Args:
      sequence: Input sound sequence to encode. String-type sequences are
        interpreted as sound file paths.
      context: Encoder input context parameters.
      candidate_texts: Candidate texts to rank.
      candidate_embeddings: Candidate embeddings to rank.
      document_top_k: Number of documents to retrieve.

    Returns:
      Dictionary of metrics, including reciprocal rank, query error, and word
      error.
    """
    _, query_embeddings = self.sound_encoder.encode(
        sequence=sequence, context=context, **self.encode_kwargs
    )
    searcher = tfrs.layers.factorized_top_k.BruteForce(k=document_top_k)
    searcher.index(
        candidates=tf.constant(candidate_embeddings, dtype=tf.float32)
    )
    _, ranked_candidate_ids = searcher(
        tf.constant(query_embeddings, dtype=tf.float32)
    )
    ranked_candidate_texts = [  # pylint: disable=g-complex-comprehension
        [candidate_texts[int(x.numpy())] for x in ids]
        for ids in ranked_candidate_ids
    ]

    assert context.language is not None
    is_english = context.language.lower() == 'en'

    word_errors, word_errors_weight = compute_word_errors(
        truth=candidate_texts[0],
        hypothesis=ranked_candidate_texts[0][0],
        is_english=is_english,
    )

    return {
        'reciprocal_rank': compute_reciprocal_rank(
            candidate_texts[0], ranked_candidate_texts[0]
        ),
        'word_errors': word_errors,
        'word_errors_weight': word_errors_weight,
        'correct': compute_correct(
            truth=candidate_texts[0],
            hypothesis=ranked_candidate_texts[0][0],
            is_english=is_english,
        ),
    }

  def combine_scores(self, scores: List[Dict[str, float]]) -> Dict[str, float]:
    return evaluator.compute_weighted_average_and_std(
        scores,
        (
            ('reciprocal_rank', 'mrr'),
            ('word_errors', 'wer'),
            ('correct', 'qer'),
        ),
    )


@dataclasses.dataclass
class RerankingCandidates:
  sound_id: str
  texts: Sequence[str]


RerankingPredictionsCache = Mapping[str, Sequence[str]]


class RerankingEvaluatorV2:
  """Evaluator for reranking tasks."""

  def __init__(
      self,
      candidate_embeddings_by_text: types.TextEmbeddingCache,
      candidate_top_k: int = 10,
  ):
    self.candidate_embeddings_by_text = candidate_embeddings_by_text
    self.candidate_top_k = candidate_top_k

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
      scores include mean reciprocal rank (MRR), word error rate (WER) and
      candidate error rate (CER).
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
          k=min(self.candidate_top_k, len(candidates.texts))
      )
      candidate_embeddings = [
          self.candidate_embeddings_by_text[text].embeddings[0]
          for text in candidates.texts
      ]
      searcher.index(
          candidates=tf.constant(candidate_embeddings, dtype=tf.float32)
      )
      _, ranked_candidate_ids = searcher(
          tf.constant(embedding, dtype=tf.float32)
      )
      ranked_candidate_texts = [  # pylint: disable=g-complex-comprehension
          [candidates.texts[int(x.numpy())] for x in ids]
          for ids in ranked_candidate_ids
      ]
      predictions[candidates.sound_id] = ranked_candidate_texts[0]
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
        'mrr': [],
        'wer': [],
        'cer': [],
    }
    for candidates in candidates_batch:
      ranked_candidate_texts = predictions[candidates.sound_id][
          : self.candidate_top_k
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
              value=compute_reciprocal_rank(
                  candidates.texts[0], ranked_candidate_texts
              ),
          )
      )

    mrr_score = mrr(
        *evaluator.compute_weighted_average_and_std_v2(values_by_metric['mrr'])
    )
    wer_score = wer(
        *evaluator.compute_weighted_average_and_std_v2(values_by_metric['wer'])
    )
    cer_score = cer(
        *evaluator.compute_weighted_average_and_std_v2(values_by_metric['cer'])
    )
    return [wer_score, cer_score, mrr_score]
