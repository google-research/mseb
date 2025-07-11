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

"""Evaluator for retrieval tasks."""

from __future__ import annotations

from typing import Dict, List, Sequence, Union

import jiwer
from mseb import encoder
from mseb import evaluator
from mseb.evaluators import retrieval_evaluator
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from whisper.normalizers import basic
from whisper.normalizers import english


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
  """Evaluator for retrieval tasks."""

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
