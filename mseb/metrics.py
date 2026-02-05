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

"""Common metrics for MSEB."""

from collections.abc import Sequence
from typing import Callable, Mapping
import jiwer
# from whisper.normalizers import basic
# from whisper.normalizers import english


def compute_reciprocal_rank(
    reference: str, predicted_neighbors: Sequence[str]
) -> float:
  """Computes the reciprocal rank for mean reciprocal rank (MRR)."""
  rank = 0
  for i, neighbor in enumerate(predicted_neighbors):
    if reference == neighbor:
      # Matched the reference at this position in the ranking.
      rank = i + 1
      break
  reciprocal_rank = 1 / rank if rank > 0 else 0
  return reciprocal_rank


def compute_exact_match(
    reference: str, predicted_neighbors: Sequence[str]
) -> float:
  """Computes the exact match for the first predicted neighbor."""
  if predicted_neighbors:
    return float(reference == predicted_neighbors[0])
  return 0.0


def _compute_levenshtein_stats(
    truth: str, hypothesis: str
) -> Mapping[str, float]:
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
    truth: str,
    hypothesis: str,
    *,
    text_transform: Callable[[str], str] | None = None
) -> tuple[float, float]:
  """Computes the word errors."""

  if text_transform:
    truth = text_transform(truth)
    hypothesis = text_transform(hypothesis)

  stats = _compute_levenshtein_stats(truth=truth, hypothesis=hypothesis)
  return (
      stats['substitutions'] + stats['deletions'] + stats['insertions'],
      stats['hits'] + stats['substitutions'] + stats['deletions'],
  )
