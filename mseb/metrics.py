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

"""Common metrics for MSEB."""

from collections.abc import Sequence
from typing import Callable, Mapping

import jiwer
import numba
import numpy as np


@numba.njit
def _jit_edit_distance_core(
    cost_matrix: np.ndarray,
    w_ins: float,
    w_del: float
) -> float:
  """Compiled Dynamic Programming core for Continuous Edit Distance.

  Args:
    cost_matrix: (N, M) array of L2 distances between frames.
    w_ins: Scalar penalty for insertions.
    w_del: Scalar penalty for deletions.

  Returns:
    The minimum cumulative edit cost to align the two sequences.
  """
  n, m = cost_matrix.shape
  dp = np.zeros((n + 1, m + 1))

  # Initialize boundary conditions (cumulative insertions/deletions)
  for i in range(1, n + 1):
    dp[i, 0] = dp[i - 1, 0] + w_del
  for j in range(1, m + 1):
    dp[0, j] = dp[0, j - 1] + w_ins

  # Fill DP table
  for i in range(1, n + 1):
    for j in range(1, m + 1):
      # Substitution cost is pulled from the pre-calculated cost_matrix
      dp[i, j] = min(
          dp[i - 1, j] + w_del,        # Deletion
          dp[i, j - 1] + w_ins,        # Insertion
          dp[i - 1, j - 1] + cost_matrix[i - 1, j - 1]  # Substitution
      )
  return dp[n, m]


@numba.njit
def _jit_dtw_core(dist_matrix: np.ndarray) -> float:
  """Compiled Dynamic Programming core for Dynamic Time Warping.

  Args:
    dist_matrix: (N, M) array of Euclidean distances between frames.

  Returns:
    The optimal non-linear alignment cost.
  """
  n, m = dist_matrix.shape
  dtw_matrix = np.full((n + 1, m + 1), np.inf)
  dtw_matrix[0, 0] = 0.0

  for i in range(1, n + 1):
    for j in range(1, m + 1):
      dtw_matrix[i, j] = dist_matrix[i - 1, j - 1] + min(
          dtw_matrix[i - 1, j],     # Vertical (Insertion)
          dtw_matrix[i, j - 1],     # Horizontal (Deletion)
          dtw_matrix[i - 1, j - 1]  # Diagonal (Match)
      )
  return dtw_matrix[n, m]


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


def compute_unit_edit_distance(
    truth_tokens: Sequence[int | str],
    hypothesis_tokens: Sequence[int | str]
) -> Mapping[str, float]:
  """Computes Unit Edit Distance (UED) statistics for discrete tokens.

  This metric treats each token ID as a discrete symbol and calculates the
  minimum number of insertions, deletions, and substitutions required to
  transform the truth sequence into the hypothesis sequence using jiwer.

  Args:
    truth_tokens: Sequence of discrete token IDs (integers or strings)
        representing the reference.
    hypothesis_tokens: Sequence of discrete token IDs representing the
        perturbed or predicted output.

  Returns:
    A mapping containing:
        'normalized_distance': Total edits divided by max(len_truth, len_hypo).
        'edit_distance': The raw total number of edits (ins + del + sub).
        'substitutions': Number of substitution operations.
        'deletions': Number of deletion operations.
        'insertions': Number of insertion operations.
        'reference_length': The number of tokens in the truth sequence.
  """
  truth = ' '.join(map(str, truth_tokens))
  hyp = ' '.join(map(str, hypothesis_tokens))

  stats = _compute_levenshtein_stats(truth=truth, hypothesis=hyp)
  raw_edits = stats['substitutions'] + stats['deletions'] + stats['insertions']

  ref_len = len(truth_tokens)
  norm_factor = max(ref_len, 1)

  return {
      'normalized_distance': raw_edits / norm_factor,
      'raw_distance': float(raw_edits),
      'substitutions': float(stats['substitutions']),
      'deletions': float(stats['deletions']),
      'insertions': float(stats['insertions']),
      'reference_length': float(ref_len),
  }


def compute_lp_norm(
    z1: np.ndarray,
    z2: np.ndarray,
    p: int = 2
) -> Mapping[str, float]:
  """Computes the standard L_p norm between two latent sequences.

  Measures rigid point-to-point distance. Requires identical temporal
  lengths and embedding dimensions.

  Args:
    z1: Array of shape (time_steps, embedding_dim).
    z2: Array of shape (time_steps, embedding_dim).
    p: The order of the norm (default 2).

  Returns:
    A mapping containing 'raw_distance' and 'reference_length'.
  """
  if z1.shape != z2.shape:
    raise ValueError(f'Shape mismatch for L_p: {z1.shape} vs {z2.shape}.')
  dist = float(np.linalg.norm(z1 - z2, ord=p))
  return {
      'raw_distance': dist,
      'reference_length': float(len(z1))
  }


def compute_dynamic_time_warping_distance(
    z1: np.ndarray,
    z2: np.ndarray
) -> Mapping[str, float]:
  """Computes Dynamic Time Warping (DTW) distance with Euclidean cost.

  Finds optimal non-linear alignment by minimizing cumulative Euclidean
  distance.

  Args:
    z1: Array of shape (N, d).
    z2: Array of shape (M, d).

  Returns:
    A mapping containing 'raw_distance' and 'reference_length'.
  """
  # Vectorized Euclidean distance matrix: sqrt(a^2 + b^2 - 2ab)
  dot_product = np.dot(z1, z2.T)
  sq_norms1 = np.sum(z1**2, axis=1).reshape(-1, 1)
  sq_norms2 = np.sum(z2**2, axis=1).reshape(1, -1)
  dist_matrix = np.sqrt(np.maximum(sq_norms1 + sq_norms2 - 2 * dot_product, 0))
  raw_dist = float(_jit_dtw_core(dist_matrix))
  return {
      'raw_distance': raw_dist,
      'reference_length': float(len(z1))
  }


def compute_continuous_edit_distance(
    z1: np.ndarray,
    z2: np.ndarray,
    w_ins: float = 2.0,
    w_del: float = 2.0,
    unit_sphere_scaling: bool = True
) -> Mapping[str, float]:
  """Computes Continuous Edit Distance (CED) with global max-norm scaling.

  Extends Levenshtein logic to continuous vectors. Global max-norm scaling
  ensures all vectors fall within a unit sphere, meaning the maximum
  possible substitution cost is 2.0 (the diameter).

  The metric is robust to temporal shifts and sequence length mismatches,
  making it ideal for stability profiling of frame-based representations.

  Args:
    z1: Array of shape (N, d) representing the reference sequence.
    z2: Array of shape (M, d) representing the hypothesis sequence.
    w_ins: Penalty for an insertion. Default 2.0 (matches unit sphere diameter).
    w_del: Penalty for a deletion. Default 2.0.
    unit_sphere_scaling: If True, scales each sequence by its maximum
        internal norm so that all vectors reside within a unit sphere.

  Returns:
    A mapping containing:
        'raw_distance': The cumulative cost of the minimum-cost edit path.
        'normalized_distance': raw_distance divided by (2.0 * reference_length).
        'reference_length': The number of frames (N) in the truth sequence.
  """
  if unit_sphere_scaling:
    # Scale sequences so all vectors reside within the unit sphere [0, 1]
    norm1 = np.linalg.norm(z1, axis=1)
    norm2 = np.linalg.norm(z2, axis=1)
    max_norm1 = np.max(norm1) if z1.any() else 1.0
    max_norm2 = np.max(norm2) if z2.any() else 1.0
    z1 = z1 / (max_norm1 + 1e-9)
    z2 = z2 / (max_norm2 + 1e-9)

  # Pre-calculate cost matrix (Euclidean)
  dot_product = np.dot(z1, z2.T)
  sq_norms1 = np.sum(z1**2, axis=1).reshape(-1, 1)
  sq_norms2 = np.sum(z2**2, axis=1).reshape(1, -1)
  cost_matrix = np.sqrt(np.maximum(sq_norms1 + sq_norms2 - 2 * dot_product, 0))
  raw_cost = _jit_edit_distance_core(cost_matrix, w_ins, w_del)
  n = len(z1)
  return {
      'raw_distance': float(raw_cost),
      'normalized_distance': float(raw_cost / (2.0 * max(n, 1))),
      'reference_length': float(n),
  }
