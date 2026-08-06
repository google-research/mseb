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

"""Brain encoding evaluator.

Evaluates speech embeddings by training ridge regression models to predict
fMRI brain responses, following the methodology from:

  Antonello, Turek, Vo, and Huth (2024). "Scaling in Speech and Language Models:
  A Path to Human-Level Performance?" arXiv:2401.10150.

The primary metric is the mean voxel-wise Pearson correlation between predicted
and actual fMRI BOLD responses on held-out test data.

Uses JAX/XLA for multi-core CPU acceleration of the ridge regression.
"""

import dataclasses
import os
from typing import Sequence

from absl import logging
import jax
import jax.numpy as jnp
from mseb import types
import numpy as np

# Default to CPU; set JAX_PLATFORMS=gpu or tpu to use accelerators.
os.environ.setdefault('JAX_PLATFORMS', 'cpu')


def mean_correlation_score(value: float = 0.0, std: float | None = None):
  return types.Score(
      metric='MeanCorrelation',
      description=(
          'Mean voxel-wise Pearson correlation between predicted and'
          ' actual fMRI BOLD responses on held-out test data.'
      ),
      value=value,
      min=-1,
      max=1,
      std=std,
  )


def median_correlation_score(value: float = 0.0):
  return types.Score(
      metric='MedianCorrelation',
      description=(
          'Median voxel-wise Pearson correlation between predicted and'
          ' actual fMRI BOLD responses on held-out test data.'
      ),
      value=value,
      min=-1,
      max=1,
  )


def fraction_significant_score(value: float = 0.0):
  return types.Score(
      metric='FractionSignificant',
      description=(
          'Fraction of voxels with statistically significant'
          ' prediction correlation (p < 0.05, uncorrected).'
      ),
      value=value,
      min=0,
      max=1,
  )


@dataclasses.dataclass
class BrainEncodingExample:
  """A single example for brain encoding evaluation.

  Attributes:
    sound_id: The identifier matching a key in the embeddings cache.
    tr_index_start: Starting TR index for this stimulus in the fMRI data.
    tr_index_end: Ending TR index (exclusive) for this stimulus.
  """

  sound_id: str
  tr_index_start: int
  tr_index_end: int


def apply_fir_delays(
    features: np.ndarray,
    delays: Sequence[int],
) -> np.ndarray:
  """Applies FIR (Finite Impulse Response) delays to feature matrix.

  Creates a delayed feature matrix by stacking time-shifted copies of the
  original features. This captures the hemodynamic response function (HRF)
  which causes fMRI responses to lag behind neural activity.

  Args:
    features: Feature matrix of shape (T, D).
    delays: Sequence of delay values in TRs (e.g., [1, 2, 3, 4]).

  Returns:
    Delayed feature matrix of shape (T, D * len(delays)).
  """
  n_time, n_features = features.shape
  delayed = np.zeros((n_time, n_features * len(delays)), dtype=features.dtype)
  for di, delay in enumerate(delays):
    start = di * n_features
    end = (di + 1) * n_features
    if delay > 0:
      delayed[delay:, start:end] = features[:-delay]
    elif delay < 0:
      delayed[:delay, start:end] = features[-delay:]
    else:
      delayed[:, start:end] = features
  return delayed


def _jax_columnwise_correlation(
    predictions: jax.Array,
    actuals: jax.Array,
) -> jax.Array:
  """Computes Pearson correlation for each column (voxel) using JAX.

  Args:
    predictions: shape (T, V).
    actuals: shape (T, V).

  Returns:
    Correlations of shape (V,).
  """
  pred_zm = predictions - jnp.mean(predictions, axis=0, keepdims=True)
  actual_zm = actuals - jnp.mean(actuals, axis=0, keepdims=True)

  pred_norm = jnp.sqrt(jnp.sum(pred_zm**2, axis=0))
  actual_norm = jnp.sqrt(jnp.sum(actual_zm**2, axis=0))

  denom = pred_norm * actual_norm
  denom = jnp.where(denom == 0, 1.0, denom)

  return jnp.sum(pred_zm * actual_zm, axis=0) / denom


@jax.jit
def _cv_one_alpha(
    s1: jax.Array,
    vt1_t: jax.Array,
    u1t_resp: jax.Array,
    stim_half2: jax.Array,
    resp_half1: jax.Array,
    s2: jax.Array,
    vt2_t: jax.Array,
    u2t_resp: jax.Array,
    stim_half1: jax.Array,
    resp_half2: jax.Array,
    alpha: jax.Array,
) -> jax.Array:
  """JIT-compiled cross-validation for a single alpha.

  Computes predictions for both CV splits and returns the average
  per-voxel correlation.

  Args:
    s1: Singular values of the first half of the training stimuli.
    vt1_t: Transpose of the right singular vectors of the first half.
    u1t_resp: Projection of the first half of the training responses onto the
      left singular vectors.
    stim_half2: Second half of the training stimuli.
    resp_half1: First half of the training responses.
    s2: Singular values of the second half of the training stimuli.
    vt2_t: Transpose of the right singular vectors of the second half.
    u2t_resp: Projection of the second half of the training responses onto the
      left singular vectors.
    stim_half1: First half of the training stimuli.
    resp_half2: Second half of the training responses.
    alpha: Regularization value.
  """
  # Predict second half from first half.
  d1 = s1 / (s1**2 + alpha)
  weights1 = vt1_t @ (d1[:, None] * u1t_resp)
  pred2 = stim_half2 @ weights1

  # Predict first half from second half.
  d2 = s2 / (s2**2 + alpha)
  weights2 = vt2_t @ (d2[:, None] * u2t_resp)
  pred1 = stim_half1 @ weights2

  # Average correlation from both splits.
  corr1 = _jax_columnwise_correlation(pred1, resp_half1)
  corr2 = _jax_columnwise_correlation(pred2, resp_half2)
  return (corr1 + corr2) / 2.0


@jax.jit
def _predict_for_alpha(
    s: jax.Array,
    vt_t: jax.Array,
    ut_resp_masked: jax.Array,
    stim_test: jax.Array,
    alpha: jax.Array,
) -> jax.Array:
  """JIT-compiled final prediction for a single alpha."""
  d = s / (s**2 + alpha)
  weights = vt_t @ (d[:, None] * ut_resp_masked)
  return stim_test @ weights


def ridge_svd(
    stim_train: np.ndarray,
    resp_train: np.ndarray,
    stim_test: np.ndarray,
    alphas: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
  """SVD-based ridge regression with per-voxel alpha selection.

  Uses JAX/XLA for accelerated matrix operations with multi-core CPU
  parallelism. SVD is computed via JAX, and the cross-validation loop
  over alphas uses JIT-compiled functions.

  This follows the approach in HuthLab's ridge_utils/ridge.py.

  Args:
    stim_train: Training features, shape (T_train, D).
    resp_train: Training responses, shape (T_train, V).
    stim_test: Test features, shape (T_test, D).
    alphas: Candidate regularization values.

  Returns:
    Tuple of (predictions, best_alphas):
      - predictions: shape (T_test, V), predicted responses for test set.
      - best_alphas: shape (V,), best alpha per voxel.
  """
  n_train = stim_train.shape[0]
  n_voxels = resp_train.shape[1]

  # Move data to JAX arrays (float32).
  stim_train_j = jnp.array(stim_train, dtype=jnp.float32)
  resp_train_j = jnp.array(resp_train, dtype=jnp.float32)
  stim_test_j = jnp.array(stim_test, dtype=jnp.float32)

  # SVD of full training stimuli.
  logging.info('Running SVD of training stimuli via JAX')
  u, s, vt = jnp.linalg.svd(stim_train_j, full_matrices=False)
  logging.info('SVD done. s shape: %s', s.shape)

  # Project responses into SVD space (precomputed, shared across alphas).
  ut_resp = u.T @ resp_train_j  # (D, V)

  # Cross-validation: split training data in half.
  half = n_train // 2
  logging.info('Running SVD of CV splits via JAX')
  u1, s1, vt1 = jnp.linalg.svd(stim_train_j[:half], full_matrices=False)
  u2, s2, vt2 = jnp.linalg.svd(stim_train_j[half:], full_matrices=False)
  logging.info('CV SVDs done')

  # Precompute projections outside the alpha loop.
  u1t_resp = u1.T @ resp_train_j[:half]  # (D, V)
  u2t_resp = u2.T @ resp_train_j[half:]  # (D, V)
  vt1_t = vt1.T
  vt2_t = vt2.T
  stim_half1 = stim_train_j[:half]
  stim_half2 = stim_train_j[half:]
  resp_half1 = resp_train_j[:half]
  resp_half2 = resp_train_j[half:]

  best_corr = jnp.full(n_voxels, -jnp.inf, dtype=jnp.float32)
  best_alpha_idx = np.zeros(n_voxels, dtype=np.int32)

  # Cross-validation loop over alphas.
  for ai, alpha in enumerate(alphas):
    logging.info('CV alpha %d/%d = %.2e', ai + 1, len(alphas), alpha)
    alpha_j = jnp.float32(alpha)

    avg_corr = _cv_one_alpha(
        s1,
        vt1_t,
        u1t_resp,
        stim_half2,
        resp_half1,
        s2,
        vt2_t,
        u2t_resp,
        stim_half1,
        resp_half2,
        alpha_j,
    )

    improved = avg_corr > best_corr
    best_corr = jnp.where(improved, avg_corr, best_corr)
    best_alpha_idx = np.where(np.asarray(improved), ai, best_alpha_idx)

  logging.info('CV done. Computing final predictions.')

  # Final predictions using best alpha per voxel.
  vt_t = vt.T
  predictions = np.zeros((stim_test.shape[0], n_voxels), dtype=np.float32)
  for ai, alpha in enumerate(alphas):
    voxel_mask = best_alpha_idx == ai
    if not np.any(voxel_mask):
      continue
    alpha_j = jnp.float32(alpha)
    preds = _predict_for_alpha(
        s, vt_t, ut_resp[:, voxel_mask], stim_test_j, alpha_j
    )
    predictions[:, voxel_mask] = np.asarray(preds)

  logging.info('Final predictions done.')
  best_alphas = np.array([alphas[i] for i in best_alpha_idx])
  return predictions, best_alphas


def _columnwise_correlation(
    predictions: np.ndarray,
    actuals: np.ndarray,
) -> np.ndarray:
  """Computes Pearson correlation for each column (voxel).

  Args:
    predictions: shape (T, V).
    actuals: shape (T, V).

  Returns:
    Correlations of shape (V,).
  """
  # Zero-mean.
  pred_zm = predictions - predictions.mean(axis=0, keepdims=True)
  actual_zm = actuals - actuals.mean(axis=0, keepdims=True)

  # Correlation = normalized dot product.
  pred_norm = np.sqrt(np.sum(pred_zm**2, axis=0))
  actual_norm = np.sqrt(np.sum(actual_zm**2, axis=0))

  denom = pred_norm * actual_norm
  # Avoid division by zero for constant columns.
  denom = np.where(denom == 0, 1.0, denom)

  return np.sum(pred_zm * actual_zm, axis=0) / denom


class BrainEncodingEvaluator:
  """Evaluator for brain encoding models using ridge regression.

  This evaluator takes precomputed speech embeddings, aligns them to fMRI
  temporal resolution (TRs), applies FIR delays to model the hemodynamic
  response, and uses ridge regression to predict fMRI BOLD responses.

  The primary metric is the mean voxel-wise Pearson correlation on held-out
  test data.
  """

  def __init__(
      self,
      tr_duration: float = 2.0,
      delays: Sequence[int] = (1, 2, 3, 4),
      alphas: Sequence[float] | None = None,
  ):
    """Initializes the BrainEncodingEvaluator.

    Args:
      tr_duration: Duration of one fMRI TR in seconds.
      delays: FIR delay values in TRs for the hemodynamic response.
      alphas: Ridge regularization candidates. If None, uses logarithmically
        spaced values from 1 to 1e6.
    """
    self.tr_duration = tr_duration
    self.delays = delays
    self.alphas = alphas or np.logspace(0, 6, 20).tolist()

  def __call__(
      self,
      embeddings: types.MultiModalEmbeddingCache,
      train_examples: Sequence[BrainEncodingExample],
      test_examples: Sequence[BrainEncodingExample],
      fmri_train: np.ndarray,
      fmri_test: np.ndarray,
  ) -> list[types.Score]:
    """Runs the brain encoding evaluation.

    Args:
      embeddings: Precomputed embeddings keyed by sound ID.
      train_examples: Training examples with sound IDs and TR indices.
      test_examples: Test examples with sound IDs and TR indices.
      fmri_train: Training fMRI responses, shape (T_train, V).
      fmri_test: Test fMRI responses, shape (T_test, V).

    Returns:
      List of Score objects with correlation metrics.
    """
    # Build TR-aligned feature matrices.
    train_features = self._build_feature_matrix(embeddings, train_examples)
    test_features = self._build_feature_matrix(embeddings, test_examples)

    # Apply FIR delays.
    train_delayed = apply_fir_delays(train_features, self.delays)
    test_delayed = apply_fir_delays(test_features, self.delays)

    # Z-score normalization.
    train_mean = train_delayed.mean(axis=0, keepdims=True)
    train_std = train_delayed.std(axis=0, keepdims=True)
    train_std = np.where(train_std == 0, 1.0, train_std)
    train_delayed = (train_delayed - train_mean) / train_std
    test_delayed = (test_delayed - train_mean) / train_std

    # Run ridge regression.
    logging.info(
        'Running ridge regression: fmri_train shape %s, train_delayed shape %s,'
        ' test_delayed shape %s',
        fmri_train.shape,
        train_delayed.shape,
        test_delayed.shape,
    )
    predictions, _ = ridge_svd(
        train_delayed, fmri_train, test_delayed, self.alphas
    )
    logging.info('Ridge regression done')

    # Compute per-voxel correlations.
    logging.info(
        'Computing per-voxel correlations: predictions shape %s, fmri_test'
        ' shape %s',
        predictions.shape,
        fmri_test.shape,
    )
    correlations = _columnwise_correlation(predictions, fmri_test)
    logging.info('Per-voxel correlations done')

    mean_corr = float(np.mean(correlations))
    std_corr = float(np.std(correlations))
    median_corr = float(np.median(correlations))
    frac_significant = float(np.mean(correlations > 0.0))

    return [
        mean_correlation_score(value=mean_corr, std=std_corr),
        median_correlation_score(value=median_corr),
        fraction_significant_score(value=frac_significant),
    ]

  def _downsample_to_tr(
      self, embedding: np.ndarray, timestamps: np.ndarray
  ) -> np.ndarray:
    """Downsamples frame-level embeddings to TR resolution by block averaging.

    If the embedding is already at TR resolution (frame duration ≈ tr_duration),
    it is returned unchanged.

    Args:
      embedding: Feature matrix of shape (T_frames, D).
      timestamps: Timestamps of shape (T_frames, 2) with [start, end] per frame.

    Returns:
      Downsampled feature matrix of shape (T_trs, D).
    """
    if len(timestamps) < 2:
      return embedding
    frame_duration = float(timestamps[1, 0] - timestamps[0, 0])
    # Already TR-aligned (within 10% tolerance).
    if frame_duration >= self.tr_duration * 0.9:
      return embedding

    frames_per_tr = int(round(self.tr_duration / frame_duration))
    n_trs = embedding.shape[0] // frames_per_tr
    logging.info(
        'Downsampling embeddings from %.1f Hz to %.1f Hz '
        '(%d frames/TR, %d TRs)',
        1.0 / frame_duration,
        1.0 / self.tr_duration,
        frames_per_tr,
        n_trs,
    )
    trimmed = embedding[: n_trs * frames_per_tr]
    return (
        trimmed.reshape(n_trs, frames_per_tr, -1)
        .mean(axis=1)
        .astype(embedding.dtype)
    )

  def _build_feature_matrix(
      self,
      embeddings: types.MultiModalEmbeddingCache,
      examples: Sequence[BrainEncodingExample],
  ) -> np.ndarray:
    """Builds a TR-aligned feature matrix from embeddings and examples.

    If the input embeddings are at a higher temporal resolution than the TR
    (e.g. 50 Hz Whisper frames vs 0.5 Hz fMRI), they are automatically
    downsampled to TR resolution by block averaging before TR-indexing.

    Args:
      embeddings: Precomputed embeddings cache.
      examples: Sequence of examples mapping sound IDs to TR ranges.

    Returns:
      Feature matrix of shape (n_trs, D).
    """

    zs = lambda v: (v - v.mean(0)) / v.std(0)  # z-score function

    features = []
    for example in examples:
      emb = embeddings[example.sound_id]
      assert hasattr(emb, 'embedding') and hasattr(
          emb, 'timestamps'
      ), f'Expected SoundEmbedding, got {type(emb)}'
      embedding = self._downsample_to_tr(emb.embedding, emb.timestamps)  # pyrefly: ignore[bad-argument-type]
      feature = zs(embedding[example.tr_index_start : example.tr_index_end])
      features.append(feature)
    features = np.nan_to_num(np.vstack(features))
    return features
