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

"""Evaluator for reconstruction task."""

from __future__ import annotations

import logging
from typing import Literal

from mseb import types
import numpy as np
import scipy.linalg
from scipy.spatial import distance as scipy_distance


logger = logging.getLogger(__name__)

_KAD_SCALE_FACTOR = 100.0


_METRIC_DESCRIPTIONS: dict[str, str] = {
    'FAD': (
        'Fréchet Audio Distance. Measures the distance between the '
        'distributions of original and reconstructed embeddings. Lower is '
        'better.'
    ),
    'KAD': (
        'Kernel Audio Distance (using MMD). Measures the distance between '
        'the distributions of original and reconstructed embeddings. Lower is '
        'better.'
    ),
    'Embedding MSE': (
        'Per-example Mean Squared Error between original and reconstructed '
        'embedding frames, averaged across the dataset. Lower is better.'
    ),
}


def frechet_audio_distance(value: float = 0.0) -> types.Score:
  return types.Score(
      metric='FAD',
      description=_METRIC_DESCRIPTIONS['FAD'],
      value=value,
      min=0.0,
      max=float('inf'),
  )


def kernel_audio_distance(value: float = 0.0) -> types.Score:
  return types.Score(
      metric='KAD',
      description=_METRIC_DESCRIPTIONS['KAD'],
      value=value,
      min=0.0,
      max=float('inf'),
  )


def embedding_mse(value: float = 0.0) -> types.Score:
  return types.Score(
      metric='Embedding MSE',
      description=_METRIC_DESCRIPTIONS['Embedding MSE'],
      value=value,
      min=0.0,
      max=float('inf'),
  )


def _frechet_distance(
    x: np.ndarray, y: np.ndarray, eps: float = 1e-6
) -> float:
  """FAD implementation in NumPy/SciPy.

  Args:
    x: The first set of embeddings, shape (N, D).
    y: The second set of embeddings, shape (M, D).
    eps: A small value to prevent numerical instability.

  Returns:
    The FAD score.
  """
  # Ensure inputs have enough samples for stable covariance calculation
  if x.shape[0] < 2 or y.shape[0] < 2:
    logger.warning(
        'Need at least 2 samples for stable FAD calculation, got %d and %d',
        x.shape[0], y.shape[0]
    )
    return float('nan')  # Covariance is undefined for 1 sample

  mu_x = np.mean(x, axis=0)
  cov_x = np.cov(x, rowvar=False)
  mu_y = np.mean(y, axis=0)
  cov_y = np.cov(y, rowvar=False)

  mu_diff = mu_x - mu_y
  diffnorm_sq = mu_diff @ mu_diff

  # Add epsilon to diagonals for numerical stability initially
  cov_x_stable = cov_x + np.eye(cov_x.shape[0]) * eps
  cov_y_stable = cov_y + np.eye(cov_y.shape[0]) * eps

  # Calculate product and sqrtm
  try:
    covmean_sqrtm, _ = scipy.linalg.sqrtm(
        cov_x_stable.dot(cov_y_stable),
        disp=False
    )
  # pylint: disable=broad-except
  except Exception as e:
    logger.error(
        'FAD: scipy.linalg.sqrtm failed: %s', e
    )
    return float('nan')
  # pylint: enable=broad-except

  # Check for instability
  if not np.isfinite(covmean_sqrtm).all():
    logger.warning('FAD: sqrtm returned non-finite values. '
                   'Offset might be needed.')
    return float('nan')

  # Numerical error might give slight imaginary component.
  if np.iscomplexobj(covmean_sqrtm):
    if not np.allclose(np.diagonal(covmean_sqrtm).imag, 0, atol=1e-3):
      max_imag = np.max(np.abs(covmean_sqrtm.imag))
      logger.warning(
          'FAD calculation: sqrtm result has non-trivial imaginary '
          'component (max_imag=%.3g). Taking real part.', max_imag
      )
    covmean_sqrtm = covmean_sqrtm.real

  tr_covmean = np.trace(covmean_sqrtm)
  score = diffnorm_sq + np.trace(cov_x) + np.trace(cov_y) - 2 * tr_covmean

  # Clamp score at 0
  final_score = max(0.0, float(score))
  if final_score != score:
    logger.debug('FAD score clamped from %.4f to 0.0', score)

  return final_score


def _median_pairwise_distance(
    x: np.ndarray,
    subsample: int | None = None
) -> float:
  """NumPy port of median pairwise distance for KAD bandwidth."""
  x = np.asarray(x, dtype=np.float32)
  n_samples = x.shape[0]

  if subsample is not None and subsample < n_samples * (n_samples - 1) / 2:
    idx1 = np.random.randint(0, n_samples, (subsample,))
    idx2 = np.random.randint(0, n_samples, (subsample,))

    mask = idx1 == idx2
    idx2[mask] = (idx2[mask] + 1) % n_samples

    distances = np.sqrt(np.sum((x[idx1] - x[idx2]) ** 2, axis=1))
  else:
    distances = scipy_distance.pdist(x)

  return float(np.median(distances))


def _kernel_audio_distance(
    x: np.ndarray,
    y: np.ndarray,
    kernel: Literal['gaussian', 'iq', 'imq'] = 'gaussian',
    bandwidth: float | None = None,
    eps: float = 1e-8,
) -> float:
  """KAD (MMD) implementation in NumPy/SciPy.

  This implementation closely follows the description in:
  Chung, Yoonjin, et al. "KAD: No More FAD! An Effective and Efficient
  Evaluation Metric for Audio Generation." arXiv:2502.15602 (2025).
  (Ported from the PyTorch code provided alongside the paper/library).

  Args:
    x: The first set of embeddings, shape (N, D).
    y: The second set of embeddings, shape (M, D).
    kernel: The kernel function to use.
    bandwidth: The bandwidth for the kernel. If None, uses median heuristic.
    eps: A small value to prevent division by zero.

  Returns:
    The KAD score.
  """
  x = np.asarray(x, dtype=np.float32)
  y = np.asarray(y, dtype=np.float32)

  if bandwidth is None:
    bandwidth = _median_pairwise_distance(y)

  m, n = x.shape[0], y.shape[0]

  gamma = 1.0 / (2 * bandwidth**2 + eps)
  if kernel == 'gaussian':
    kernel_fn = lambda a: np.exp(-gamma * a)
  elif kernel == 'iq':
    kernel_fn = lambda a: 1.0 / (1.0 + gamma * a)
  elif kernel == 'imq':
    kernel_fn = lambda a: 1.0 / np.sqrt(1.0 + gamma * a)
  else:
    raise ValueError(f'Invalid kernel type: {kernel}')

  # K_xx term
  xx = x @ x.T
  x_sqnorms = np.diag(xx)
  d2_xx = x_sqnorms[:, np.newaxis] + x_sqnorms[np.newaxis, :] - 2 * xx
  k_xx = kernel_fn(d2_xx)
  k_xx = k_xx - np.diag(np.diag(k_xx))
  k_xx_mean = k_xx.sum() / (m * (m - 1))

  # K_yy term
  yy = y @ y.T
  y_sqnorms = np.diag(yy)
  d2_yy = y_sqnorms[:, np.newaxis] + y_sqnorms[np.newaxis, :] - 2 * yy
  k_yy = kernel_fn(d2_yy)
  k_yy = k_yy - np.diag(np.diag(k_yy))
  k_yy_mean = k_yy.sum() / (n * (n - 1))

  # K_xy term
  xy = x @ y.T
  d2_xy = x_sqnorms[:, np.newaxis] + y_sqnorms[np.newaxis, :] - 2 * xy
  k_xy = kernel_fn(d2_xy)
  k_xy_mean = k_xy.mean()

  score = (k_xx_mean + k_yy_mean - 2 * k_xy_mean) * _KAD_SCALE_FACTOR
  return max(0.0, float(score))


def _get_embedding_array(
    embedding_obj: types.MultiModalEmbedding,
) -> np.ndarray:
  if isinstance(embedding_obj, (types.SoundEmbedding, types.TextEmbedding)):
    return embedding_obj.embedding
  raise TypeError(
      f'Unsupported embedding type: {type(embedding_obj)}'
  )


class ReconstructionEvaluator:
  """Evaluates audio reconstruction quality using FAD, KAD, and MSE.

  This evaluator compares a set of original "reference" embeddings against a set
  of "predicted" embeddings from a reconstruction model.

  It computes two types of metrics:
  1.  **Distributional Metrics (FAD, KAD):** These metrics treat all frames from
      all examples as two large "bags" of embeddings. They measure the
      perceptual distance between the *entire distribution* of original sound
      embeddings and the distribution of reconstructed sound embeddings.
      Lower is better.
  2.  **Per-Example Metric (Embedding MSE):** This metric measures the direct,
      frame-by-frame fidelity of the reconstruction for each example. It
      calculates the Mean Squared Error between the original and reconstructed
      embedding for each example, and then averages these MSEs across the
      dataset. Lower is better.
  """

  def __init__(
      self,
      kernel: Literal['gaussian', 'iq', 'imq'] = 'gaussian'
  ):
    """Initializes the ReconstructionEvaluator.

    Args:
      kernel: The kernel to use for KAD calculation ('gaussian', 'iq', 'imq').
    """
    self.kernel = kernel

  def compute_metrics(
      self,
      predictions: types.MultiModalEmbeddingCache,
      references: types.MultiModalEmbeddingCache,
  ) -> list[types.Score]:
    """Computes FAD, KAD, and MSE between two sets of embeddings.

    Args:
      predictions: The embedding cache for the reconstructed audio.
      references: The embedding cache for the original audio.

    Returns:
      A list of `types.Score` objects for FAD, KAD, and Embedding MSE.
    """
    original_embeddings_list = []
    reconstructed_embeddings_list = []
    example_mse_scores = []

    for example_id, ref_embedding in references.items():
      if example_id not in predictions:
        logger.warning(
            'Missing prediction for example_id %s. Skipping.', example_id
        )
        continue

      pred_embedding = predictions[example_id]
      x_example = _get_embedding_array(ref_embedding)
      y_example = _get_embedding_array(pred_embedding)
      original_embeddings_list.append(x_example)
      reconstructed_embeddings_list.append(y_example)

      if x_example.shape == y_example.shape:
        mse = np.mean((x_example - y_example) ** 2)
        example_mse_scores.append(mse)
      else:
        logger.warning(
            'Skipping MSE calculation for example %s: shape mismatch '
            '(original: %s, reconstructed: %s)',
            example_id,
            x_example.shape,
            y_example.shape,
        )

    if not original_embeddings_list:
      logger.error('No matching predictions and references found.')
      return []

    final_metrics = []

    # --- Aggregate and Compute MSE ---
    if example_mse_scores:
      final_mse = float(np.mean(example_mse_scores))
      final_metrics.append(embedding_mse(final_mse))

    # --- Aggregate and Compute FAD & KAD ---
    x_all_frames = np.concatenate(original_embeddings_list, axis=0)
    y_all_frames = np.concatenate(reconstructed_embeddings_list, axis=0)

    logger.info(
        'Calculating FAD/KAD on %d original and %d reconstructed frames.',
        x_all_frames.shape[0],
        y_all_frames.shape[0],
    )

    fad_score = _frechet_distance(x_all_frames, y_all_frames)
    kad_score = _kernel_audio_distance(
        x_all_frames, y_all_frames, kernel=self.kernel
    )

    final_metrics.extend([
        frechet_audio_distance(fad_score),
        kernel_audio_distance(kad_score),
    ])

    return final_metrics
