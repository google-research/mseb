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

"""Tests for BrainEncodingEvaluator class."""

from absl.testing import absltest
import jax.numpy as jnp
from mseb import types
from mseb.evaluators import brain_encoding_evaluator
import numpy as np


class ApplyFirDelaysTest(absltest.TestCase):

  def test_single_delay(self):
    features = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    delayed = brain_encoding_evaluator.apply_fir_delays(features, delays=[1])
    expected = np.array([[0, 0], [1, 2], [3, 4]], dtype=np.float32)
    np.testing.assert_array_equal(delayed, expected)

  def test_zero_delay(self):
    features = np.array([[1, 2], [3, 4]], dtype=np.float32)
    delayed = brain_encoding_evaluator.apply_fir_delays(features, delays=[0])
    np.testing.assert_array_equal(delayed, features)

  def test_multiple_delays(self):
    features = np.array([[1], [2], [3], [4]], dtype=np.float32)
    delayed = brain_encoding_evaluator.apply_fir_delays(
        features, delays=[0, 1, 2]
    )
    expected = np.array(
        [[1, 0, 0], [2, 1, 0], [3, 2, 1], [4, 3, 2]], dtype=np.float32
    )
    np.testing.assert_array_equal(delayed, expected)

  def test_negative_delay(self):
    features = np.array([[1], [2], [3], [4]], dtype=np.float32)
    delayed = brain_encoding_evaluator.apply_fir_delays(features, delays=[-1])
    expected = np.array([[2], [3], [4], [0]], dtype=np.float32)
    np.testing.assert_array_equal(delayed, expected)

  def test_mixed_delays(self):
    features = np.array([[1], [2], [3]], dtype=np.float32)
    delayed = brain_encoding_evaluator.apply_fir_delays(
        features, delays=[-1, 0, 1]
    )
    expected = np.array([[2, 1, 0], [3, 2, 1], [0, 3, 2]], dtype=np.float32)
    np.testing.assert_array_equal(delayed, expected)

  def test_large_delay_all_zeros(self):
    """Delay larger than T should produce all zeros."""
    features = np.array([[1], [2], [3]], dtype=np.float32)
    delayed = brain_encoding_evaluator.apply_fir_delays(features, delays=[5])
    np.testing.assert_array_equal(delayed, np.zeros((3, 1), dtype=np.float32))

  def test_output_shape(self):
    features = np.random.randn(10, 5).astype(np.float32)
    delays = [1, 2, 3, 4]
    delayed = brain_encoding_evaluator.apply_fir_delays(features, delays)
    self.assertEqual(delayed.shape, (10, 20))

  def test_output_dtype_matches_input(self):
    features = np.random.randn(5, 3).astype(np.float64)
    delayed = brain_encoding_evaluator.apply_fir_delays(features, delays=[1])
    self.assertEqual(delayed.dtype, np.float64)

  def test_zero_padded_not_circular(self):
    """Verify padding is zero, not circular."""
    features = np.array([[10], [20], [30]], dtype=np.float32)
    delayed = brain_encoding_evaluator.apply_fir_delays(features, delays=[2])
    # First two rows should be zero-padded.
    np.testing.assert_array_equal(delayed[:2], [[0], [0]])
    np.testing.assert_array_equal(delayed[2:], [[10]])


class ColumnwiseCorrelationTest(absltest.TestCase):
  """Tests for the NumPy _columnwise_correlation."""

  def test_perfect_correlation(self):
    a = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    corr = brain_encoding_evaluator._columnwise_correlation(a, a)
    np.testing.assert_array_almost_equal(corr, [1.0, 1.0])

  def test_anticorrelation(self):
    a = np.array([[1], [2], [3]], dtype=np.float64)
    b = np.array([[3], [2], [1]], dtype=np.float64)
    corr = brain_encoding_evaluator._columnwise_correlation(a, b)
    np.testing.assert_array_almost_equal(corr, [-1.0])

  def test_zero_correlation(self):
    a = np.array([[1], [0], [-1], [0]], dtype=np.float64)
    b = np.array([[0], [1], [0], [-1]], dtype=np.float64)
    corr = brain_encoding_evaluator._columnwise_correlation(a, b)
    np.testing.assert_array_almost_equal(corr, [0.0])

  def test_constant_column_returns_zero(self):
    """Constant column should not cause NaN."""
    a = np.array([[5], [5], [5]], dtype=np.float64)
    b = np.array([[1], [2], [3]], dtype=np.float64)
    corr = brain_encoding_evaluator._columnwise_correlation(a, b)
    np.testing.assert_array_almost_equal(corr, [0.0])

  def test_output_shape(self):
    a = np.random.randn(20, 7)
    b = np.random.randn(20, 7)
    corr = brain_encoding_evaluator._columnwise_correlation(a, b)
    self.assertEqual(corr.shape, (7,))

  def test_correlation_in_range(self):
    np.random.seed(99)
    a = np.random.randn(50, 10)
    b = np.random.randn(50, 10)
    corr = brain_encoding_evaluator._columnwise_correlation(a, b)
    self.assertTrue(np.all(corr >= -1.0))
    self.assertTrue(np.all(corr <= 1.0))


class JaxColumnwiseCorrelationTest(absltest.TestCase):
  """Tests for the JAX _jax_columnwise_correlation."""

  def test_perfect_correlation(self):
    a = jnp.array([[1, 2], [3, 4], [5, 6]], dtype=jnp.float32)
    corr = brain_encoding_evaluator._jax_columnwise_correlation(a, a)
    np.testing.assert_array_almost_equal(
        np.asarray(corr), [1.0, 1.0], decimal=5
    )

  def test_anticorrelation(self):
    a = jnp.array([[1], [2], [3]], dtype=jnp.float32)
    b = jnp.array([[3], [2], [1]], dtype=jnp.float32)
    corr = brain_encoding_evaluator._jax_columnwise_correlation(a, b)
    np.testing.assert_array_almost_equal(np.asarray(corr), [-1.0], decimal=5)

  def test_constant_column_returns_zero(self):
    a = jnp.array([[5], [5], [5]], dtype=jnp.float32)
    b = jnp.array([[1], [2], [3]], dtype=jnp.float32)
    corr = brain_encoding_evaluator._jax_columnwise_correlation(a, b)
    np.testing.assert_array_almost_equal(np.asarray(corr), [0.0], decimal=5)

  def test_matches_numpy_version(self):
    """JAX and NumPy correlation should agree."""
    np.random.seed(42)
    a = np.random.randn(30, 5).astype(np.float32)
    b = np.random.randn(30, 5).astype(np.float32)
    np_corr = brain_encoding_evaluator._columnwise_correlation(a, b)
    jax_corr = brain_encoding_evaluator._jax_columnwise_correlation(
        jnp.array(a), jnp.array(b)
    )
    np.testing.assert_array_almost_equal(
        np.asarray(jax_corr), np_corr, decimal=4
    )


class RidgeSvdTest(absltest.TestCase):

  def test_recovers_linear_relationship(self):
    """Ridge regression should recover a simple linear relationship."""
    np.random.seed(42)
    n_train, n_test, n_features, n_voxels = 200, 50, 10, 5

    true_weights = np.random.randn(n_features, n_voxels)

    stim_train = np.random.randn(n_train, n_features)
    stim_test = np.random.randn(n_test, n_features)
    resp_train = stim_train @ true_weights + 0.1 * np.random.randn(
        n_train, n_voxels
    )
    resp_test = stim_test @ true_weights + 0.1 * np.random.randn(
        n_test, n_voxels
    )

    alphas = np.logspace(-2, 4, 10).tolist()
    predictions, best_alphas = brain_encoding_evaluator.ridge_svd(
        stim_train, resp_train, stim_test, alphas
    )

    self.assertEqual(predictions.shape, (n_test, n_voxels))
    self.assertEqual(best_alphas.shape, (n_voxels,))

    corrs = brain_encoding_evaluator._columnwise_correlation(
        predictions, resp_test
    )
    for corr in corrs:
      self.assertGreater(corr, 0.9)

  def test_output_dtype_is_float32(self):
    np.random.seed(0)
    stim_train = np.random.randn(40, 4)
    resp_train = np.random.randn(40, 2)
    stim_test = np.random.randn(10, 4)
    predictions, _ = brain_encoding_evaluator.ridge_svd(
        stim_train, resp_train, stim_test, [1.0, 10.0]
    )
    self.assertEqual(predictions.dtype, np.float32)

  def test_single_alpha(self):
    """Should work with just one alpha."""
    np.random.seed(1)
    stim_train = np.random.randn(40, 4).astype(np.float32)
    resp_train = np.random.randn(40, 3).astype(np.float32)
    stim_test = np.random.randn(10, 4).astype(np.float32)
    predictions, best_alphas = brain_encoding_evaluator.ridge_svd(
        stim_train, resp_train, stim_test, [1.0]
    )
    self.assertEqual(predictions.shape, (10, 3))
    # With one alpha, all voxels should pick it.
    np.testing.assert_array_equal(best_alphas, [1.0, 1.0, 1.0])

  def test_best_alphas_from_candidates(self):
    """best_alphas values should be from the provided candidates."""
    np.random.seed(2)
    alphas = [0.01, 1.0, 100.0]
    stim_train = np.random.randn(60, 5).astype(np.float32)
    resp_train = np.random.randn(60, 4).astype(np.float32)
    stim_test = np.random.randn(10, 5).astype(np.float32)
    _, best_alphas = brain_encoding_evaluator.ridge_svd(
        stim_train, resp_train, stim_test, alphas
    )
    for a in best_alphas:
      self.assertIn(a, alphas)

  def test_noisy_data_low_correlation(self):
    """Pure noise should yield low correlations."""
    np.random.seed(3)
    stim_train = np.random.randn(100, 5).astype(np.float32)
    resp_train = np.random.randn(100, 5).astype(np.float32)
    stim_test = np.random.randn(20, 5).astype(np.float32)
    resp_test = np.random.randn(20, 5).astype(np.float32)
    predictions, _ = brain_encoding_evaluator.ridge_svd(
        stim_train, resp_train, stim_test, [1.0, 100.0]
    )
    corrs = brain_encoding_evaluator._columnwise_correlation(
        predictions, resp_test
    )
    # Random data should have near-zero mean correlation.
    self.assertLess(abs(np.mean(corrs)), 0.5)


class CvOneAlphaTest(absltest.TestCase):
  """Tests for the JIT-compiled _cv_one_alpha function."""

  def test_returns_per_voxel_correlation(self):
    np.random.seed(10)
    n_half, d, v = 50, 5, 3
    stim = np.random.randn(2 * n_half, d).astype(np.float32)
    resp = np.random.randn(2 * n_half, v).astype(np.float32)

    u1, s1, vt1 = jnp.linalg.svd(jnp.array(stim[:n_half]), full_matrices=False)
    u2, s2, vt2 = jnp.linalg.svd(jnp.array(stim[n_half:]), full_matrices=False)

    resp_j = jnp.array(resp)
    avg_corr = brain_encoding_evaluator._cv_one_alpha(
        s1,
        vt1.T,
        u1.T @ resp_j[:n_half],
        jnp.array(stim[n_half:]),
        resp_j[:n_half],
        s2,
        vt2.T,
        u2.T @ resp_j[n_half:],
        jnp.array(stim[:n_half]),
        resp_j[n_half:],
        jnp.float32(1.0),
    )
    self.assertEqual(np.asarray(avg_corr).shape, (v,))
    # Correlations should be in [-1, 1].
    self.assertTrue(np.all(np.asarray(avg_corr) >= -1.0))
    self.assertTrue(np.all(np.asarray(avg_corr) <= 1.0))


class PredictForAlphaTest(absltest.TestCase):
  """Tests for the JIT-compiled _predict_for_alpha function."""

  def test_output_shape(self):
    np.random.seed(11)
    d, v, t_test = 5, 3, 10
    s = jnp.array(np.random.rand(d).astype(np.float32))
    vt_t = jnp.array(np.random.randn(d, d).astype(np.float32))
    ut_resp = jnp.array(np.random.randn(d, v).astype(np.float32))
    stim_test = jnp.array(np.random.randn(t_test, d).astype(np.float32))

    preds = brain_encoding_evaluator._predict_for_alpha(
        s, vt_t, ut_resp, stim_test, jnp.float32(1.0)
    )
    self.assertEqual(np.asarray(preds).shape, (t_test, v))

  def test_high_regularization_shrinks_predictions(self):
    """Very large alpha should shrink predictions toward zero."""
    np.random.seed(12)
    d, v, t_test = 5, 3, 10
    s = jnp.array(np.random.rand(d).astype(np.float32))
    vt_t = jnp.array(np.random.randn(d, d).astype(np.float32))
    ut_resp = jnp.array(np.random.randn(d, v).astype(np.float32))
    stim_test = jnp.array(np.random.randn(t_test, d).astype(np.float32))

    preds_low = brain_encoding_evaluator._predict_for_alpha(
        s, vt_t, ut_resp, stim_test, jnp.float32(0.01)
    )
    preds_high = brain_encoding_evaluator._predict_for_alpha(
        s, vt_t, ut_resp, stim_test, jnp.float32(1e10)
    )
    # Higher regularization should produce smaller magnitude predictions.
    self.assertGreater(
        float(jnp.mean(jnp.abs(preds_low))),
        float(jnp.mean(jnp.abs(preds_high))),
    )


class BrainEncodingExampleTest(absltest.TestCase):

  def test_dataclass_fields(self):
    ex = brain_encoding_evaluator.BrainEncodingExample(
        sound_id='story_a', tr_index_start=0, tr_index_end=20
    )
    self.assertEqual(ex.sound_id, 'story_a')
    self.assertEqual(ex.tr_index_start, 0)
    self.assertEqual(ex.tr_index_end, 20)

  def test_negative_indices(self):
    """Negative indices should be allowed (used for trimming)."""
    ex = brain_encoding_evaluator.BrainEncodingExample(
        sound_id='story_b', tr_index_start=10, tr_index_end=-5
    )
    self.assertEqual(ex.tr_index_start, 10)
    self.assertEqual(ex.tr_index_end, -5)


class BuildFeatureMatrixTest(absltest.TestCase):
  """Tests for the _build_feature_matrix method."""

  def _make_embedding(self, sound_id, n_trs, n_dim):
    timestamps = np.array(
        [[i * 2.0, (i + 1) * 2.0] for i in range(n_trs)], dtype=np.float32
    )
    return types.SoundEmbedding(
        embedding=np.random.randn(n_trs, n_dim).astype(np.float32),
        timestamps=timestamps,  # pyrefly: ignore[bad-argument-type]
        context=types.SoundContextParams(id=sound_id, sample_rate=0, length=0),
    )

  def test_single_example_full_range(self):
    np.random.seed(20)
    evaluator = brain_encoding_evaluator.BrainEncodingEvaluator()
    emb = self._make_embedding('s1', n_trs=30, n_dim=4)
    embeddings = {'s1': emb}
    examples = [
        brain_encoding_evaluator.BrainEncodingExample(
            sound_id='s1', tr_index_start=0, tr_index_end=30
        )
    ]
    features = evaluator._build_feature_matrix(embeddings, examples)
    self.assertEqual(features.shape, (30, 4))

  def test_slicing_with_trim(self):
    np.random.seed(21)
    evaluator = brain_encoding_evaluator.BrainEncodingEvaluator()
    emb = self._make_embedding('s1', n_trs=30, n_dim=4)
    embeddings = {'s1': emb}
    examples = [
        brain_encoding_evaluator.BrainEncodingExample(
            sound_id='s1', tr_index_start=5, tr_index_end=-3
        )
    ]
    features = evaluator._build_feature_matrix(embeddings, examples)
    self.assertEqual(features.shape, (22, 4))  # 30 - 5 - 3 = 22

  def test_multiple_examples_vstacked(self):
    np.random.seed(22)
    evaluator = brain_encoding_evaluator.BrainEncodingEvaluator()
    emb_a = self._make_embedding('a', n_trs=20, n_dim=4)
    emb_b = self._make_embedding('b', n_trs=15, n_dim=4)
    embeddings = {'a': emb_a, 'b': emb_b}
    examples = [
        brain_encoding_evaluator.BrainEncodingExample(
            sound_id='a', tr_index_start=0, tr_index_end=20
        ),
        brain_encoding_evaluator.BrainEncodingExample(
            sound_id='b', tr_index_start=0, tr_index_end=15
        ),
    ]
    features = evaluator._build_feature_matrix(embeddings, examples)
    self.assertEqual(features.shape, (35, 4))

  def test_output_is_z_scored(self):
    """Each story's features are z-scored before vstacking."""
    np.random.seed(23)
    evaluator = brain_encoding_evaluator.BrainEncodingEvaluator()
    emb = self._make_embedding('s1', n_trs=50, n_dim=4)
    embeddings = {'s1': emb}
    examples = [
        brain_encoding_evaluator.BrainEncodingExample(
            sound_id='s1', tr_index_start=0, tr_index_end=50
        )
    ]
    features = evaluator._build_feature_matrix(embeddings, examples)
    # Z-scored: mean ≈ 0, std ≈ 1 per column.
    np.testing.assert_array_almost_equal(
        features.mean(axis=0), np.zeros(4), decimal=5
    )
    np.testing.assert_array_almost_equal(
        features.std(axis=0), np.ones(4), decimal=1
    )

  def test_nan_replaced_with_zero(self):
    """Constant columns (std=0) should produce 0 instead of NaN."""
    evaluator = brain_encoding_evaluator.BrainEncodingEvaluator()
    # Embedding with a constant column.
    emb_data = np.random.randn(20, 3).astype(np.float32)
    emb_data[:, 1] = 5.0  # Constant column
    timestamps = np.array(
        [[i * 2.0, (i + 1) * 2.0] for i in range(20)], dtype=np.float32
    )
    emb = types.SoundEmbedding(
        embedding=emb_data,
        timestamps=timestamps,  # pyrefly: ignore[bad-argument-type]
        context=types.SoundContextParams(id='s1', sample_rate=0, length=0),
    )
    embeddings = {'s1': emb}
    examples = [
        brain_encoding_evaluator.BrainEncodingExample(
            sound_id='s1', tr_index_start=0, tr_index_end=20
        )
    ]
    features = evaluator._build_feature_matrix(embeddings, examples)
    # No NaNs.
    self.assertFalse(np.any(np.isnan(features)))
    # Constant column should be all zeros.
    np.testing.assert_array_equal(features[:, 1], np.zeros(20))

  def test_frame_level_input_downsampled(self):
    """Frame-level embeddings (e.g. 50 Hz) are downsampled to TR."""
    np.random.seed(25)
    evaluator = brain_encoding_evaluator.BrainEncodingEvaluator(tr_duration=2.0)
    n_frames = 3000  # 60 seconds at 50 Hz = 30 TRs at 2s
    n_dim = 4
    frame_dur = 0.02  # 20ms = 50 Hz
    timestamps = np.array(
        [[i * frame_dur, (i + 1) * frame_dur] for i in range(n_frames)],
        dtype=np.float32,
    )
    emb = types.SoundEmbedding(
        embedding=np.random.randn(n_frames, n_dim).astype(np.float32),
        timestamps=timestamps,  # pyrefly: ignore[bad-argument-type]
        context=types.SoundContextParams(id='s1', sample_rate=0, length=0),
    )
    embeddings = {'s1': emb}
    examples = [
        brain_encoding_evaluator.BrainEncodingExample(
            sound_id='s1', tr_index_start=0, tr_index_end=30
        )
    ]
    features = evaluator._build_feature_matrix(embeddings, examples)
    # Should be downsampled to 30 TRs.
    self.assertEqual(features.shape, (30, n_dim))


class DownsampleToTrTest(absltest.TestCase):
  """Tests for the _downsample_to_tr method."""

  def _make_evaluator(self, tr_duration=2.0):
    return brain_encoding_evaluator.BrainEncodingEvaluator(
        tr_duration=tr_duration
    )

  def test_already_tr_aligned_passthrough(self):
    """TR-aligned input should be returned unchanged."""
    evaluator = self._make_evaluator(tr_duration=2.0)
    embedding = np.random.randn(30, 4).astype(np.float32)
    timestamps = np.array(
        [[i * 2.0, (i + 1) * 2.0] for i in range(30)], dtype=np.float32
    )
    result = evaluator._downsample_to_tr(embedding, timestamps)
    np.testing.assert_array_equal(result, embedding)

  def test_50hz_to_tr(self):
    """50 Hz (20ms frames) → 0.5 Hz (2s TR) = 100× downsampling."""
    evaluator = self._make_evaluator(tr_duration=2.0)
    n_frames = 1000  # 20 seconds = 10 TRs
    n_dim = 8
    frame_dur = 0.02
    embedding = np.ones((n_frames, n_dim), dtype=np.float32)
    timestamps = np.array(
        [[i * frame_dur, (i + 1) * frame_dur] for i in range(n_frames)],
        dtype=np.float32,
    )
    result = evaluator._downsample_to_tr(embedding, timestamps)
    self.assertEqual(result.shape, (10, n_dim))
    # All ones averaged = ones.
    np.testing.assert_array_almost_equal(result, np.ones((10, n_dim)))

  def test_averaging_is_correct(self):
    """Verify block averaging produces correct values."""
    evaluator = self._make_evaluator(tr_duration=1.0)
    # 10 Hz frames, 1s TR → 10 frames per TR.
    frame_dur = 0.1
    n_frames = 20  # 2 TRs
    embedding = np.arange(n_frames, dtype=np.float32).reshape(-1, 1)
    timestamps = np.array(
        [[i * frame_dur, (i + 1) * frame_dur] for i in range(n_frames)],
        dtype=np.float32,
    )
    result = evaluator._downsample_to_tr(embedding, timestamps)
    self.assertEqual(result.shape, (2, 1))
    # First TR: mean(0..9) = 4.5, Second TR: mean(10..19) = 14.5.
    np.testing.assert_array_almost_equal(result, [[4.5], [14.5]])

  def test_preserves_dtype(self):
    evaluator = self._make_evaluator(tr_duration=2.0)
    embedding = np.random.randn(200, 4).astype(np.float32)
    timestamps = np.array(
        [[i * 0.02, (i + 1) * 0.02] for i in range(200)], dtype=np.float32
    )
    result = evaluator._downsample_to_tr(embedding, timestamps)
    self.assertEqual(result.dtype, np.float32)

  def test_single_frame_passthrough(self):
    """Single-frame embedding should pass through."""
    evaluator = self._make_evaluator()
    embedding = np.array([[1.0, 2.0]], dtype=np.float32)
    timestamps = np.array([[0.0, 2.0]], dtype=np.float32)
    result = evaluator._downsample_to_tr(embedding, timestamps)
    np.testing.assert_array_equal(result, embedding)

  def test_truncates_incomplete_tr(self):
    """Frames that don't fill a complete TR are truncated."""
    evaluator = self._make_evaluator(tr_duration=1.0)
    # 15 frames at 10 Hz = 1.5s → only 1 complete TR.
    frame_dur = 0.1
    n_frames = 15
    embedding = np.ones((n_frames, 2), dtype=np.float32)
    timestamps = np.array(
        [[i * frame_dur, (i + 1) * frame_dur] for i in range(n_frames)],
        dtype=np.float32,
    )
    result = evaluator._downsample_to_tr(embedding, timestamps)
    self.assertEqual(result.shape, (1, 2))


class BrainEncodingEvaluatorTest(absltest.TestCase):

  def _make_sound_embedding(
      self, sound_id: str, n_trs: int, n_dim: int, tr_duration: float = 2.0
  ) -> types.SoundEmbedding:
    """Creates a SoundEmbedding with one frame per TR."""
    timestamps = np.array(
        [[i * tr_duration, (i + 1) * tr_duration] for i in range(n_trs)],
        dtype=np.float32,
    )
    return types.SoundEmbedding(
        embedding=np.random.randn(n_trs, n_dim).astype(np.float32),
        timestamps=timestamps,  # pyrefly: ignore[bad-argument-type]
        context=types.SoundContextParams(id=sound_id, sample_rate=0, length=0),
    )

  def test_end_to_end_with_synthetic_data(self):
    """Full evaluation pipeline with synthetic data."""
    np.random.seed(42)
    n_dim = 16
    n_voxels = 10
    n_train_trs = 100
    n_test_trs = 30

    embeddings = {}
    train_examples = []
    test_examples = []

    for i in range(5):
      sound_id = f'train_stim_{i}'
      n_trs = 20
      embeddings[sound_id] = self._make_sound_embedding(
          sound_id, n_trs=n_trs, n_dim=n_dim
      )
      train_examples.append(
          brain_encoding_evaluator.BrainEncodingExample(
              sound_id=sound_id,
              tr_index_start=0,
              tr_index_end=n_trs,
          )
      )

    for i in range(2):
      sound_id = f'test_stim_{i}'
      n_trs = 15
      embeddings[sound_id] = self._make_sound_embedding(
          sound_id, n_trs=n_trs, n_dim=n_dim
      )
      test_examples.append(
          brain_encoding_evaluator.BrainEncodingExample(
              sound_id=sound_id,
              tr_index_start=0,
              tr_index_end=n_trs,
          )
      )

    fmri_train = np.random.randn(n_train_trs, n_voxels).astype(np.float32)
    fmri_test = np.random.randn(n_test_trs, n_voxels).astype(np.float32)

    evaluator = brain_encoding_evaluator.BrainEncodingEvaluator(
        tr_duration=2.0,
        delays=[1, 2, 3, 4],
        alphas=np.logspace(0, 4, 5).tolist(),
    )

    scores = evaluator(
        embeddings=embeddings,
        train_examples=train_examples,
        test_examples=test_examples,
        fmri_train=fmri_train,
        fmri_test=fmri_test,
    )

    self.assertLen(scores, 3)
    metric_names = {s.metric for s in scores}
    self.assertIn('MeanCorrelation', metric_names)
    self.assertIn('MedianCorrelation', metric_names)
    self.assertIn('FractionSignificant', metric_names)

    for score in scores:
      self.assertGreaterEqual(score.value, score.min)
      self.assertLessEqual(score.value, score.max)

  def test_evaluator_default_alphas(self):
    evaluator = brain_encoding_evaluator.BrainEncodingEvaluator()
    self.assertLen(evaluator.alphas, 20)
    self.assertEqual(evaluator.tr_duration, 2.0)
    self.assertEqual(evaluator.delays, (1, 2, 3, 4))

  def test_evaluator_custom_params(self):
    evaluator = brain_encoding_evaluator.BrainEncodingEvaluator(
        tr_duration=1.5,
        delays=[1, 2],
        alphas=[1.0, 10.0],
    )
    self.assertEqual(evaluator.tr_duration, 1.5)
    self.assertEqual(evaluator.delays, [1, 2])
    self.assertLen(evaluator.alphas, 2)

  def test_fraction_significant_range(self):
    """FractionSignificant should always be in [0, 1]."""
    np.random.seed(50)
    evaluator = brain_encoding_evaluator.BrainEncodingEvaluator(
        alphas=[1.0, 100.0]
    )
    embeddings = {}
    train_examples = []
    for i in range(3):
      sid = f'tr_{i}'
      embeddings[sid] = self._make_sound_embedding(sid, n_trs=20, n_dim=8)
      train_examples.append(
          brain_encoding_evaluator.BrainEncodingExample(
              sound_id=sid,
              tr_index_start=0,
              tr_index_end=20,
          )
      )
    test_examples = []
    for i in range(2):
      sid = f'te_{i}'
      embeddings[sid] = self._make_sound_embedding(sid, n_trs=15, n_dim=8)
      test_examples.append(
          brain_encoding_evaluator.BrainEncodingExample(
              sound_id=sid,
              tr_index_start=0,
              tr_index_end=15,
          )
      )
    scores = evaluator(
        embeddings=embeddings,
        train_examples=train_examples,
        test_examples=test_examples,
        fmri_train=np.random.randn(60, 5).astype(np.float32),
        fmri_test=np.random.randn(30, 5).astype(np.float32),
    )
    frac_score = [s for s in scores if s.metric == 'FractionSignificant'][0]
    self.assertGreaterEqual(frac_score.value, 0.0)
    self.assertLessEqual(frac_score.value, 1.0)


class ScoreFactoryTest(absltest.TestCase):

  def test_mean_correlation_score(self):
    score = brain_encoding_evaluator.mean_correlation_score(0.5, 0.1)
    self.assertEqual(score.metric, 'MeanCorrelation')
    self.assertEqual(score.value, 0.5)
    self.assertEqual(score.std, 0.1)
    self.assertEqual(score.min, -1)
    self.assertEqual(score.max, 1)

  def test_mean_correlation_score_defaults(self):
    score = brain_encoding_evaluator.mean_correlation_score()
    self.assertEqual(score.value, 0.0)
    self.assertIsNone(score.std)

  def test_median_correlation_score(self):
    score = brain_encoding_evaluator.median_correlation_score(0.3)
    self.assertEqual(score.metric, 'MedianCorrelation')
    self.assertEqual(score.value, 0.3)
    self.assertEqual(score.min, -1)
    self.assertEqual(score.max, 1)

  def test_fraction_significant_score(self):
    score = brain_encoding_evaluator.fraction_significant_score(0.7)
    self.assertEqual(score.metric, 'FractionSignificant')
    self.assertEqual(score.value, 0.7)
    self.assertEqual(score.min, 0)
    self.assertEqual(score.max, 1)


if __name__ == '__main__':
  absltest.main()
