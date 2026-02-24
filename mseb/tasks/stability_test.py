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

from typing import Iterable
import unittest
from unittest import mock

from mseb import types
from mseb.tasks import stability
import numpy as np


class MockStabilityTask(stability.StabilityTask):
  """A concrete implementation of StabilityTask for testing."""

  def base_sounds(self) -> Iterable[types.Sound]:
    # Yields a single reference sound
    yield types.Sound(
        waveform=np.zeros(16000),
        context=types.SoundContextParams(
            id="test_sample",
            sample_rate=16000,
            length=16000
        )
    )


class StabilityTaskTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    # Task with 2 augmentations for quick testing
    self.task = MockStabilityTask(num_augmentations=2)

  def _make_emb(self, data: np.ndarray, dtype=None) -> types.SoundEmbedding:
    """Helper to create a valid SoundEmbedding with all required fields."""
    if dtype is None:
      dtype = data.dtype
    return types.SoundEmbedding(
        embedding=data.astype(dtype),
        timestamps=np.zeros((data.shape[0], 2)),
        context=types.SoundContextParams(
            id="test",
            sample_rate=16000,
            length=data.shape[0]
        )
    )

  def test_sounds_generation_flow(self):
    all_sounds = list(self.task.sounds())
    self.assertEqual(len(all_sounds), 3)  # 1 base + 2 augs
    self.assertEqual(all_sounds[0].context.id, "test_sample")
    self.assertIn("aug_0", all_sounds[1].context.id)
    self.assertIn("aug_1", all_sounds[2].context.id)

  def test_compute_scores_ced_logic(self):
    cache = mock.MagicMock(spec=types.MultiModalEmbeddingCache)
    # Ref: Two orthogonal frames on the unit sphere
    ref_emb = self._make_emb(np.array([[1.0, 0.0], [0.0, 1.0]]))
    # Aug: Frames are flipped 180 degrees (Max distance on unit sphere)
    # Each frame has an L2 distance of 2.0 from the reference
    aug0_emb = self._make_emb(np.array([[1.0, 0.0], [0.0, 1.0]]))
    aug1_emb = self._make_emb(np.array([[-1.0, 0.0], [0.0, -1.0]]))

    embedding_map = {
        "test_sample": ref_emb,
        "test_sample_aug_0": aug0_emb,
        "test_sample_aug_1": aug1_emb,
    }
    cache.get.side_effect = embedding_map.get
    scores_dict = self.task.compute_scores(cache)
    # Filter for the CED (Cosine Embedding Distance) results
    scores = {
        s.metric: s for s in scores_dict["stability"] if "CED" in s.metric
    }
    self.assertAlmostEqual(scores["Corpus_Mean_CED"].value, 0.5)
    self.assertAlmostEqual(scores["Mean_Local_IS_CED"].value, 0.5)
    self.assertAlmostEqual(scores["Mean_Local_IS_CED"].std, 0.5)

  def test_compute_scores_continuous_suite_logic(self):
    cache = mock.MagicMock(spec=types.MultiModalEmbeddingCache)
    ref_data = np.array([[1.0, 0.0], [0.0, 1.0]])
    ref_emb = self._make_emb(ref_data)
    aug_data = np.array([[-1.0, 0.0], [0.0, -1.0]])
    aug_emb = self._make_emb(aug_data)

    embedding_map = {
        "test_sample": ref_emb,
        "test_sample_aug_0": aug_emb,
        "test_sample_aug_1": aug_emb,
    }
    cache.get.side_effect = embedding_map.get

    scores_dict = self.task.compute_scores(cache)
    scores = {s.metric: s for s in scores_dict["stability"]}

    # Total Cost: 2.0 (frame 1) + 2.0 (frame 2) = 4.0
    # Normalization: 4.0 / (2.0 factor * 2 length) = 1.0 (100% drift)
    self.assertAlmostEqual(scores["Corpus_Mean_CED"].value, 1.0)
    self.assertAlmostEqual(scores["Mean_Local_IS_CED"].std, 0.0)

    # In this simple case, DTW path matches CED substitution path
    # Total Cost: 4.0. Note: DTW is typically reported as raw cost.
    self.assertAlmostEqual(scores["Corpus_Mean_DTW"].value, 4.0 / 2.0)

    self.assertAlmostEqual(
        scores["Corpus_Mean_L2"].value,
        1.0,
        places=5
    )

  def test_compute_scores_ued_logic(self):
    cache = mock.MagicMock(spec=types.MultiModalEmbeddingCache)
    # UED (Unit Edit Distance) logic usually expects 1D arrays of discrete units
    ref_emb = self._make_emb(np.array([1, 2, 3]), dtype=np.int32)
    aug_emb = self._make_emb(np.array([1, 9, 3]), dtype=np.int32)

    embedding_map = {
        "test_sample": ref_emb,
        "test_sample_aug_0": aug_emb,
        "test_sample_aug_1": aug_emb,
    }
    cache.get.side_effect = embedding_map.get
    scores_dict = self.task.compute_scores(cache)
    scores = {
        s.metric: s for s in scores_dict["stability"] if "UED" in s.metric
    }
    # 1 edit (substitution) / 3 total units = 0.333
    self.assertAlmostEqual(scores["Corpus_Mean_UED"].value, 1/3)


if __name__ == "__main__":
  unittest.main()
