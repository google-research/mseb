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

  def __init__(self, sounds_data: list[np.ndarray], **kwargs):
    super().__init__(**kwargs)
    self.sounds_data = sounds_data

  def base_sounds(self) -> Iterable[types.Sound]:
    for i, _ in enumerate(self.sounds_data):
      yield types.Sound(
          waveform=np.zeros(16000),
          context=types.SoundContextParams(
              id=f"sample_{i}",
              sample_rate=16000,
              length=16000
          )
      )


class StabilityTaskTest(unittest.TestCase):

  def _make_emb(self, data: np.ndarray) -> types.SoundEmbedding:
    """Helper to create a valid SoundEmbedding."""
    return types.SoundEmbedding(
        embedding=data.astype(np.float32),
        timestamps=np.zeros((data.shape[0], 2)),
        context=types.SoundContextParams(
            id="test", sample_rate=16000, length=data.shape[0]
        )
    )

  def test_sounds_generation_flow(self):
    # 2 base sounds * (1 clean + 2 augs) = 6 total sounds
    task = MockStabilityTask(
        sounds_data=[np.zeros(10), np.zeros(10)],
        num_augmentations=2
    )
    all_sounds = list(task.sounds())
    self.assertEqual(len(all_sounds), 6)
    # Check first triplet
    self.assertEqual(all_sounds[0].context.id, "sample_0")
    self.assertEqual(all_sounds[1].context.id, "sample_0_aug_0")
    self.assertEqual(all_sounds[2].context.id, "sample_0_aug_1")

  def test_compute_scores_hierarchical_logic(self):
    # Setup: 2 base utterances, each with 2 augmentations
    task = MockStabilityTask(
        sounds_data=[np.zeros(10), np.zeros(10)],
        num_augmentations=2
    )
    cache = mock.MagicMock(spec=types.MultiModalEmbeddingCache)
    # Reference frame (Unit Sphere)
    ref_vec = np.array([[1.0, 0.0]])
    # Utterance 0: Perfectly stable (No drift across variants)
    # Drifts = [0.0, 0.0] -> Mean=0.0, Std=0.0
    u0_ref = self._make_emb(ref_vec)

    # Utterance 1: Volatile (Aug 0 is stable, Aug 1 is 180° flip)
    # Aug 1 Sub cost = 2.0. Normalized drift (2.0 / (2.0*1)) = 1.0
    # Drifts = [0.0, 1.0] -> Mean=0.5, Std=0.5
    u1_ref = self._make_emb(ref_vec)
    flip_vec = np.array([[-1.0, 0.0]])
    embedding_map = {
        "sample_0": u0_ref,
        "sample_0_aug_0": u0_ref,
        "sample_0_aug_1": u0_ref,
        "sample_1": u1_ref,
        "sample_1_aug_0": u1_ref,
        "sample_1_aug_1": self._make_emb(flip_vec),
    }
    cache.get.side_effect = embedding_map.get
    scores_dict = task.compute_scores(cache)
    scores = {
        s.metric: s for s in scores_dict["stability"] if "CED" in s.metric
    }
    # 1. Corpus Mean (Micro-Average)
    # Total Costs: 0+0 (U0) + 0+2.0 (U1) = 2.0
    # Total Norm: (2.0 factor * 1 len * 4 total variant pairs) = 8.0
    # 2.0 / 8.0 = 0.25
    self.assertAlmostEqual(scores["Corpus_Mean_CED"].value, 0.25)

    # 2. Mean Local IS (Macro-Average)
    # Mean of Means: (0.0 + 0.5) / 2 = 0.25
    # Mean of Stds (Instability): (0.0 + 0.5) / 2 = 0.25
    self.assertAlmostEqual(scores["Mean_Local_IS_CED"].value, 0.25)
    self.assertAlmostEqual(scores["Mean_Local_IS_CED"].std, 0.25)

  def test_insertion_overflow_logic(self):
    task = MockStabilityTask(sounds_data=[np.zeros(10)], num_augmentations=1)
    cache = mock.MagicMock(spec=types.MultiModalEmbeddingCache)

    ref = self._make_emb(np.array([[1.0, 0.0]]))
    # Hypothesis is 3x the length.
    # 2 Insertions (cost 2.0 each) + 1 Match (cost 0)
    # Raw Cost = 4.0. Normalized = 4.0 / (2.0 * 1 ref_len) = 2.0
    aug = self._make_emb(np.array([[1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]))

    embedding_map = {"sample_0": ref, "sample_0_aug_0": aug}
    cache.get.side_effect = embedding_map.get

    scores_dict = task.compute_scores(cache)
    score = next(
        s for s in scores_dict["stability"] if s.metric == "Corpus_Mean_CED"
    )

    self.assertEqual(score.value, 2.0)
    self.assertEqual(score.max, float("inf"))


if __name__ == "__main__":
  unittest.main()
