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

import os
import pathlib
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from mseb import dataset
from mseb import types
import numpy as np
import pytest


svq = pytest.importorskip("mseb.tasks.stabilities.speech.svq")


class SVQStabilityTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Locate the testdata directory relative to this file
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent.parent.parent,
        "testdata",
    )
    # Redirect the dataset loader to use the mini local version
    self.enter_context(
        flagsaver.flagsaver((
            dataset._DATASET_BASEPATH,
            os.path.join(testdata_path, "svq_mini"),
        ))
    )

  def test_dynamic_registration(self):
    """Verifies that the factory successfully registers both variant classes."""
    self.assertTrue(hasattr(svq, "SVQEnUsContinuousStability"))
    self.assertTrue(hasattr(svq, "SVQEnUsDiscreteStability"))

  @mock.patch("mseb.utils.download_from_hf")
  def test_svq_stability_sounds_clean_filtering(self, _):
    """Verifies that only 'clean' environment sounds are picked as anchors."""
    # num_augmentations=0 to only count the base anchor sounds.
    task = svq.SVQEnUsContinuousStability(num_augmentations=0)
    sounds = list(task.sounds())
    # Based on the jsonl, there are exactly 3 'clean' en_us utterances.
    self.assertLen(sounds, 3)
    for sound in sounds:
      self.assertEqual(sound.context.language, "en_us")
      # Ensure no noise-environment IDs were accidentally included.
      self.assertNotIn(sound.context.id, [
          "utt_6844631007344632667",   # background_speech
          "utt_2295501949967963013",   # media_noise
          "utt_15933473411391011897",  # traffic_noise
      ])

  @mock.patch("mseb.utils.download_from_hf")
  def test_svq_stability_augmentation_expansion(self, _):
    """Verifies that sounds are correctly expanded by the number of augmentations."""
    # 3 clean anchors * (1 clean + 2 augmented) = 9 total sounds.
    task = svq.SVQEnUsContinuousStability(num_augmentations=2)
    sounds = list(task.sounds())
    self.assertLen(sounds, 9)
    # Check naming convention of an augmented sample
    # (StabilityTask usually names them {base_id}_aug_{n})
    aug_ids = [s.context.id for s in sounds if "aug" in s.context.id]
    self.assertLen(aug_ids, 6)
    self.assertIn("utt_13729869686284260222_aug_0", aug_ids)

  @mock.patch("mseb.utils.download_from_hf")
  def test_metadata_main_score_assignment(self, _):
    """Verifies that the factory assigns the correct main_score for ranking."""
    cont_task = svq.SVQEnUsContinuousStability()
    disc_task = svq.SVQEnUsDiscreteStability()
    self.assertEqual(cont_task.metadata.main_score, "Corpus_Mean_CED")
    self.assertEqual(disc_task.metadata.main_score, "Corpus_Mean_UED")

  @mock.patch("mseb.utils.download_from_hf")
  def test_compute_scores_integration(self, _):
    """Verifies compute_scores processes multi-modal embeddings correctly."""
    task = svq.SVQEnUsContinuousStability(num_augmentations=1)
    cache = mock.MagicMock(spec=types.MultiModalEmbeddingCache)
    # Mock a sound embedding to simulate a continuous model output.
    dummy_emb = types.SoundEmbedding(
        embedding=np.zeros((2, 128)),
        timestamps=np.zeros((2, 2)),
        context=types.SoundContextParams(id="test", sample_rate=16000, length=2)
    )
    cache.get.return_value = dummy_emb
    # The parent StabilityTask.compute_scores should handle the math.
    scores_dict = task.compute_scores(cache)
    self.assertIn("stability", scores_dict)
    # Ensure CED is present in the results.
    metric_names = [s.metric for s in scores_dict["stability"]]
    self.assertTrue(any("CED" in name for name in metric_names))


if __name__ == "__main__":
  absltest.main()
