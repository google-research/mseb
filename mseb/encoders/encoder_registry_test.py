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

from absl.testing import absltest
from absl.testing import parameterized
import pytest

encoder_registry = pytest.importorskip("mseb.encoders.encoder_registry")


@pytest.mark.whisper
@pytest.mark.optional
class EncoderRegistryTest(parameterized.TestCase):

  @parameterized.parameters(
      encoder_registry.raw_encoder_25ms_10ms,
      encoder_registry.whisper_base_speech_to_text,
      encoder_registry.whisper_base_pooled_last,
      encoder_registry.whisper_base_pooled_mean,
      encoder_registry.whisper_base_pooled_max,
      encoder_registry.whisper_forced_alignment,
      encoder_registry.laion_clap_encoder,
      encoder_registry.gemma_with_title_and_context,
      encoder_registry.hf_llm_with_title_and_context,
      encoder_registry.openai_llm_with_title_and_context,
      encoder_registry.openai_speech_to_text,
      encoder_registry.litellm_speech_to_text,
      encoder_registry.litellm_embedding,
  )
  def test_load_encoder(self, meta):
    encoder = meta.load()
    self.assertIsNotNone(encoder)

  def test_get_encoder_metadata(self):
    meta = encoder_registry.get_encoder_metadata(
        "raw_spectrogram_25ms_10ms_mean"
    )
    self.assertIsInstance(meta, encoder_registry.EncoderMetadata)
    self.assertEqual(meta.name, "raw_spectrogram_25ms_10ms_mean")

  def test_get_encoder_metadata_not_found(self):
    with self.assertRaises(ValueError):
      encoder_registry.get_encoder_metadata("non_existent_encoder")


if __name__ == "__main__":
  absltest.main()
