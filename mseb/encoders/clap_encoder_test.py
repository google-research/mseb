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

from typing import cast
from unittest import mock

from absl.testing import absltest
from mseb import encoder
from mseb import types
from mseb.encoders import clap_encoder
import numpy as np
import torch


class ClapEncoderTest(absltest.TestCase):
  """Tests for the CLAP encoder classes and factory function."""

  def setUp(self):
    super().setUp()
    self.model_path = "laion/clap-htsat-unfused"
    self.dummy_embedding_dim = 512  # Standard CLAP embedding dimension

    # Dummy data for a Sound object
    self.sound_context = types.SoundContextParams(
        id="sound1",
        sample_rate=48000,
        length=48000
    )
    self.dummy_sound_batch = [
        types.Sound(
            waveform=np.random.randn(48000).astype(np.float32),
            context=self.sound_context,
        )
    ]

    # Dummy data for a Text object
    self.text_context = types.TextContextParams(id="text1")
    self.dummy_text_batch = [
        types.Text(
            text="the sound of a dog barking",
            context=self.text_context
        )
    ]

  def test_clap_encoder_factory(self):
    encoder_instance = clap_encoder.ClapEncoder(model_path=self.model_path)

    self.assertIsInstance(encoder_instance, encoder.CollectionEncoder)
    encoder_map = encoder_instance._encoder_by_input_type
    self.assertIn(types.Sound, encoder_map)
    self.assertIn(types.Text, encoder_map)
    self.assertIsInstance(
        encoder_map[types.Sound],
        clap_encoder._CLAPAudioEncoder
    )
    self.assertIsInstance(
        encoder_map[types.Text],
        clap_encoder._CLAPTextEncoder
    )

  @mock.patch(
      "mseb.encoders.clap_encoder.transformers.ClapProcessor.from_pretrained"
  )
  @mock.patch(
      "mseb.encoders.clap_encoder.transformers.ClapModel.from_pretrained"
  )
  def test_audio_encoder_setup(self, mock_model_load, mock_processor_load):
    audio_encoder = clap_encoder._CLAPAudioEncoder(model_path=self.model_path)
    audio_encoder.setup()

    mock_processor_load.assert_called_once_with(self.model_path)
    mock_model_load.assert_called_once_with(self.model_path)
    self.assertIsNotNone(audio_encoder.model)
    self.assertIsNotNone(audio_encoder.processor)

  @mock.patch(
      "mseb.encoders.clap_encoder.transformers.ClapProcessor.from_pretrained"
  )
  @mock.patch(
      "mseb.encoders.clap_encoder.transformers.ClapModel.from_pretrained"
  )
  def test_audio_encoder_encode(self, mock_model_load, mock_processor_load):
    mock_processor = mock.Mock()
    mock_model = mock.Mock()
    mock_processor_load.return_value = mock_processor
    mock_model_load.return_value = mock_model
    # Mock the return value of the processor call
    mock_processor.return_value = {"input_features": torch.randn(1, 1, 1024)}
    # Mock the return value of the model's audio feature extraction
    dummy_embedding = torch.randn(1, self.dummy_embedding_dim)
    mock_model.get_audio_features.return_value = dummy_embedding

    audio_encoder = clap_encoder._CLAPAudioEncoder(model_path=self.model_path)
    audio_encoder.setup()
    embeddings = audio_encoder.encode(self.dummy_sound_batch)

    # Check that the processor was called correctly
    mock_processor.assert_called_once()
    self.assertEqual(mock_processor.call_args.kwargs["sampling_rate"], [48000])

    # Check that the model was called
    mock_model.get_audio_features.assert_called_once()
    self.assertLen(embeddings, 1)
    sound_embedding = cast(types.SoundEmbedding, embeddings[0])
    self.assertIsInstance(sound_embedding, types.SoundEmbedding)
    self.assertEqual(
        sound_embedding.embedding.shape,
        (1, self.dummy_embedding_dim)
    )
    self.assertEqual(sound_embedding.context.id, "sound1")

  @mock.patch(
      "mseb.encoders.clap_encoder.transformers.ClapProcessor.from_pretrained"
  )
  @mock.patch(
      "mseb.encoders.clap_encoder.transformers.ClapModel.from_pretrained"
  )
  def test_text_encoder_encode(self, mock_model_load, mock_processor_load):
    mock_processor = mock.Mock()
    mock_model = mock.Mock()
    mock_processor_load.return_value = mock_processor
    mock_model_load.return_value = mock_model
    # Mock the return value of the processor call
    mock_processor.return_value = {"input_ids": torch.randint(0, 100, (1, 77))}
    # Mock the return value of the model's text feature extraction
    dummy_embedding = torch.randn(1, self.dummy_embedding_dim)
    mock_model.get_text_features.return_value = dummy_embedding

    text_encoder = clap_encoder._CLAPTextEncoder(model_path=self.model_path)
    text_encoder.setup()
    embeddings = text_encoder.encode(self.dummy_text_batch)
    # Check that the processor was called correctly
    mock_processor.assert_called_once()
    self.assertEqual(
        mock_processor.call_args.kwargs["text"],
        ["the sound of a dog barking"]
    )

    # Check that the model was called
    mock_model.get_text_features.assert_called_once()
    # Check the output format
    self.assertLen(embeddings, 1)
    text_embedding = cast(types.TextEmbedding, embeddings[0])
    self.assertIsInstance(text_embedding, types.TextEmbedding)
    self.assertEqual(
        text_embedding.embedding.shape,
        (1, self.dummy_embedding_dim)
    )
    self.assertEqual(text_embedding.context.id, "text1")


if __name__ == "__main__":
  absltest.main()
