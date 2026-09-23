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

from collections.abc import Sequence
import inspect
from typing import Any, cast
from unittest import mock

from absl.testing import absltest
from mseb import encoder
from mseb import types
from mseb.encoders import clap_encoder
import numpy as np
import pytest
import torch
import transformers
from transformers import modeling_outputs


def _pooled(
    pooler_output: torch.Tensor,
) -> modeling_outputs.BaseModelOutputWithPooling:
  """Returns `pooler_output` wrapped the way transformers v5 CLAP returns it.

  In transformers v4, `ClapModel.get_audio_features()` and `get_text_features()`
  returned a bare `torch.FloatTensor`. In v5 they return a
  `BaseModelOutputWithPooling` whose `pooler_output` holds the projected,
  normalized embedding, so the encoder has to unwrap it.

  Tests build the real v5 output object rather than a `mock.Mock` on purpose: a
  `Mock` answers *any* attribute access, so it would happily mock through both
  the presence and the absence of the `.pooler_output` unwrapping.

  `last_hidden_state` is deliberately filled with zeros of the same shape as
  `pooler_output`, so a regression that reads the wrong field fails on values
  rather than passing by accident.

  Args:
    pooler_output: The embedding tensor the model should appear to return.

  Returns:
    A `BaseModelOutputWithPooling` carrying `pooler_output`.
  """
  return modeling_outputs.BaseModelOutputWithPooling(
      last_hidden_state=torch.zeros_like(pooler_output),
      pooler_output=pooler_output,
  )


# Optional due to segmentation fault when run under pytest.
@pytest.mark.optional
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
            waveform=np.random.randn(48000).astype(np.float32),  # pyrefly: ignore[bad-argument-type]
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
    mock_processor.feature_extractor.sampling_rate = 48000
    mock_processor_load.return_value = mock_processor
    mock_model_load.return_value = mock_model
    # Mock the return value of the processor call
    mock_processor.return_value = {"input_features": torch.randn(1, 1, 1024)}
    # Mock the return value of the model's audio feature extraction
    dummy_embedding = torch.randn(1, self.dummy_embedding_dim)
    mock_model.get_audio_features.return_value = _pooled(dummy_embedding)

    audio_encoder = clap_encoder._CLAPAudioEncoder(model_path=self.model_path)
    audio_encoder.setup()
    embeddings = audio_encoder.encode(self.dummy_sound_batch)

    # Check that the processor was called correctly
    mock_processor.assert_called_once()
    self.assertEqual(mock_processor.call_args.kwargs["sampling_rate"], 48000)
    # v4 spelled this kwarg `audios=`; in v5 that lands in `**kwargs`, leaving
    # `audio=None`, so the audio would silently never be processed.
    np.testing.assert_array_equal(
        mock_processor.call_args.kwargs["audio"][0],
        self.dummy_sound_batch[0].waveform,
    )

    # Check that the model was called
    mock_model.get_audio_features.assert_called_once()
    self.assertLen(embeddings, 1)
    sound_embedding = cast(types.SoundEmbedding, embeddings[0])
    self.assertIsInstance(sound_embedding, types.SoundEmbedding)
    self.assertEqual(
        sound_embedding.embedding.shape,
        (1, self.dummy_embedding_dim)
    )
    # The embedding has to come from `.pooler_output`, not `last_hidden_state`,
    # which `_pooled` fills with zeros.
    np.testing.assert_allclose(
        sound_embedding.embedding, dummy_embedding.numpy()
    )
    self.assertEqual(sound_embedding.context.id, "sound1")

  @mock.patch("mseb.encoders.clap_encoder.encoder.resample_sound")
  @mock.patch(
      "mseb.encoders.clap_encoder.transformers.ClapProcessor.from_pretrained"
  )
  @mock.patch(
      "mseb.encoders.clap_encoder.transformers.ClapModel.from_pretrained"
  )
  def test_audio_encoder_resamples_input(
      self,
      mock_model_load,
      mock_processor_load,
      mock_resample_sound
  ):
    mock_processor = mock.Mock()
    mock_processor.feature_extractor.sampling_rate = 48000
    mock_processor.return_value = {"input_features": torch.randn(1, 1, 1024)}
    mock_processor_load.return_value = mock_processor
    mock_model = mock.Mock()
    mock_model.get_audio_features.return_value = _pooled(
        torch.randn(1, self.dummy_embedding_dim)
    )
    mock_model_load.return_value = mock_model
    # The resample function should return a sound object
    mock_resample_sound.return_value = types.Sound(
        waveform=np.random.randn(48000),  # pyrefly: ignore[bad-argument-type]
        context=types.SoundContextParams(
            id="resampled",
            sample_rate=48000,
            length=48000
        )
    )
    bad_sr_context = types.SoundContextParams(
        id="sound_44k",
        sample_rate=44100,
        length=44100
    )
    bad_sr_batch = [types.Sound(
        waveform=np.random.randn(44100),  # pyrefly: ignore[bad-argument-type]
        context=bad_sr_context
    )]
    audio_encoder = clap_encoder._CLAPAudioEncoder(
        model_path=self.model_path
    )
    audio_encoder.setup()
    audio_encoder.encode(bad_sr_batch)
    mock_resample_sound.assert_called_once()
    self.assertEqual(
        mock_resample_sound.call_args.kwargs["target_sr"],
        48000
    )
    mock_processor.assert_called_once()
    self.assertEqual(
        mock_processor.call_args.kwargs["sampling_rate"],
        48000
    )

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
    mock_model.get_text_features.return_value = _pooled(dummy_embedding)

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
    # The embedding has to come from `.pooler_output`, not `last_hidden_state`,
    # which `_pooled` fills with zeros.
    np.testing.assert_allclose(
        text_embedding.embedding, dummy_embedding.numpy()
    )
    self.assertEqual(text_embedding.context.id, "text1")

  # --- Input-type dispatch. No setup() needed, so no model loading at all. ---

  def test_audio_encoder_rejects_text_batch(self):
    audio_encoder = clap_encoder._CLAPAudioEncoder(model_path=self.model_path)

    with self.assertRaises(ValueError):
      audio_encoder._check_input_types(self.dummy_text_batch)

  def test_text_encoder_rejects_sound_batch(self):
    text_encoder = clap_encoder._CLAPTextEncoder(model_path=self.model_path)

    with self.assertRaises(ValueError):
      text_encoder._check_input_types(self.dummy_sound_batch)

  def test_audio_encoder_accepts_sound_subclass(self):
    """`SoundWithTitleAndContext` is what the SVQ retrieval tasks emit."""
    audio_encoder = clap_encoder._CLAPAudioEncoder(model_path=self.model_path)
    batch = [
        types.SoundWithTitleAndContext(
            waveform=np.zeros(48000, dtype=np.float32),  # pyrefly: ignore[bad-argument-type]
            context=self.sound_context,
        )
    ]

    audio_encoder._check_input_types(batch)  # Should not raise.


class _TinyClapProcessor:
  """Stand-in for `ClapProcessor` shaped for `ClapRealModelTest`'s tiny model.

  A real `ClapProcessor` cannot be used here: it needs `from_pretrained` for the
  Roberta tokenizer vocabulary, and a pretrained feature extractor emits 64-mel,
  1024-frame spectrograms that the tiny audio tower cannot consume.

  Unlike a `mock.Mock`, this stub only accepts the keywords the real
  `ClapProcessor.__call__` accepts, so the encoder calling it with v4's
  `audios=` keyword would raise `TypeError` here rather than pass silently.
  """

  def __init__(
      self,
      sampling_rate: int,
      time_length: int,
      num_mel_bins: int,
      vocab_size: int,
      seq_length: int,
  ):
    # The real feature extractor, which only supplies `sampling_rate` to the
    # encoder, and needs no pretrained files.
    self.feature_extractor = transformers.ClapFeatureExtractor(
        sampling_rate=sampling_rate
    )
    self._time_length = time_length
    self._num_mel_bins = num_mel_bins
    self._vocab_size = vocab_size
    self._seq_length = seq_length
    self.call_kwargs: dict[str, Any] = {}

  def __call__(
      self,
      text: Sequence[str] | None = None,
      audio: Sequence[np.ndarray] | None = None,
      sampling_rate: int | None = None,
      return_tensors: str | None = None,
      padding: bool | None = None,
  ) -> dict[str, torch.Tensor]:
    self.call_kwargs = dict(
        text=text,
        audio=audio,
        sampling_rate=sampling_rate,
        return_tensors=return_tensors,
        padding=padding,
    )
    if audio is not None:
      return {
          "input_features": torch.randn(
              len(audio), 1, self._time_length, self._num_mel_bins
          )
      }
    assert text is not None
    return {
        "input_ids": torch.randint(
            0, self._vocab_size, (len(text), self._seq_length)
        )
    }


# Optional due to segmentation fault when run under pytest.
@pytest.mark.optional
class ClapRealModelTest(absltest.TestCase):
  """Runs the CLAP encoders against a real `ClapModel`.

  The model is not mocked here. `ClapEncoderTest` above mocks `transformers`,
  which means it *pins* the v5 API shape rather than *checking* it: those tests
  would stay green if a future transformers upgrade changed the return type of
  `get_audio_features()`, while production silently broke. These tests run the
  encoders against the real library, so such an upgrade fails here instead.

  The model is built from a tiny randomly-initialized config rather than
  `from_pretrained`, so there is no weight download and no network access. It is
  built once per class because, even tiny, it is the expensive part of these
  tests.

  Note that `ClapAudioConfig` is a `@strict` dataclass, so it rejects unknown
  fields; the vendored upstream `test_modeling_clap.py` is stale with respect to
  it and its config kwargs no longer apply.
  """

  SAMPLE_RATE = 48000
  # `ClapAudioModel.reshape_mel2img` derives freq_ratio = spec_size //
  # num_mel_bins = 4, and requires input_features of shape
  # (batch, 1, time <= spec_size * freq_ratio, mels <= spec_size // freq_ratio),
  # i.e. (batch, 1, 256, 16) here.
  SPEC_SIZE = 64
  NUM_MEL_BINS = 16
  TIME_LENGTH = 256
  HIDDEN_SIZE = 32
  PROJECTION_DIM = 64
  VOCAB_SIZE = 99
  SEQ_LENGTH = 7

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    audio_config = transformers.ClapAudioConfig(
        spec_size=cls.SPEC_SIZE,
        num_mel_bins=cls.NUM_MEL_BINS,
        patch_size=4,
        patch_stride=(4, 4),
        window_size=4,
        hidden_size=cls.HIDDEN_SIZE,
        depths=(2, 2),
        num_attention_heads=(2, 2),
        num_hidden_layers=2,
        patch_embeds_hidden_size=16,
        enable_fusion=False,
    )
    text_config = transformers.ClapTextConfig(
        vocab_size=cls.VOCAB_SIZE,
        hidden_size=cls.HIDDEN_SIZE,
        projection_dim=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        max_position_embeddings=512,
    )
    config = transformers.ClapConfig(
        text_config=text_config.to_dict(),
        audio_config=audio_config.to_dict(),
        projection_dim=cls.PROJECTION_DIM,
    )
    cls.tiny_model = transformers.ClapModel(config)
    cls.tiny_model.eval()

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)
    self.processor = _TinyClapProcessor(
        sampling_rate=self.SAMPLE_RATE,
        time_length=self.TIME_LENGTH,
        num_mel_bins=self.NUM_MEL_BINS,
        vocab_size=self.VOCAB_SIZE,
        seq_length=self.SEQ_LENGTH,
    )
    self.sound_batch = [
        types.Sound(
            waveform=np.random.randn(self.SAMPLE_RATE).astype(np.float32),  # pyrefly: ignore[bad-argument-type]
            context=types.SoundContextParams(
                id=f"sound{i}",
                sample_rate=self.SAMPLE_RATE,
                length=self.SAMPLE_RATE,
            ),
        )
        for i in range(2)
    ]
    self.text_batch = [
        types.Text(text=text, context=types.TextContextParams(id=f"text{i}"))
        for i, text in enumerate(["a dog barking", "a vacuum cleaner"])
    ]

  def test_audio_encoder_encodes_with_real_model(self):
    audio_encoder = clap_encoder._CLAPAudioEncoder(
        model_path="unused", device="cpu"
    )
    # Injected instead of loaded by `setup()`, which would download weights.
    audio_encoder.model = self.tiny_model
    audio_encoder.processor = self.processor  # pyrefly: ignore[bad-assignment]

    embeddings = audio_encoder.encode(self.sound_batch)

    self.assertEqual(
        self.processor.call_kwargs["sampling_rate"], self.SAMPLE_RATE
    )
    np.testing.assert_array_equal(
        self.processor.call_kwargs["audio"][0], self.sound_batch[0].waveform
    )
    self.assertLen(embeddings, len(self.sound_batch))
    for sound, embedding in zip(self.sound_batch, embeddings):
      sound_embedding = cast(types.SoundEmbedding, embedding)
      self.assertIsInstance(sound_embedding, types.SoundEmbedding)
      self.assertEqual(
          sound_embedding.embedding.shape, (1, self.PROJECTION_DIM)
      )
      self.assertEqual(sound_embedding.context.id, sound.context.id)
      np.testing.assert_array_equal(sound_embedding.timestamps, [[0.0, 1.0]])
      # `get_audio_features` L2-normalizes `pooler_output`, so a regression that
      # read `last_hidden_state` instead would not land on the unit sphere.
      np.testing.assert_allclose(
          np.linalg.norm(sound_embedding.embedding), 1.0, rtol=1e-5
      )

  def test_text_encoder_encodes_with_real_model(self):
    text_encoder = clap_encoder._CLAPTextEncoder(
        model_path="unused", device="cpu"
    )
    # Injected instead of loaded by `setup()`, which would download weights.
    text_encoder.model = self.tiny_model
    text_encoder.processor = self.processor  # pyrefly: ignore[bad-assignment]

    embeddings = text_encoder.encode(self.text_batch)

    self.assertEqual(
        self.processor.call_kwargs["text"],
        [text.text for text in self.text_batch],
    )
    self.assertLen(embeddings, len(self.text_batch))
    for text, embedding in zip(self.text_batch, embeddings):
      text_embedding = cast(types.TextEmbedding, embedding)
      self.assertIsInstance(text_embedding, types.TextEmbedding)
      self.assertEqual(text_embedding.embedding.shape, (1, self.PROJECTION_DIM))
      self.assertEqual(text_embedding.context.id, text.context.id)
      np.testing.assert_array_equal(text_embedding.spans, [[0, len(text.text)]])
      # `get_text_features` L2-normalizes `pooler_output`, so a regression that
      # read `last_hidden_state` instead would not land on the unit sphere.
      np.testing.assert_allclose(
          np.linalg.norm(text_embedding.embedding), 1.0, rtol=1e-5
      )

  def test_clap_processor_takes_audio_and_text_keywords(self):
    """The encoders call the processor with `audio=` and `text=`.

    v4 spelled the audio keyword `audios=`. Neither the mocked tests nor
    `_TinyClapProcessor` can see a rename on the real class, so check its
    signature directly.
    """
    params = inspect.signature(transformers.ClapProcessor.__call__).parameters

    self.assertIn("audio", params)
    self.assertNotIn("audios", params)
    self.assertIn("text", params)


if __name__ == "__main__":
  absltest.main()
