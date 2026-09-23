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

import logging
from typing import Sequence
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from mseb import encoder
from mseb import types
import numpy as np
import numpy.testing as npt


logger = logging.getLogger(__name__)


class MockMultiModalEncoder(encoder.MultiModalEncoder):
  """A concrete implementation of MultiModalEncoder for testing purposes."""

  def _check_input_types(
      self, batch: Sequence[types.MultiModalObject]
  ) -> None:
    pass

  def _setup(self):
    pass

  def _encode(
      self, sound_batch: Sequence[types.Sound]
  ) -> Sequence[types.SoundEmbedding]:
    return []

  def __init__(self, use_magic_mock=True):
    super().__init__()
    if use_magic_mock:
      self._setup = mock.MagicMock(side_effect=lambda: None)
      self._encode = mock.MagicMock(
          return_value=[
              types.SoundEmbedding(
                  embedding=np.zeros((10, 8)),
                  timestamps=np.zeros((10, 2)),
                  context=types.SoundContextParams(
                      id="test", sample_rate=16000, length=10
                  ),
              )
          ]
      )


class MultiModalEncoderTest(absltest.TestCase):

  def test_initialization_is_lazy_and_does_not_call_setup(self):
    mock_encoder = MockMultiModalEncoder()
    mock_encoder._setup.assert_not_called()

  def test_setup_is_called_only_once(self):
    mock_encoder = MockMultiModalEncoder()
    mock_encoder.setup()
    mock_encoder._setup.assert_called_once()
    mock_encoder.setup()
    mock_encoder._setup.assert_called_once()

  def test_encode_does_not_trigger_setup(self):
    mock_encoder = MockMultiModalEncoder()
    sound = types.Sound(
        waveform=np.array([1.0, 2.0, 3.0, 4.0]),
        context=types.SoundContextParams(sample_rate=2, length=4, id="test"),
    )

    mock_encoder.encode([sound])
    mock_encoder._setup.assert_not_called()

  def test_final_decorator_prevents_override_in_static_analysis(self):
    class BadMultiModalEncoder(encoder.MultiModalEncoder):

      def encode(
          self, sound_batch: Sequence[types.Sound]
      ) -> Sequence[types.SoundEmbedding]:
        return []

      def _check_input_types(self):
        pass

      def _setup(self):
        pass

      def _encode(
          self, sound_batch: Sequence[types.Sound]
      ) -> Sequence[types.SoundEmbedding]:
        return []

    bad_encoder = BadMultiModalEncoder()
    self.assertIsNotNone(bad_encoder)

  def test_output_type(self) -> type[types.MultiModalObject]:
    mock_encoder = MockMultiModalEncoder(use_magic_mock=False)
    self.assertEqual(mock_encoder.output_type(), types.SoundEmbedding)


class CascadeEncoderTest(absltest.TestCase):

  def test_initialization(self):
    enc = encoder.CascadeEncoder(encoders=[MockMultiModalEncoder()])
    self.assertLen(enc._encoders, 1)
    self.assertIsInstance(enc._encoders[0], encoder.MultiModalEncoder)
    enc._encoders[0]._setup.assert_not_called()

  def test_setup(self):
    enc = encoder.CascadeEncoder(encoders=[MockMultiModalEncoder()])
    enc.setup()
    for e in enc._encoders:
      e._setup.assert_called_once()
    enc.setup()
    for e in enc._encoders:
      e._setup.assert_called_once()

  def test_encode(self):
    enc = encoder.CascadeEncoder(encoders=[MockMultiModalEncoder()])
    sound = types.Sound(
        waveform=np.array([1.0, 2.0, 3.0, 4.0]),
        context=types.SoundContextParams(sample_rate=2, length=4, id="test"),
    )
    sound_embeddings = enc.encode([sound])
    self.assertLen(sound_embeddings, 1)
    sound_embedding = sound_embeddings[0]
    self.assertIsInstance(sound_embedding, types.SoundEmbedding)
    self.assertEqual(sound_embedding.embedding.shape, (10, 8))

  def test_output_type(self):
    enc = encoder.CascadeEncoder(
        encoders=[MockMultiModalEncoder(use_magic_mock=False)]
    )
    self.assertEqual(enc.output_type(), types.SoundEmbedding)


class IdentityEncoderTest(absltest.TestCase):

  def test_initialization(self):
    enc = encoder.IdentityEncoder(output_type=types.TextEmbedding)
    enc.setup()
    self.assertEqual(enc._output_type, types.TextEmbedding)
    outputs = enc.encode(
        [
            types.TextEmbedding(
                embedding=np.zeros((10, 8)),
                spans=np.zeros((10, 2)),
                context=types.TextContextParams(id="test"),
            )
        ]
    )
    self.assertLen(outputs, 1)
    output = outputs[0]
    self.assertIsInstance(output, types.TextEmbedding)
    self.assertEqual(output.embedding.shape, (10, 8))
    self.assertEqual(output.spans.shape, (10, 2))
    self.assertEqual(output.context.id, "test")


class CollectionEncoderTest(absltest.TestCase):

  def test_initialization(self):
    enc = encoder.CollectionEncoder(
        encoder_by_input_type={types.Sound: MockMultiModalEncoder()}
    )
    self.assertLen(enc._encoder_by_input_type, 1)
    self.assertIn(types.Sound, enc._encoder_by_input_type)
    self.assertIsInstance(
        enc._encoder_by_input_type[types.Sound], encoder.MultiModalEncoder
    )
    enc._encoder_by_input_type[types.Sound]._setup.assert_not_called()

  def test_setup(self):
    enc = encoder.CollectionEncoder(
        encoder_by_input_type={types.Sound: MockMultiModalEncoder()}
    )
    enc.setup()
    for e in enc._encoder_by_input_type.values():
      e._setup.assert_called_once()
    enc.setup()
    for e in enc._encoder_by_input_type.values():
      e._setup.assert_called_once()

  def test_encode(self):
    enc = encoder.CollectionEncoder(
        encoder_by_input_type={types.Sound: MockMultiModalEncoder()}
    )
    sound = types.Sound(
        waveform=np.array([1.0, 2.0, 3.0, 4.0]),
        context=types.SoundContextParams(sample_rate=2, length=4, id="test"),
    )
    sound_embeddings = enc.encode([sound])
    self.assertLen(sound_embeddings, 1)
    sound_embedding = sound_embeddings[0]
    self.assertIsInstance(sound_embedding, types.SoundEmbedding)
    self.assertEqual(sound_embedding.embedding.shape, (10, 8))

  def test_encode_subclass_falls_back_to_base_class_encoder(self):
    # SoundWithTitleAndContext is not registered, so it must resolve to the
    # encoder registered for its base class, Sound.
    base_encoder = MockMultiModalEncoder()
    enc = encoder.CollectionEncoder(
        encoder_by_input_type={types.Sound: base_encoder}
    )
    sound = types.SoundWithTitleAndContext(
        waveform=np.array([1.0, 2.0, 3.0, 4.0]),
        context=types.SoundContextParams(sample_rate=2, length=4, id="test"),
        title_text="title",
        context_text="context",
    )
    sound_embeddings = enc.encode([sound])
    self.assertLen(sound_embeddings, 1)
    self.assertIsInstance(sound_embeddings[0], types.SoundEmbedding)

  def test_encoder_for_prefers_exact_match_over_base_class(self):
    base_encoder = MockMultiModalEncoder()
    exact_encoder = MockMultiModalEncoder()
    enc = encoder.CollectionEncoder(
        encoder_by_input_type={
            types.Sound: base_encoder,
            types.SoundWithTitleAndContext: exact_encoder,
        }
    )
    self.assertIs(
        enc._encoder_for(types.SoundWithTitleAndContext), exact_encoder
    )
    self.assertIs(enc._encoder_for(types.Sound), base_encoder)

  def test_encoder_for_unregistered_type_raises(self):
    enc = encoder.CollectionEncoder(
        encoder_by_input_type={types.Sound: MockMultiModalEncoder()}
    )
    with self.assertRaisesRegex(KeyError, "No encoder for input type Text"):
      enc._encoder_for(types.Text)

  def _collection_encoder(self):
    return encoder.CollectionEncoder(
        encoder_by_input_type={
            types.Sound: MockMultiModalEncoder(),
            types.Text: MockMultiModalEncoder(),
        }
    )

  def _sound(self, id_="test"):
    return types.Sound(
        waveform=np.array([1.0, 2.0, 3.0, 4.0]),
        context=types.SoundContextParams(sample_rate=2, length=4, id=id_),
    )

  def _sound_with_title_and_context(self):
    return types.SoundWithTitleAndContext(
        waveform=np.array([1.0, 2.0, 3.0, 4.0]),
        context=types.SoundContextParams(sample_rate=2, length=4, id="test"),
        title_text="title",
        context_text="context",
    )

  def test_check_input_types_accepts_registered_type(self):
    enc = self._collection_encoder()
    enc._check_input_types([self._sound()])  # Should not raise.

  def test_check_input_types_accepts_homogeneous_batch(self):
    enc = self._collection_encoder()
    enc._check_input_types([self._sound("a"), self._sound("b")])

  def test_check_input_types_accepts_subclass_of_registered_type(self):
    # Must agree with _encode, which resolves subclasses to the encoder of
    # their nearest registered base class.
    enc = self._collection_encoder()
    enc._check_input_types([self._sound_with_title_and_context()])

  def test_check_input_types_rejects_unregistered_type(self):
    enc = encoder.CollectionEncoder(
        encoder_by_input_type={types.Sound: MockMultiModalEncoder()}
    )
    text = types.Text(text="hello", context=types.TextContextParams(id="test"))
    with self.assertRaisesRegex(
        KeyError,
        "No encoder for input type Text; registered input types are"
        r" \('Sound',\)",
    ):
      enc._check_input_types([text])

  def test_check_input_types_rejects_mixed_batch(self):
    enc = self._collection_encoder()
    text = types.Text(text="hello", context=types.TextContextParams(id="test"))
    with self.assertRaisesRegex(ValueError, "all inputs of the same type"):
      enc._check_input_types([self._sound(), text])

  def test_check_input_types_rejects_mixed_batch_in_either_order(self):
    # The homogeneity test keys off batch[0], so guard both orderings.
    enc = self._collection_encoder()
    text = types.Text(text="hello", context=types.TextContextParams(id="test"))
    with self.assertRaisesRegex(ValueError, "all inputs of the same type"):
      enc._check_input_types([text, self._sound()])

  def test_check_input_types_rejects_base_class_after_subclass(self):
    # isinstance() is asymmetric: a plain Sound is not an instance of
    # SoundWithTitleAndContext, so this ordering is heterogeneous.
    enc = self._collection_encoder()
    with self.assertRaisesRegex(ValueError, "all inputs of the same type"):
      enc._check_input_types(
          [self._sound_with_title_and_context(), self._sound()]
      )

  def test_check_input_types_empty_batch_does_not_raise(self):
    enc = self._collection_encoder()
    enc._check_input_types([])

  def test_encode_rejects_unregistered_type(self):
    enc = encoder.CollectionEncoder(
        encoder_by_input_type={types.Sound: MockMultiModalEncoder()}
    )
    text = types.Text(text="hello", context=types.TextContextParams(id="test"))
    with self.assertRaisesRegex(
        KeyError,
        "No encoder for input type Text; registered input types are"
        r" \('Sound',\)",
    ):
      enc.encode([text])


class ResampleSoundTest(parameterized.TestCase):

  def test_resample_sound_no_resampling_float(self):
    waveform = np.linspace(-1.0, 1.0, 16000, dtype=np.float32)
    context = types.SoundContextParams(
        id="test", sample_rate=16000, length=16000
    )
    sound = types.Sound(waveform=waveform, context=context)
    resampled_sound = encoder.resample_sound(sound, 16000)
    self.assertIs(resampled_sound, sound)

  @parameterized.named_parameters(
      dict(
          testcase_name="int16_to_float32",
          dtype=np.int16,
          target_dtype=np.float32,
          waveform_array=[-32768, 0, 32767],
          expected_waveform=[-1.0, 0.0, 32767.0 / 32768.0],
      ),
      dict(
          testcase_name="int8_to_float32",
          dtype=np.int8,
          target_dtype=np.float32,
          waveform_array=[-128, 0, 127],
          expected_waveform=[-1.0, 0.0, 127.0 / 128.0],
      ),
      dict(
          testcase_name="int32_to_float32",
          dtype=np.int32,
          target_dtype=np.float32,
          waveform_array=[-2147483648, 0, 2147483647],
          expected_waveform=[-1.0, 0.0, 2147483647.0 / 2147483648.0],
      ),
      dict(
          testcase_name="int16_to_int16",
          dtype=np.int16,
          target_dtype=np.int16,
          waveform_array=[-32768, 0, 32767],
          expected_waveform=[-32768, 0, 32767],
      ),
      dict(
          testcase_name="int8_to_int8",
          dtype=np.int8,
          target_dtype=np.int8,
          waveform_array=[-128, 0, 127],
          expected_waveform=[-128, 0, 127],
      ),
      dict(
          testcase_name="int32_to_int32",
          dtype=np.int32,
          target_dtype=np.int32,
          waveform_array=[-2147483648, 0, 2147483647],
          expected_waveform=[-2147483648, 0, 2147483647],
      ),
      dict(
          testcase_name="float64_to_float64",
          dtype=np.float64,
          target_dtype=np.float64,
          waveform_array=[-1.0, 0.0, 1.0],
          expected_waveform=[-1.0, 0.0, 1.0],
      ),
  )
  def test_resample_sound_no_resampling_dtype(
      self, dtype, target_dtype, waveform_array, expected_waveform
  ):
    waveform = np.array(waveform_array, dtype=dtype)
    context = types.SoundContextParams(id="test", sample_rate=16000, length=3)
    sound = types.Sound(waveform=waveform, context=context)
    resampled_sound = encoder.resample_sound(
        sound, 16000, target_dtype=target_dtype
    )
    if dtype == target_dtype:
      self.assertIs(resampled_sound, sound)
    else:
      self.assertIsNot(resampled_sound, sound)
    self.assertEqual(resampled_sound.waveform.dtype, target_dtype)
    self.assertEqual(resampled_sound.context.sample_rate, 16000)
    npt.assert_allclose(resampled_sound.waveform, expected_waveform, atol=1e-4)

  @parameterized.named_parameters(
      dict(
          testcase_name="float32",
          dtype=np.float32,
          min_val=-1.0,
          max_val=1.0,
      ),
      dict(
          testcase_name="int16",
          dtype=np.int16,
          min_val=-32768,
          max_val=32767,
      ),
      dict(
          testcase_name="int8",
          dtype=np.int8,
          min_val=-128,
          max_val=127,
      ),
      dict(
          testcase_name="int32",
          dtype=np.int32,
          min_val=-2147483648,
          max_val=2147483647,
      ),
      dict(
          testcase_name="float64",
          dtype=np.float64,
          min_val=-1.0,
          max_val=1.0,
      ),
  )
  def test_resample_sound_downsample(self, dtype, min_val, max_val):
    waveform = np.linspace(min_val, max_val, 16000).astype(dtype)
    context = types.SoundContextParams(
        id="test", sample_rate=16000, length=16000
    )
    sound = types.Sound(waveform=waveform, context=context)
    resampled_sound = encoder.resample_sound(sound, 8000)
    self.assertEqual(resampled_sound.context.sample_rate, 8000)
    self.assertEqual(resampled_sound.waveform.shape[0], 8000)
    self.assertEqual(resampled_sound.waveform.dtype, np.float32)


if __name__ == "__main__":
  absltest.main()
