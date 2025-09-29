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

import logging
from typing import Sequence
from unittest import mock

from absl.testing import absltest
from mseb import encoder
from mseb import types
import numpy as np


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

  def __init__(self):
    super().__init__()
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


if __name__ == "__main__":
  absltest.main()
