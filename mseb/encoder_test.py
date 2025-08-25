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
from typing import Any
from unittest import mock

from absl.testing import absltest
from mseb import encoder
from mseb import types
import numpy as np


logger = logging.getLogger(__name__)


class MockSoundEncoder(encoder.SoundEncoder):
  """A concrete implementation of SoundEncoder for testing purposes."""

  def setup(self):
    pass

  def _encode_batch(self, waveform_batch, params_batch, **kwargs):
    pass

  def __init__(self, model_path: str, **kwargs: Any):
    super().__init__(model_path, **kwargs)
    self.setup = mock.MagicMock(side_effect=self._setup_impl)
    self._encode_batch = mock.MagicMock(
        return_value=[(np.zeros((10, 8)), np.zeros((10, 2)))]
    )

  def _setup_impl(self):
    """The actual implementation for the mocked setup method."""
    self._model_loaded = True


class FaultySetupEncoder(encoder.SoundEncoder):
  """An encoder that "forgets" to set the _model_loaded flag in setup."""

  def setup(self):
    logger.info("Faulty setup was called, but did not set the flag.")

  def _encode_batch(self, waveform_batch, params_batch, **kwargs):
    return [(np.array([]), np.array([]))]


class SoundEncoderTest(absltest.TestCase):

  def test_initialization_is_lazy_and_does_not_call_setup(self):
    mock_encoder = MockSoundEncoder("path/to/model")
    mock_encoder.setup.assert_not_called()

  def test_encode_triggers_setup_exactly_once(self):
    mock_encoder = MockSoundEncoder("path/to/model")
    sound = types.Sound(
        waveform=np.array([1.0, 2.0, 3.0, 4.0]),
        context=types.SoundContextParams(sample_rate=2, length=4, id="test"),
    )

    mock_encoder.encode(sound)
    mock_encoder.setup.assert_called_once()

    sound2 = types.Sound(
        waveform=np.array([5.0, 6.0, 7.0, 8.0]),
        context=types.SoundContextParams(sample_rate=2, length=4, id="test"),
    )
    mock_encoder.encode(sound2)
    mock_encoder.setup.assert_called_once()

  def test_encode_batch_triggers_setup_exactly_once(self):
    mock_encoder = MockSoundEncoder("path/to/model")
    params = types.SoundContextParams(sample_rate=2, length=4, id="test")
    batch = [
        types.Sound(waveform=np.array([1.0, 2.0, 3.0, 4.0]), context=params),
        types.Sound(waveform=np.array([5.0, 6.0, 7.0, 8.0]), context=params),
    ]
    mock_encoder.encode_batch(batch)
    mock_encoder.setup.assert_called_once()
    mock_encoder.encode_batch(batch)
    mock_encoder.setup.assert_called_once()

  def test_encode_batch_delegates_to_encode_batch_with_correct_args(self):
    mock_encoder = MockSoundEncoder("path/to/model")
    params = types.SoundContextParams(sample_rate=2, length=4, id="test")
    sound_batch = [
        types.Sound(waveform=np.array([1.0, 2.0, 4.0, 8.0]), context=params),
        types.Sound(waveform=np.array([2.0, 5.0, 3.0, 7.0]), context=params),
        types.Sound(waveform=np.array([3.0, 7.0, 8.0, 9.0]), context=params),
    ]
    mock_encoder.encode_batch(sound_batch, runtime_kwarg="hello")
    mock_encoder._encode_batch.assert_called_once()
    args, kwargs = mock_encoder._encode_batch.call_args
    self.assertEqual(args[0], sound_batch)
    self.assertEqual(kwargs, {"runtime_kwarg": "hello"})

  def test_default_encode_calls_encode_batch_with_single_item(self):
    mock_encoder = MockSoundEncoder("path/to/model")
    mock_encoder.encode_batch = mock.MagicMock()
    sound = types.Sound(
        waveform=np.array([1.0, 2.0, 3.0, 4.0]),
        context=types.SoundContextParams(sample_rate=2, length=4, id="test"),
    )

    mock_encoder.encode(sound)
    mock_encoder.encode_batch.assert_called_once_with([sound], **{})

  def test_faulty_setup_raises_runtime_error(self):
    mock_encoder = FaultySetupEncoder("faulty/path")
    sound = types.Sound(
        waveform=np.array([1.0, 2.0, 4.0, 8.0]),
        context=types.SoundContextParams(sample_rate=2, length=4, id="test"),
    )
    with self.assertRaises(RuntimeError):
      mock_encoder.encode(sound)

  def test_init_kwargs_are_stored_for_later_use(self):
    mock_encoder = MockSoundEncoder(
        "path/to/model", special_param=123, other_param="abc")
    self.assertIn("special_param", mock_encoder._kwargs)
    self.assertEqual(mock_encoder._kwargs["special_param"], 123)

  def test_final_decorator_prevents_override_in_static_analysis(self):
    class BadSoundEncoder(encoder.SoundEncoder):
      def encode_batch(self, waveform_batch, params_batch, **kwargs):
        return [(np.array([-1]), np.array([-1]))]

      def setup(self):
        self._model_loaded = True

      def _encode_batch(self, waveform_batch, params_batch, **kwargs):
        return [(np.array([0]), np.array([0]))]

    bad_encoder = BadSoundEncoder("path")
    self.assertIsNotNone(bad_encoder)


class MockTextEncoder(encoder.TextEncoder):
  """A concrete implementation of TextEncoder for testing purposes."""

  def setup(self):
    pass

  def _encode_batch(self, text_batch, **kwargs):
    pass

  def __init__(self, **kwargs: Any):
    super().__init__(**kwargs)
    self.setup = mock.MagicMock(side_effect=self._setup_impl)
    self._encode_batch = mock.MagicMock(
        return_value=[
            types.TextEmbeddings(
                embeddings=np.zeros((10, 8)),
                spans=np.zeros((10, 2)),
                context=types.TextContextParams(id="id"),
            )
        ]
    )

  def _setup_impl(self):
    """The actual implementation for the mocked setup method."""
    self._model_loaded = True


class FaultySetupTextEncoder(encoder.TextEncoder):
  """An encoder that "forgets" to set the _model_loaded flag in setup."""

  def setup(self):
    logger.info("Faulty setup was called, but did not set the flag.")

  def _encode_batch(self, text_batch, **kwargs):
    return [
        types.TextEmbeddings(
            embeddings=np.array([]),
            spans=np.array([]),
            context=types.TextContextParams(id="id"),
        )
    ]


class TextEncoderTest(absltest.TestCase):

  def test_initialization_is_lazy_and_does_not_call_setup(self):
    mock_encoder = MockTextEncoder()
    mock_encoder.setup.assert_not_called()

  def test_encode_triggers_setup_exactly_once(self):
    mock_encoder = MockTextEncoder()

    mock_encoder.encode(
        types.Text(
            text="This is a text.", context=types.TextContextParams(id="id")
        )
    )
    mock_encoder.setup.assert_called_once()

    mock_encoder.encode_batch([
        types.Text(
            text="This is another text.",
            context=types.TextContextParams(id="id"),
        )
    ])
    mock_encoder.setup.assert_called_once()

  def test_encode_batch_triggers_setup_exactly_once(self):
    mock_encoder = MockTextEncoder()
    batch = [
        types.Text(
            text="This is a text.", context=types.TextContextParams(id="id1")
        ),
        types.Text(
            text="This is another text.",
            context=types.TextContextParams(id="id2"),
        ),
    ]
    mock_encoder.encode_batch(batch)
    mock_encoder.setup.assert_called_once()

  def test_encode_batch_delegates_to_encode_batch_with_correct_args(self):
    mock_encoder = MockTextEncoder()
    text_batch = [
        types.Text(
            text="This is a text.", context=types.TextContextParams(id="id1")
        ),
        types.Text(
            text="This is another text.",
            context=types.TextContextParams(id="id2"),
        ),
        types.Text(
            text="This is the third text.",
            context=types.TextContextParams(id="id3"),
        ),
    ]
    mock_encoder.encode_batch(text_batch, runtime_kwarg="hello")
    mock_encoder._encode_batch.assert_called_once_with(
        text_batch, runtime_kwarg="hello"
    )

  def test_default_encode_calls_encode_batch_with_single_item(self):
    mock_encoder = MockTextEncoder()
    mock_encoder.encode_batch = mock.MagicMock()
    text = types.Text(
        text="This is a text.", context=types.TextContextParams(id="test")
    )

    mock_encoder.encode(text)
    mock_encoder.encode_batch.assert_called_once_with([text], **{})

  def test_faulty_setup_raises_runtime_error(self):
    mock_encoder = FaultySetupTextEncoder()
    with self.assertRaises(RuntimeError):
      mock_encoder.encode(
          types.Text(
              text="This is a text.", context=types.TextContextParams(id="id")
          )
      )

  def test_init_kwargs_are_stored_for_later_use(self):
    mock_encoder = MockTextEncoder(special_param=123, other_param="abc")
    self.assertIn("special_param", mock_encoder._kwargs)
    self.assertEqual(mock_encoder._kwargs["special_param"], 123)

  def test_final_decorator_prevents_override_in_static_analysis(self):
    class BadTextEncoder(encoder.TextEncoder):

      def encode_batch(self, text_batch, **kwargs):
        return types.TextEmbeddings(
            embeddings=np.array([-1]),
            spans=np.array([0]),
            context=types.TextContextParams(id=""),
        )

      def setup(self):
        self._model_loaded = True

      def _encode_batch(self, text_batch, params_batch, **kwargs):
        return types.TextEmbeddings(
            embeddings=np.array([-1]),
            spans=np.array([0]),
            context=types.TextContextParams(id=""),
        )

    bad_encoder = BadTextEncoder()
    self.assertIsNotNone(bad_encoder)


if __name__ == "__main__":
  absltest.main()
