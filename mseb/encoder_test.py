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

  def _encode(self, audio, params, **kwargs):
    pass

  def __init__(self, model_path: str, **kwargs: Any):
    super().__init__(model_path, **kwargs)
    self.setup = mock.MagicMock(side_effect=self._setup_impl)
    self._encode = mock.MagicMock(
        return_value=(np.zeros((10, 8)), np.zeros((10, 2)))
    )

  def _setup_impl(self):
    """The actual implementation for the mocked setup method."""
    self._model_loaded = True


class FaultySetupEncoder(encoder.SoundEncoder):
  """An encoder that "forgets" to set the _model_loaded flag in setup."""

  def setup(self):
    logger.info("Faulty setup was called, but did not set the flag.")

  def _encode(self, waveform, params, **kwargs):
    return (np.array([]), np.array([]))


class SoundEncoderTest(absltest.TestCase):

  def test_initialization_is_lazy_and_does_not_call_setup(self):
    mock_encoder = MockSoundEncoder("path/to/model")
    mock_encoder.setup.assert_not_called()

  def test_encode_triggers_setup_exactly_once(self):
    mock_encoder = MockSoundEncoder("path/to/model")
    params = types.SoundContextParams(
        sample_rate=2,
        length=4,
    )

    mock_encoder.encode([1.0, 2.0, 3.0, 4.0], params)
    mock_encoder.setup.assert_called_once()

    mock_encoder.encode([5.0, 6.0, 7.0, 8.0], params)
    mock_encoder.setup.assert_called_once()

  def test_encode_batch_triggers_setup_exactly_once(self):
    mock_encoder = MockSoundEncoder("path/to/model")
    params = types.SoundContextParams(
        sample_rate=2,
        length=4,
    )
    batch = [([1.0, 2.0, 3.0, 4.0], params),
             ([5.0, 6.0, 7.0, 8.0], params)]
    mock_encoder.encode_batch([a[0] for a in batch], [a[1] for a in batch])
    mock_encoder.setup.assert_called_once()
    mock_encoder.encode_batch([a[0] for a in batch], [a[1] for a in batch])
    mock_encoder.setup.assert_called_once()

  def test_encode_delegates_to_encode_with_correct_args(self):
    mock_encoder = MockSoundEncoder("path/to/model")
    params = types.SoundContextParams(
        sample_rate=2,
        length=4,
        language="en",
    )
    waveform = [1.0, 2.0, 3.0, 4.0]
    mock_encoder.encode(waveform, params, runtime_kwarg="hello")
    mock_encoder._encode.assert_called_once_with(
        waveform, params, runtime_kwarg="hello"
    )

  def test_default_encode_batch_calls_encode_for_each_item(self):
    mock_encoder = MockSoundEncoder("path/to/model")
    mock_encoder.encode = mock.MagicMock()
    params = types.SoundContextParams(
        sample_rate=2,
        length=4,
    )
    waveform_batch = [
        [1.0, 2.0, 4.0, 8.0],
        [2.0, 5.0, 3.0, 7.0],
        [3.0, 7.0, 8.0, 9.0]
    ]
    params_batch = [params] * 3

    mock_encoder.encode_batch(waveform_batch, params_batch)
    self.assertEqual(mock_encoder.encode.call_count, 3)
    mock_encoder.encode.assert_any_call([1.0, 2.0, 4.0, 8.0], params)
    mock_encoder.encode.assert_any_call([3.0, 7.0, 8.0, 9.0], params)

  def test_faulty_setup_raises_runtime_error(self):
    mock_encoder = FaultySetupEncoder("faulty/path")
    params = types.SoundContextParams(
        sample_rate=2,
        length=4,
    )
    with self.assertRaises(RuntimeError):
      mock_encoder.encode([1.0, 2.0, 4.0, 8.0], params)

  def test_init_kwargs_are_stored_for_later_use(self):
    mock_encoder = MockSoundEncoder(
        "path/to/model", special_param=123, other_param="abc")
    self.assertIn("special_param", mock_encoder._kwargs)
    self.assertEqual(mock_encoder._kwargs["special_param"], 123)

  def test_final_decorator_prevents_override_in_static_analysis(self):
    class BadSoundEncoder(encoder.SoundEncoder):
      def encode(self, waveform, params, **kwargs):
        return (np.array([-1]), np.array([-1]))

      def setup(self):
        self._model_loaded = True

      def _encode(self, waveform, params, **kwargs):
        return (np.array([0]), np.array([0]))

    bad_encoder = BadSoundEncoder("path")
    self.assertIsNotNone(bad_encoder)

if __name__ == "__main__":
  absltest.main()
