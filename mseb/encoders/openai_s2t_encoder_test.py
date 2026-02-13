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

"""Tests for the OpenAI API-based encoder."""

import os
import pathlib
from unittest import mock

from absl.testing import absltest
from mseb import types
from mseb import utils
from openai.types.audio.transcription import Transcription
from openai.types.audio.transcription_verbose import TranscriptionVerbose
from openai.types.audio.transcription_word import TranscriptionWord
import pytest

s2t_encoder_lib = pytest.importorskip('mseb.encoders.openai_s2t_encoder')


class MockOpenAISpeechToTextEncoder(s2t_encoder_lib.OpenAISpeechToTextEncoder):
  """Mock OpenAISpeechToTextEncoder for testing."""

  def __init__(self, word_timestamps: bool = False):
    super().__init__(
        server_url='https://mock_server_url',
        api_key='mock_api_key',
        model_name='mock_model_name',
        word_timestamps=word_timestamps,
    )

  def _setup(self):
    self._client = mock.MagicMock()
    if self._word_timestamps:
      self._client.audio.transcriptions.create.return_value = (
          TranscriptionVerbose(
              duration=3.069999933242798,
              language='english',
              text='Roses are red. Violets are blue.',
              segments=None,
              usage=None,
              words=[
                  TranscriptionWord(
                      end=0.7599999904632568, start=0.0, word='Roses'
                  ),
                  TranscriptionWord(
                      end=1.0, start=0.7599999904632568, word='are'
                  ),
                  TranscriptionWord(
                      end=1.440000057220459, start=1.0, word='red'
                  ),
                  TranscriptionWord(
                      end=2.140000104904175,
                      start=2.0799999237060547,
                      word='Violets',
                  ),
                  TranscriptionWord(
                      end=2.359999895095825, start=2.140000104904175, word='are'
                  ),
                  TranscriptionWord(
                      end=2.5199999809265137,
                      start=2.359999895095825,
                      word='blue',
                  ),
              ],
          )
      )
    else:
      self._client.audio.transcriptions.create.return_value = Transcription(
          text='Roses are red. Violets are blue.',
          usage=None,
      )
    super()._setup()


class OpenAISpeechToTextEncoderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, 'testdata'
    )
    with open(os.path.join(testdata_path, 'roses-are.wav'), 'rb') as f:
      wav_bytes = f.read()
      samples, sample_rate = utils.wav_bytes_to_waveform(wav_bytes)
    self.test_sound = types.Sound(
        waveform=samples,
        context=types.SoundContextParams(
            id='test',
            sample_rate=sample_rate,
            length=len(samples),
            waveform_start_second=0.0,
            waveform_end_second=len(samples) / sample_rate,
        ),
    )

  def test_encode(self):
    s2t_encoder = MockOpenAISpeechToTextEncoder()
    s2t_encoder.setup()
    outputs = s2t_encoder.encode([self.test_sound])
    self.assertLen(outputs, 1)
    self.assertIsInstance(outputs[0], types.SoundEmbedding)
    self.assertEqual(
        str(outputs[0].embedding[0]), 'Roses are red. Violets are blue.'
    )
    self.assertEqual(outputs[0].timestamps.shape, (1, 2))
    self.assertEqual(outputs[0].timestamps[0, 0], 0.0)
    self.assertEqual(outputs[0].timestamps[0, 1], 3.0763125)

  def test_encode_with_word_timestamps(self):
    s2t_encoder = MockOpenAISpeechToTextEncoder(word_timestamps=True)
    s2t_encoder.setup()
    outputs = s2t_encoder.encode([self.test_sound])
    self.assertLen(outputs, 1)
    self.assertIsInstance(outputs[0], types.SoundEmbedding)
    self.assertEqual(
        outputs[0].embedding.tolist(),
        ['Roses', 'are', 'red', 'Violets', 'are', 'blue'],
    )
    self.assertEqual(outputs[0].timestamps.shape, (6, 2))
    self.assertEqual(outputs[0].timestamps[0, 0], 0.0)
    self.assertEqual(outputs[0].timestamps[0, 1], 0.7599999904632568)

  def test_encode_with_no_response(self):
    s2t_encoder = s2t_encoder_lib.OpenAISpeechToTextEncoder(
        server_url='mock_server_url',
        api_key='mock_api_key',
        model_name='mock_model_name',
    )
    s2t_encoder.setup()
    outputs = s2t_encoder.encode([self.test_sound])
    self.assertLen(outputs, 1)
    self.assertIsInstance(outputs[0], types.SoundEmbedding)
    self.assertEqual(
        str(outputs[0].embedding[0]), types.LLM_NO_RESPONSE_STR
    )
    self.assertEqual(outputs[0].timestamps.shape, (1, 2))
    self.assertEqual(outputs[0].timestamps[0, 0], 0.0)
    self.assertEqual(outputs[0].timestamps[0, 1], 0.0)

if __name__ == '__main__':
  absltest.main()
