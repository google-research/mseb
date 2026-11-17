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

import json
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

  def __init__(
      self,
      word_timestamps: bool = False,
      context_prompt_template: str | None = None,
  ):
    super().__init__(
        server_url='https://mock_server_url',
        api_key='mock_api_key',
        model_name='mock_model_name',
        word_timestamps=word_timestamps,
        context_prompt_template=context_prompt_template,
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
        waveform=samples,  # pyrefly: ignore[bad-argument-type]
        context=types.SoundContextParams(
            id='test',
            sample_rate=sample_rate,
            length=len(samples),
            waveform_start_second=0.0,
            waveform_end_second=len(samples) / sample_rate,
        ),
    )
    self.test_sound_with_context = types.SoundWithTitleAndContext(
        waveform=self.test_sound.waveform,
        context=self.test_sound.context,
        context_text='Roses, Violets',
    )

  def _create_kwargs(self, s2t_encoder):
    """Returns the keyword arguments of the last transcription API call."""
    create = s2t_encoder._client.audio.transcriptions.create  # pylint: disable=protected-access
    create.assert_called_once()
    return create.call_args.kwargs

  def test_encode(self):
    s2t_encoder = MockOpenAISpeechToTextEncoder()
    s2t_encoder.setup()
    outputs = s2t_encoder.encode([self.test_sound])
    self.assertLen(outputs, 1)
    output = outputs[0]
    assert isinstance(output, types.SoundEmbedding)
    self.assertEqual(
        str(output.embedding[0]), 'Roses are red. Violets are blue.'
    )
    self.assertEqual(output.timestamps.shape, (1, 2))
    self.assertEqual(output.timestamps[0, 0], 0.0)
    self.assertEqual(output.timestamps[0, 1], 3.0763125)
    self.assertNotIn('prompt', self._create_kwargs(s2t_encoder))
    self.assertIsNone(output.context.debug_text)

  def test_encode_with_context_prompt(self):
    s2t_encoder = MockOpenAISpeechToTextEncoder(
        context_prompt_template='Terms: {context}'
    )
    s2t_encoder.setup()
    outputs = s2t_encoder.encode([self.test_sound_with_context])
    self.assertLen(outputs, 1)
    output = outputs[0]
    assert isinstance(output, types.SoundEmbedding)
    self.assertEqual(
        str(output.embedding[0]), 'Roses are red. Violets are blue.'
    )
    self.assertEqual(
        self._create_kwargs(s2t_encoder)['prompt'], 'Terms: Roses, Violets'
    )
    self.assertIsNotNone(output.context.debug_text)
    assert output.context.debug_text is not None
    self.assertEqual(
        json.loads(output.context.debug_text),
        {
            'prompt_text': 'Terms: Roses, Violets',
            'model_response': 'Roses are red. Violets are blue.',
        },
    )
    # The input context is not mutated.
    self.assertIsNone(self.test_sound_with_context.context.debug_text)

  def test_encode_with_default_context_prompt_template(self):
    s2t_encoder = MockOpenAISpeechToTextEncoder(
        context_prompt_template=s2t_encoder_lib.DEFAULT_CONTEXT_PROMPT_TEMPLATE
    )
    s2t_encoder.setup()
    s2t_encoder.encode([self.test_sound_with_context])
    prompt = self._create_kwargs(s2t_encoder)['prompt']
    self.assertEndsWith(prompt, 'use this exact spelling: Roses, Violets')

  def test_encode_with_context_prompt_ignores_plain_sound(self):
    s2t_encoder = MockOpenAISpeechToTextEncoder(
        context_prompt_template='Terms: {context}'
    )
    s2t_encoder.setup()
    s2t_encoder.encode([self.test_sound])
    self.assertNotIn('prompt', self._create_kwargs(s2t_encoder))

  def test_encode_with_context_prompt_ignores_empty_context(self):
    s2t_encoder = MockOpenAISpeechToTextEncoder(
        context_prompt_template='Terms: {context}'
    )
    s2t_encoder.setup()
    sound = types.SoundWithTitleAndContext(
        waveform=self.test_sound.waveform,
        context=self.test_sound.context,
        context_text='',
    )
    s2t_encoder.encode([sound])
    self.assertNotIn('prompt', self._create_kwargs(s2t_encoder))

  def test_encode_without_template_ignores_context(self):
    s2t_encoder = MockOpenAISpeechToTextEncoder()
    s2t_encoder.setup()
    s2t_encoder.encode([self.test_sound_with_context])
    self.assertNotIn('prompt', self._create_kwargs(s2t_encoder))

  def test_encode_with_word_timestamps(self):
    s2t_encoder = MockOpenAISpeechToTextEncoder(word_timestamps=True)
    s2t_encoder.setup()
    outputs = s2t_encoder.encode([self.test_sound])
    self.assertLen(outputs, 1)
    output = outputs[0]
    assert isinstance(output, types.SoundEmbedding)
    self.assertEqual(
        output.embedding.tolist(),
        ['Roses', 'are', 'red', 'Violets', 'are', 'blue'],
    )
    self.assertEqual(output.timestamps.shape, (6, 2))
    self.assertEqual(output.timestamps[0, 0], 0.0)
    self.assertEqual(output.timestamps[0, 1], 0.7599999904632568)

  def _encoder_with_retries(self, create: mock.MagicMock):
    s2t_encoder = s2t_encoder_lib.OpenAISpeechToTextEncoder(
        server_url='https://mock_server_url',
        api_key='mock_api_key',
        model_name='mock_model_name',
        max_num_retry=3,
        wait_time=0.0,
    )
    s2t_encoder._client = mock.MagicMock()
    s2t_encoder._client.audio.transcriptions.create = create
    s2t_encoder.setup()
    return s2t_encoder

  def test_successful_request_is_not_repeated(self):
    create = mock.MagicMock(return_value=Transcription(text='ok', usage=None))
    s2t_encoder = self._encoder_with_retries(create)
    output = s2t_encoder.encode([self.test_sound])[0]
    assert isinstance(output, types.SoundEmbedding)
    self.assertEqual(str(output.embedding[0]), 'ok')
    self.assertEqual(create.call_count, 1)

  def test_failed_request_is_retried_until_it_succeeds(self):
    create = mock.MagicMock(
        side_effect=[
            RuntimeError('unavailable'),
            Transcription(text='ok', usage=None),
        ]
    )
    s2t_encoder = self._encoder_with_retries(create)
    output = s2t_encoder.encode([self.test_sound])[0]
    assert isinstance(output, types.SoundEmbedding)
    self.assertEqual(str(output.embedding[0]), 'ok')
    self.assertEqual(create.call_count, 2)

  def test_audio_is_sent_from_its_start_on_every_attempt(self):
    sent = []

    def create_that_reads_the_file_without_rewinding(**kwargs):
      sent.append(kwargs['file'].read())
      if len(sent) == 1:
        raise RuntimeError('unavailable')
      return Transcription(text='ok', usage=None)

    create = mock.MagicMock(
        side_effect=create_that_reads_the_file_without_rewinding
    )
    s2t_encoder = self._encoder_with_retries(create)
    s2t_encoder.encode([self.test_sound])
    self.assertLen(sent, 2)
    # The audio starts with the marker of a FLAC stream.
    self.assertEqual(sent[0][:4], b'fLaC')
    self.assertEqual(sent[0], sent[1])

  def test_no_response_after_all_attempts_failed(self):
    create = mock.MagicMock(side_effect=RuntimeError('unavailable'))
    s2t_encoder = self._encoder_with_retries(create)
    output = s2t_encoder.encode([self.test_sound])[0]
    assert isinstance(output, types.SoundEmbedding)
    self.assertEqual(output.embedding.tolist(), [types.LLM_NO_RESPONSE_STR])
    self.assertEqual(output.timestamps.tolist(), [[0.0, 0.0]])
    self.assertEqual(create.call_count, 3)

  def test_encode_with_no_response(self):
    s2t_encoder = s2t_encoder_lib.OpenAISpeechToTextEncoder(
        server_url='mock_server_url',
        api_key='mock_api_key',
        model_name='mock_model_name',
    )
    s2t_encoder.setup()
    outputs = s2t_encoder.encode([self.test_sound])
    self.assertLen(outputs, 1)
    output = outputs[0]
    assert isinstance(output, types.SoundEmbedding)
    self.assertEqual(
        str(output.embedding[0]), types.LLM_NO_RESPONSE_STR
    )
    self.assertEqual(output.timestamps.shape, (1, 2))
    self.assertEqual(output.timestamps[0, 0], 0.0)
    self.assertEqual(output.timestamps[0, 1], 0.0)

if __name__ == '__main__':
  absltest.main()
