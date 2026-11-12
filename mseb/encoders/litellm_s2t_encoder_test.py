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

"""Tests for the LiteLLM API-based encoder."""

import os
import pathlib
from unittest import mock

from absl.testing import absltest
from litellm.types.utils import TranscriptionResponse
from mseb import types
from mseb import utils
import pytest

s2t_encoder_lib = pytest.importorskip('mseb.encoders.litellm_s2t_encoder')


class LiteLLMSpeechToTextEncoderTest(absltest.TestCase):

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
    self.mock_response = TranscriptionResponse.model_construct(
        text='Roses are red, violets are blue.',
        usage=None,
        task='transcribe',
        language='eng',
        words=[
            {'word': 'Roses', 'start': 0.479, 'end': 0.84},
            {'word': 'are', 'start': 0.839, 'end': 0.959},
            {'word': 'red,', 'start': 0.959, 'end': 1.839},
            {'word': 'violets', 'start': 1.86, 'end': 2.199},
            {'word': 'are', 'start': 2.22, 'end': 2.299},
            {'word': 'blue.', 'start': 2.319, 'end': 3.039},
        ],
        duration=3.0763125,
    )

  @mock.patch('litellm.transcription')
  def test_encode(self, mock_transcription):
    mock_transcription.return_value = self.mock_response
    s2t_encoder = s2t_encoder_lib.LiteLLMSpeechToTextEncoder(
        model_name='test_model',
        api_key='test_api_key',
    )
    s2t_encoder.setup()
    outputs = s2t_encoder.encode([self.test_sound])
    self.assertLen(outputs, 1)
    output = outputs[0]
    assert isinstance(output, types.SoundEmbedding)
    self.assertEqual(
        str(output.embedding[0]), 'Roses are red, violets are blue.'
    )
    self.assertEqual(output.timestamps.shape, (1, 2))
    self.assertEqual(output.timestamps[0, 0], 0.479)
    self.assertEqual(output.timestamps[0, 1], 3.039)

  @mock.patch('litellm.transcription')
  def test_encode_with_word_timestamps(self, mock_transcription):
    mock_transcription.return_value = self.mock_response
    s2t_encoder = s2t_encoder_lib.LiteLLMSpeechToTextEncoder(
        model_name='test_model',
        api_key='test_api_key',
        word_timestamps=True,
    )
    s2t_encoder.setup()
    outputs = s2t_encoder.encode([self.test_sound])
    self.assertLen(outputs, 1)
    output = outputs[0]
    assert isinstance(output, types.SoundEmbedding)
    self.assertEqual(
        output.embedding.tolist(),
        [' Roses', ' are', ' red,', ' violets', ' are', ' blue.'],
    )
    self.assertEqual(output.timestamps.shape, (6, 2))
    self.assertEqual(output.timestamps[0, 0], 0.479)
    self.assertEqual(output.timestamps[0, 1], 0.84)

  @mock.patch('litellm.transcription')
  def test_no_transcription_kwargs_by_default(self, mock_transcription):
    mock_transcription.return_value = self.mock_response
    s2t_encoder = s2t_encoder_lib.LiteLLMSpeechToTextEncoder(
        model_name='test_model',
        api_key='test_api_key',
    )
    s2t_encoder.setup()
    s2t_encoder.encode([self.test_sound])
    mock_transcription.assert_called_once()
    self.assertEqual(
        set(mock_transcription.call_args.kwargs),
        {
            'model',
            'file',
            'response_format',
            'timestamp_granularities',
            'api_key',
            'additional_drop_params',
            'drop_params',
        },
    )

  @mock.patch('litellm.transcription')
  def test_transcription_kwargs_are_forwarded(self, mock_transcription):
    mock_transcription.return_value = self.mock_response
    s2t_encoder = s2t_encoder_lib.LiteLLMSpeechToTextEncoder(
        model_name='test_model',
        api_key='test_api_key',
        transcription_kwargs=dict(
            tag_audio_events=False, diarize=False, no_verbatim=True
        ),
    )
    s2t_encoder.setup()
    outputs = s2t_encoder.encode([self.test_sound])
    self.assertLen(outputs, 1)
    mock_transcription.assert_called_once()
    kwargs = mock_transcription.call_args.kwargs
    self.assertEqual(kwargs['model'], 'test_model')
    self.assertEqual(kwargs['tag_audio_events'], False)
    self.assertEqual(kwargs['diarize'], False)
    self.assertEqual(kwargs['no_verbatim'], True)

  def _empty_response(self, **kwargs):
    # ElevenLabs Scribe with no_verbatim=True returns an empty transcript and
    # an empty `words` list for utterances made only of disfluencies.
    return TranscriptionResponse.model_construct(
        text='',
        usage=None,
        task='transcribe',
        language='eng',
        words=[],
        **kwargs,
    )

  def _encode_one(self, mock_transcription, response, sound=None, **kwargs):
    mock_transcription.return_value = response
    s2t_encoder = s2t_encoder_lib.LiteLLMSpeechToTextEncoder(
        model_name='test_model',
        api_key='test_api_key',
        **kwargs,
    )
    s2t_encoder.setup()
    outputs = s2t_encoder.encode([sound or self.test_sound])
    self.assertLen(outputs, 1)
    output = outputs[0]
    assert isinstance(output, types.SoundEmbedding)
    return output

  @mock.patch('litellm.transcription')
  def test_encode_empty_transcript(self, mock_transcription):
    output = self._encode_one(mock_transcription, self._empty_response())
    self.assertEqual(output.embedding.tolist(), [''])
    self.assertEqual(output.timestamps.shape, (1, 2))
    self.assertEqual(output.timestamps[0, 0], 0.0)
    self.assertEqual(
        output.timestamps[0, 1],
        len(self.test_sound.waveform) / self.test_sound.context.sample_rate,
    )

  @mock.patch('litellm.transcription')
  def test_encode_empty_transcript_uses_response_duration(
      self, mock_transcription
  ):
    output = self._encode_one(
        mock_transcription, self._empty_response(duration=1.25)
    )
    self.assertEqual(output.embedding.tolist(), [''])
    self.assertEqual(output.timestamps.tolist(), [[0.0, 1.25]])

  @mock.patch('litellm.transcription')
  def test_encode_empty_transcript_default_context(self, mock_transcription):
    # `waveform_end_second` defaults to float32 max; the fallback must use the
    # actual waveform length instead.
    sound = types.Sound(
        waveform=self.test_sound.waveform,
        context=types.SoundContextParams(
            id='test',
            sample_rate=self.test_sound.context.sample_rate,
            length=len(self.test_sound.waveform),
        ),
    )
    output = self._encode_one(
        mock_transcription, self._empty_response(), sound=sound
    )
    self.assertEqual(output.timestamps.shape, (1, 2))
    self.assertEqual(output.timestamps[0, 0], 0.0)
    self.assertEqual(
        output.timestamps[0, 1],
        len(sound.waveform) / sound.context.sample_rate,
    )

  @mock.patch('litellm.transcription')
  def test_encode_missing_words_and_none_text(self, mock_transcription):
    response = TranscriptionResponse.model_construct(
        text=None, usage=None, task='transcribe', language='eng'
    )
    output = self._encode_one(mock_transcription, response)
    self.assertEqual(output.embedding.tolist(), [''])
    self.assertEqual(output.timestamps.shape, (1, 2))
    self.assertEqual(output.timestamps[0, 0], 0.0)
    self.assertEqual(
        output.timestamps[0, 1],
        len(self.test_sound.waveform) / self.test_sound.context.sample_rate,
    )

  @mock.patch('litellm.transcription')
  def test_encode_empty_transcript_with_word_timestamps(
      self, mock_transcription
  ):
    output = self._encode_one(
        mock_transcription, self._empty_response(), word_timestamps=True
    )
    self.assertEqual(output.embedding.tolist(), [])
    self.assertEqual(output.timestamps.shape, (0, 2))

  @mock.patch('litellm.transcription')
  def test_encode_missing_words_with_word_timestamps(self, mock_transcription):
    response = TranscriptionResponse.model_construct(
        text=None, usage=None, task='transcribe', language='eng'
    )
    output = self._encode_one(
        mock_transcription, response, word_timestamps=True
    )
    self.assertEqual(output.embedding.tolist(), [])
    self.assertEqual(output.timestamps.shape, (0, 2))


if __name__ == '__main__':
  absltest.main()
