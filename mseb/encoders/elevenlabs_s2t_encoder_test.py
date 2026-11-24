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

"""Tests for the ElevenLabs SDK-based speech-to-text encoder."""

import json
import os
import pathlib
import pickle
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from mseb import types
from mseb import utils
import pytest

api_error = pytest.importorskip('elevenlabs.core.api_error')
elevenlabs_http_client = pytest.importorskip('elevenlabs.core.http_client')
speech_to_text_client = pytest.importorskip('elevenlabs.speech_to_text.client')
speech_to_text_chunk_response_model = pytest.importorskip(
    'elevenlabs.types.speech_to_text_chunk_response_model'
)
speech_to_text_word_response_model = pytest.importorskip(
    'elevenlabs.types.speech_to_text_word_response_model'
)
s2t_encoder_lib = pytest.importorskip('mseb.encoders.elevenlabs_s2t_encoder')

_Chunk = speech_to_text_chunk_response_model.SpeechToTextChunkResponseModel
_Word = speech_to_text_word_response_model.SpeechToTextWordResponseModel


def _word(text, start, end, type_='word'):
  return _Word(text=text, start=start, end=end, type=type_, logprob=0.0)


_RESPONSE = _Chunk(
    language_code='eng',
    language_probability=1.0,
    text='Roses are red. Violets are blue.',
    words=[
        _word('Roses', 0.0, 0.76),
        _word(' ', 0.76, 0.76, 'spacing'),
        _word('are', 0.76, 1.0),
        _word(' ', 1.0, 1.0, 'spacing'),
        _word('red.', 1.0, 1.44),
        _word(' ', 1.44, 2.08, 'spacing'),
        _word('Violets', 2.08, 2.14),
        _word(' ', 2.14, 2.14, 'spacing'),
        _word('are', 2.14, 2.36),
        _word(' ', 2.36, 2.36, 'spacing'),
        _word('blue.', 2.36, 2.52),
    ],
)


class KeytermsFromContextTest(parameterized.TestCase):

  def test_splits_strips_and_dedupes(self):
    keyterms, dropped = s2t_encoder_lib.keyterms_from_context(
        'Anna,  Gazprom , Anna, , Tele2 Russia'
    )
    self.assertEqual(keyterms, ['Anna', 'Gazprom', 'Tele2 Russia'])
    self.assertEmpty(dropped)

  @parameterized.named_parameters(
      dict(
          testcase_name='max_chars_minus_one_kept',
          context_text='x' * (s2t_encoder_lib.MAX_KEYTERM_CHARS - 1),
          expected_keyterms=['x' * (s2t_encoder_lib.MAX_KEYTERM_CHARS - 1)],
          expected_dropped=[],
      ),
      dict(
          testcase_name='max_chars_dropped',
          context_text='x' * s2t_encoder_lib.MAX_KEYTERM_CHARS,
          expected_keyterms=[],
          expected_dropped=['x' * s2t_encoder_lib.MAX_KEYTERM_CHARS],
      ),
      dict(
          testcase_name='max_words_kept',
          context_text='one two three four five',
          expected_keyterms=['one two three four five'],
          expected_dropped=[],
      ),
      dict(
          testcase_name='over_max_words_dropped',
          context_text='one two three four five six',
          expected_keyterms=[],
          expected_dropped=['one two three four five six'],
      ),
      dict(
          testcase_name='no_response_sentinel_filtered',
          context_text=f'Roses, {types.LLM_NO_RESPONSE_STR}, Violets',
          expected_keyterms=['Roses', 'Violets'],
          expected_dropped=[],
      ),
  )
  def test_term_validation_boundaries(
      self, context_text, expected_keyterms, expected_dropped
  ):
    keyterms, dropped = s2t_encoder_lib.keyterms_from_context(context_text)
    self.assertEqual(keyterms, expected_keyterms)
    self.assertEqual(dropped, expected_dropped)

  def test_caps_number_of_terms(self):
    n = s2t_encoder_lib.MAX_KEYTERMS + 3
    keyterms, dropped = s2t_encoder_lib.keyterms_from_context(
        ', '.join(f't{i}' for i in range(n))
    )
    self.assertLen(keyterms, s2t_encoder_lib.MAX_KEYTERMS)
    self.assertEqual(dropped, [f't{i}' for i in range(n - 3, n)])

  def test_custom_separator(self):
    keyterms, _ = s2t_encoder_lib.keyterms_from_context('a|b', separator='|')
    self.assertEqual(keyterms, ['a', 'b'])


class ElevenLabsSpeechToTextEncoderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, 'testdata'
    )
    with open(os.path.join(testdata_path, 'roses-are.wav'), 'rb') as f:
      samples, sample_rate = utils.wav_bytes_to_waveform(f.read())
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

  def _make_encoder(self, response=_RESPONSE, **kwargs):
    enc = s2t_encoder_lib.ElevenLabsSpeechToTextEncoder(
        api_key='mock_api_key', **kwargs
    )
    client = mock.create_autospec(
        speech_to_text_client.SpeechToTextClient,
        instance=True,
    )
    client.convert.return_value = response
    with mock.patch.object(
        speech_to_text_client,
        'SpeechToTextClient',
        autospec=True,
        return_value=client,
    ):
      enc.setup()
    return enc, client.convert

  def test_encode(self):
    enc, convert = self._make_encoder()
    outputs = enc.encode([self.test_sound])
    self.assertLen(outputs, 1)
    output = outputs[0]
    assert isinstance(output, types.SoundEmbedding)
    self.assertEqual(
        output.embedding.tolist(), ['Roses are red. Violets are blue.']
    )
    # Utterance span is taken from the first/last word.
    self.assertEqual(output.timestamps.tolist(), [[0.0, 2.52]])
    self.assertIsNone(output.context.debug_text)
    convert.assert_called_once()
    kwargs = convert.call_args.kwargs
    self.assertEqual(kwargs['model_id'], 'scribe_v2')
    self.assertTrue(kwargs['no_verbatim'])
    self.assertFalse(kwargs['tag_audio_events'])
    self.assertFalse(kwargs['diarize'])
    self.assertNotIn('request_options', kwargs)
    self.assertNotIn('keyterms', kwargs)
    self.assertNotIn('language_code', kwargs)
    filename, wav_bytes, content_type = kwargs['file']
    self.assertEqual((filename, content_type), ('file.wav', 'audio/wav'))
    self.assertEqual(wav_bytes[:4], b'RIFF')

  def test_encode_with_word_timestamps(self):
    enc, _ = self._make_encoder(word_timestamps=True)
    output = enc.encode([self.test_sound])[0]
    assert isinstance(output, types.SoundEmbedding)
    # Preceding spacing tokens are attached to each word token.
    self.assertEqual(
        output.embedding.tolist(),
        ['Roses', ' are', ' red.', ' Violets', ' are', ' blue.'],
    )
    self.assertEqual(
        ''.join(output.embedding.tolist()),
        'Roses are red. Violets are blue.',
    )
    self.assertEqual(output.timestamps.shape, (6, 2))
    self.assertEqual(output.timestamps[0].tolist(), [0.0, 0.76])
    self.assertEqual(output.timestamps[-1].tolist(), [2.36, 2.52])

  def test_encode_with_keyterms(self):
    enc, convert = self._make_encoder(use_keyterms=True)
    output = enc.encode([self.test_sound_with_context])[0]
    assert isinstance(output, types.SoundEmbedding)
    self.assertEqual(convert.call_args.kwargs['keyterms'], ['Roses', 'Violets'])
    self.assertIsNotNone(output.context.debug_text)
    assert output.context.debug_text is not None
    self.assertEqual(
        json.loads(output.context.debug_text),
        {
            'keyterms': ['Roses', 'Violets'],
            'dropped_keyterms': [],
            'model_response': 'Roses are red. Violets are blue.',
        },
    )
    # The input context is not mutated.
    self.assertIsNone(self.test_sound_with_context.context.debug_text)

  def test_encode_with_keyterms_records_dropped_terms(self):
    enc, convert = self._make_encoder(use_keyterms=True)
    too_long = 'x' * s2t_encoder_lib.MAX_KEYTERM_CHARS
    sound = types.SoundWithTitleAndContext(
        waveform=self.test_sound.waveform,
        context=self.test_sound.context,
        context_text=f'Roses, {too_long}',
    )
    output = enc.encode([sound])[0]
    assert isinstance(output, types.SoundEmbedding)
    assert output.context.debug_text is not None
    self.assertEqual(convert.call_args.kwargs['keyterms'], ['Roses'])
    self.assertEqual(
        json.loads(output.context.debug_text)['dropped_keyterms'], [too_long]
    )

  def test_encode_all_keyterms_dropped_still_records_debug_text(self):
    enc, convert = self._make_encoder(use_keyterms=True)
    too_long = 'x' * s2t_encoder_lib.MAX_KEYTERM_CHARS
    sound = types.SoundWithTitleAndContext(
        waveform=self.test_sound.waveform,
        context=self.test_sound.context,
        context_text=too_long,
    )
    output = enc.encode([sound])[0]
    assert isinstance(output, types.SoundEmbedding)
    self.assertNotIn('keyterms', convert.call_args.kwargs)
    self.assertIsNotNone(output.context.debug_text)
    assert output.context.debug_text is not None
    self.assertEqual(
        json.loads(output.context.debug_text),
        {
            'keyterms': [],
            'dropped_keyterms': [too_long],
            'model_response': 'Roses are red. Violets are blue.',
        },
    )

  def test_encode_keyterms_serialized_as_repeated_multipart_fields(self):
    captured_bodies: list[bytes] = []

    def handler(request):
      captured_bodies.append(request.read())
      return elevenlabs_http_client.httpx.Response(
          200,
          json={
              'language_code': 'eng',
              'language_probability': 1.0,
              'text': 'Roses are red.',
              'words': [
                  {
                      'text': 'Roses',
                      'start': 0.0,
                      'end': 0.5,
                      'type': 'word',
                      'logprob': 0.0,
                  }
              ],
          },
      )

    transport = elevenlabs_http_client.httpx.MockTransport(handler)
    enc = s2t_encoder_lib.ElevenLabsSpeechToTextEncoder(
        api_key='k', base_url='https://example.test', use_keyterms=True
    )
    with mock.patch.object(
        elevenlabs_http_client.httpx,
        'Client',
        return_value=elevenlabs_http_client.httpx.Client(transport=transport),
    ):
      enc.setup()
    output = enc.encode([self.test_sound_with_context])[0]
    assert isinstance(output, types.SoundEmbedding)
    self.assertEqual(output.embedding.tolist(), ['Roses are red.'])
    self.assertLen(captured_bodies, 1)
    body = captured_bodies[0]
    self.assertIn(b'name="keyterms"\r\n\r\nRoses\r\n', body)
    self.assertIn(b'name="keyterms"\r\n\r\nViolets\r\n', body)
    self.assertNotIn(b"['Roses', 'Violets']", body)

  @parameterized.named_parameters(
      dict(
          testcase_name='plain_sound',
          use_keyterms=True,
          context_text=None,
      ),
      dict(
          testcase_name='empty_context',
          use_keyterms=True,
          context_text='',
      ),
      dict(
          testcase_name='use_keyterms_false',
          use_keyterms=False,
          context_text='Roses, Violets',
      ),
  )
  def test_encode_ignores_keyterms(self, use_keyterms, context_text):
    enc, convert = self._make_encoder(use_keyterms=use_keyterms)
    if context_text is None:
      sound = self.test_sound
    else:
      sound = types.SoundWithTitleAndContext(
          waveform=self.test_sound.waveform,
          context=self.test_sound.context,
          context_text=context_text,
      )
    output = enc.encode([sound])[0]
    assert isinstance(output, types.SoundEmbedding)
    self.assertNotIn('keyterms', convert.call_args.kwargs)
    self.assertIsNone(output.context.debug_text)

  def test_encode_passes_verbatim_and_language(self):
    enc, convert = self._make_encoder(no_verbatim=False, language_code='en')
    enc.encode([self.test_sound])
    kwargs = convert.call_args.kwargs
    self.assertFalse(kwargs['no_verbatim'])
    self.assertEqual(kwargs['language_code'], 'en')

  def test_encode_with_no_response(self):
    enc, convert = self._make_encoder(max_num_retry=1)
    convert.side_effect = RuntimeError('boom')
    with mock.patch.object(s2t_encoder_lib.time, 'sleep') as mock_sleep:
      output = enc.encode([self.test_sound])[0]
    assert isinstance(output, types.SoundEmbedding)
    self.assertEqual(output.embedding.tolist(), [types.LLM_NO_RESPONSE_STR])
    self.assertEqual(output.timestamps.tolist(), [[0.0, 0.0]])
    mock_sleep.assert_not_called()

  def test_encode_retries_then_succeeds(self):
    enc, convert = self._make_encoder()
    convert.side_effect = [RuntimeError('boom'), _RESPONSE]
    with mock.patch.object(s2t_encoder_lib.time, 'sleep') as mock_sleep:
      output = enc.encode([self.test_sound])[0]
    assert isinstance(output, types.SoundEmbedding)
    self.assertEqual(
        output.embedding.tolist(), ['Roses are red. Violets are blue.']
    )
    self.assertEqual(convert.call_count, 2)
    mock_sleep.assert_called_once()

  @parameterized.parameters(401, 403)
  def test_encode_raises_on_auth_error(self, status_code):
    enc, convert = self._make_encoder(max_num_retry=3)
    convert.side_effect = api_error.ApiError(
        status_code=status_code, body='unauth'
    )
    with mock.patch.object(s2t_encoder_lib.time, 'sleep') as mock_sleep:
      with self.assertRaises(api_error.ApiError):
        enc.encode([self.test_sound])
    convert.assert_called_once()
    mock_sleep.assert_not_called()

  def test_encode_does_not_retry_permanent_client_error(self):
    enc, convert = self._make_encoder(max_num_retry=3)
    convert.side_effect = api_error.ApiError(
        status_code=422, body='unprocessable'
    )
    with mock.patch.object(s2t_encoder_lib.time, 'sleep') as mock_sleep:
      output = enc.encode([self.test_sound])[0]
    assert isinstance(output, types.SoundEmbedding)
    self.assertEqual(output.embedding.tolist(), [types.LLM_NO_RESPONSE_STR])
    convert.assert_called_once()
    mock_sleep.assert_not_called()

  def test_encode_retries_rate_limit_error(self):
    enc, convert = self._make_encoder(max_num_retry=3)
    convert.side_effect = [
        api_error.ApiError(status_code=429, body='rate limit'),
        _RESPONSE,
    ]
    with mock.patch.object(s2t_encoder_lib.time, 'sleep') as mock_sleep:
      output = enc.encode([self.test_sound])[0]
    assert isinstance(output, types.SoundEmbedding)
    self.assertEqual(
        output.embedding.tolist(), ['Roses are red. Violets are blue.']
    )
    self.assertEqual(convert.call_count, 2)
    mock_sleep.assert_called_once()

  def test_encode_empty_words_falls_back_to_waveform_duration(self):
    empty_response = _Chunk(
        language_code='eng',
        language_probability=1.0,
        text='',
        words=[],
    )
    enc, _ = self._make_encoder(response=empty_response)
    sound = types.Sound(
        waveform=self.test_sound.waveform,
        context=types.SoundContextParams(
            id='default_span',
            sample_rate=self.test_sound.context.sample_rate,
            length=len(self.test_sound.waveform),
        ),
    )
    output = enc.encode([sound])[0]
    assert isinstance(output, types.SoundEmbedding)
    expected_end = len(sound.waveform) / sound.context.sample_rate
    self.assertEqual(output.embedding.tolist(), [''])
    self.assertAlmostEqual(float(output.timestamps[0][0]), 0.0)
    self.assertAlmostEqual(float(output.timestamps[0][1]), expected_end)

  def test_setup_builds_client_with_api_key_and_base_url(self):
    enc = s2t_encoder_lib.ElevenLabsSpeechToTextEncoder(
        api_key='k', base_url='https://example.test'
    )
    enc.setup()
    wrapper = enc._client._client_wrapper  # pylint: disable=protected-access
    self.assertEqual(wrapper.get_headers()['xi-api-key'], 'k')
    self.assertEqual(wrapper.get_base_url(), 'https://example.test')
    self.assertFalse(wrapper.httpx_client.httpx_client.follow_redirects)

  def test_setup_raises_without_api_key(self):
    enc = s2t_encoder_lib.ElevenLabsSpeechToTextEncoder()
    with mock.patch.dict(os.environ, {}, clear=True):
      with self.assertRaisesRegex(ValueError, 'ElevenLabs API key is required'):
        enc.setup()

  def test_setup_allows_missing_api_key_with_custom_base_url(self):
    enc = s2t_encoder_lib.ElevenLabsSpeechToTextEncoder(
        base_url='http://localhost:8080'
    )
    with mock.patch.dict(os.environ, {'ELEVENLABS_API_KEY': ''}, clear=True):
      enc.setup()
    wrapper = enc._client._client_wrapper  # pylint: disable=protected-access
    self.assertNotIn('xi-api-key', wrapper.get_headers())
    self.assertEqual(wrapper.get_base_url(), 'http://localhost:8080')

  def test_pickle_after_setup_resets_client(self):
    enc = s2t_encoder_lib.ElevenLabsSpeechToTextEncoder(
        api_key='k', base_url='https://example.test'
    )
    enc.setup()
    self.assertIsNotNone(enc._client)  # pylint: disable=protected-access
    state = enc.__getstate__()
    self.assertIsNone(state['_client'])
    self.assertFalse(state['_is_setup'])
    self.assertNotEmpty(pickle.dumps(enc))

  def test_rejects_non_sound_inputs(self):
    enc, _ = self._make_encoder()
    with self.assertRaises(ValueError):
      enc.encode(
          [types.Text(text='x', context=types.TextContextParams(id='t'))]
      )


if __name__ == '__main__':
  absltest.main()
