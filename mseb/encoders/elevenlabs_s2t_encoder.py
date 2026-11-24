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

"""Speech-to-text encoder using the official ElevenLabs Python SDK.

Unlike `litellm_s2t_encoder`, this talks to the ElevenLabs Speech-to-Text API
directly, which makes the provider-specific `keyterms` contextual-biasing
parameter usable: LiteLLM's ElevenLabs adapter stringifies list-valued
provider parameters into a single form field, so `keyterms` cannot be passed
through it.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import json
import os
import time
from typing import Any, TYPE_CHECKING

from absl import logging
from mseb import encoder
from mseb import types
from mseb import utils
import numpy as np

if TYPE_CHECKING:
  from elevenlabs.speech_to_text import client as speech_to_text_client  # pylint: disable=g-import-not-at-top

# Default ElevenLabs API endpoint (`ElevenLabsEnvironment.PRODUCTION`).
DEFAULT_BASE_URL = 'https://api.elevenlabs.io'
# Default request timeout of the SDK client, in seconds.
DEFAULT_TIMEOUT_SECONDS = 240.0

# Limits documented for the `keyterms` parameter, see
# https://elevenlabs.io/docs/api-reference/speech-to-text/convert.
MAX_KEYTERMS = 1000
MAX_KEYTERM_CHARS = 50  # Each keyterm must be shorter than this.
MAX_KEYTERM_WORDS = 5

# Separator used by MSEB tasks to render a list of biasing terms as the single
# `SoundWithTitleAndContext.context_text` string.
DEFAULT_KEYTERM_SEPARATOR = ', '


def keyterms_from_context(
    context_text: str, separator: str = DEFAULT_KEYTERM_SEPARATOR
) -> tuple[list[str], list[str]]:
  """Splits a rendered context string into API-compliant keyterms.

  Terms are stripped and de-duplicated (first occurrence wins, order is kept).
  Terms that violate the API limits (empty, `MAX_KEYTERM_CHARS` or more
  characters, more than `MAX_KEYTERM_WORDS` words) are dropped, and the list is
  truncated to `MAX_KEYTERMS` entries.

  Args:
    context_text: The biasing terms joined by `separator`.
    separator: The separator used to join the terms.

  Returns:
    A tuple `(keyterms, dropped)` with the terms to send and the terms that
    were dropped (invalid or beyond the cap).
  """
  keyterms: list[str] = []
  dropped: list[str] = []
  seen: set[str] = set()
  for raw in context_text.split(separator):
    term = raw.strip()
    if not term or term == types.LLM_NO_RESPONSE_STR or term in seen:
      continue
    seen.add(term)
    if (
        len(term) >= MAX_KEYTERM_CHARS
        or len(term.split()) > MAX_KEYTERM_WORDS
        or len(keyterms) >= MAX_KEYTERMS
    ):
      dropped.append(term)
    else:
      keyterms.append(term)
  return keyterms, dropped


class ElevenLabsSpeechToTextEncoder(encoder.MultiModalEncoder):
  """Encode speech as text using the ElevenLabs Speech-to-Text API."""

  def __init__(
      self,
      api_key: str | None = None,
      model_id: str = 'scribe_v2',
      word_timestamps: bool = False,
      no_verbatim: bool = True,
      tag_audio_events: bool = False,
      diarize: bool = False,
      language_code: str | None = None,
      use_keyterms: bool = False,
      keyterm_separator: str = DEFAULT_KEYTERM_SEPARATOR,
      max_num_retry: int = 3,
      wait_time: float = 1.0,
      base_url: str | None = None,
  ):
    """Initializes the ElevenLabs Speech-to-text encoder.

    Args:
      api_key: ElevenLabs API key. If None, the SDK falls back to the
        `ELEVENLABS_API_KEY` environment variable.
      model_id: Speech-to-Text model, e.g. `scribe_v2` or `scribe_v1`.
      word_timestamps: Whether to return word-level or utterance-level
        timestamps.
      no_verbatim: If True, request a clean transcript (disfluencies removed);
        if False, a verbatim one.
      tag_audio_events: Whether to tag audio events such as (laughter).
      diarize: Whether to annotate speakers.
      language_code: Optional ISO-639-1/3 language code of the audio.
      use_keyterms: If True and an input is a `SoundWithTitleAndContext` with a
        non-empty `context_text`, the context is split with `keyterm_separator`
        and sent as the `keyterms` contextual-biasing parameter (see
        `keyterms_from_context`). If False (the default), context is ignored, so
        plain `Sound` inputs and context-free tasks are unaffected. Note that
        `keyterms` incur additional API cost, and more than 100 keyterms impose
        a 20 s minimum billable duration per request.
      keyterm_separator: Separator of the terms in `context_text`.
      max_num_retry: The maximum number of API call attempts.
      wait_time: The base wait time in seconds between retries.
      base_url: Optional API base URL override.
    """
    super().__init__()
    self._api_key = api_key
    self._model_id = model_id
    self._word_timestamps = word_timestamps
    self._no_verbatim = no_verbatim
    self._tag_audio_events = tag_audio_events
    self._diarize = diarize
    self._language_code = language_code
    self._use_keyterms = use_keyterms
    self._keyterm_separator = keyterm_separator
    self._max_try = max_num_retry
    self._wait_time = wait_time
    self._base_url = base_url
    self._client: speech_to_text_client.SpeechToTextClient | None = None

  def __getstate__(self) -> dict[str, Any]:
    state = self.__dict__.copy()
    if state.get('_client') is not None:
      state['_client'] = None
      state['_is_setup'] = False
    return state

  def _setup(self) -> None:
    """Creates the ElevenLabs Speech-to-Text client.

    The client is built from `elevenlabs.speech_to_text` and `elevenlabs.core`
    only, as `elevenlabs.ElevenLabs` would pull in the whole SDK, including
    `pyaudio` and its native ALSA dependency, which is not needed here.
    """
    if self._client is None:
      from elevenlabs.core import client_wrapper  # pylint: disable=g-import-not-at-top
      from elevenlabs.core import http_client as elevenlabs_http_client  # pylint: disable=g-import-not-at-top
      from elevenlabs.speech_to_text import client as speech_to_text_client  # pylint: disable=g-import-not-at-top

      api_key = self._api_key or os.getenv('ELEVENLABS_API_KEY') or None
      if not api_key and not self._base_url:
        raise ValueError(
            'ElevenLabs API key is required: pass `api_key` or set the'
            ' `ELEVENLABS_API_KEY` environment variable.'
        )
      wrapper = client_wrapper.SyncClientWrapper(
          api_key=api_key,
          base_url=self._base_url or DEFAULT_BASE_URL,
          timeout=DEFAULT_TIMEOUT_SECONDS,
          httpx_client=elevenlabs_http_client.httpx.Client(
              timeout=DEFAULT_TIMEOUT_SECONDS, follow_redirects=False
          ),
      )
      self._client = speech_to_text_client.SpeechToTextClient(
          client_wrapper=wrapper
      )

  def _check_input_types(self, batch: Sequence[types.MultiModalObject]) -> None:
    if not all(isinstance(x, types.Sound) for x in batch):
      raise ValueError(
          'ElevenLabsSpeechToTextEncoder only supports a batch of all Sound'
          ' inputs.'
      )

  def _encode(
      self,
      batch: Sequence[types.MultiModalObject],
  ) -> Sequence[types.SoundEmbedding]:
    """Encodes a batch of sounds into transcripts, one API call per sound."""
    outputs = []
    for example in batch:
      assert isinstance(example, types.Sound)
      outputs.append(self._encode_sound(example))
    return outputs

  def _keyterms(self, sound: types.Sound) -> tuple[list[str], list[str]]:
    """Returns `(keyterms, dropped)` for the sound, or empty lists."""
    if not self._use_keyterms:
      return [], []
    if not isinstance(sound, types.SoundWithTitleAndContext):
      return [], []
    if not sound.context_text:
      return [], []
    keyterms, dropped = keyterms_from_context(
        sound.context_text, self._keyterm_separator
    )
    if dropped:
      logging.warning(
          'Dropped %d of %d keyterms for %s (API limits).',
          len(dropped),
          len(keyterms) + len(dropped),
          sound.context.id,
      )
    return keyterms, dropped

  def _encode_sound(self, sound: types.Sound) -> types.SoundEmbedding:
    """Transcribes a single sound.

    Args:
      sound: A `Sound` (or `SoundWithTitleAndContext`) to transcribe.

    Returns:
      A `SoundEmbedding` whose `embedding` holds the transcript (one string) or
      the words (one string per word) when `word_timestamps` is set, with
      matching `timestamps` in seconds. When keyterms were sent, the output
      context's `debug_text` records them and the raw response as JSON
      (`{"keyterms": [...], "dropped_keyterms": [...], "model_response": ...}`).
      On persistent API failure the transcript is `types.LLM_NO_RESPONSE_STR`.
    """
    assert self._client is not None
    from elevenlabs.core import api_error  # pylint: disable=g-import-not-at-top

    keyterms, dropped = self._keyterms(sound)
    request: dict[str, Any] = dict(
        model_id=self._model_id,
        file=('file.wav', utils.sound_to_wav_bytes(sound), 'audio/wav'),
        no_verbatim=self._no_verbatim,
        tag_audio_events=self._tag_audio_events,
        diarize=self._diarize,
    )
    if self._language_code:
      request['language_code'] = self._language_code
    if keyterms:
      request['keyterms'] = keyterms

    response = None
    for n_try in range(self._max_try):
      try:
        response = self._client.convert(**request)
        break
      except Exception as e:  # pylint: disable=broad-exception-caught
        if isinstance(e, api_error.ApiError):
          if e.status_code in (401, 403):
            raise
          if (
              e.status_code is not None
              and 400 <= e.status_code < 500
              and e.status_code not in (408, 409, 429)
          ):
            logging.exception(
                'Permanent API error (%s) for %s',
                e.status_code,
                sound.context.id,
            )
            break
        logging.exception(
            'Failed to get prediction for %s', sound.context.id
        )
        if n_try + 1 < self._max_try:
          logging.warning('Failed to get prediction, retrying %d', n_try)
          time.sleep(self._wait_time * 1.5 ** (n_try + 1))

    text = getattr(response, 'text', None)
    words_with_spacing: list[tuple[str, float, float]] = []
    pending_spacing = ''
    for token in getattr(response, 'words', None) or []:
      if token.type == 'spacing':
        pending_spacing += token.text
      elif (
          token.type == 'word'
          and token.start is not None
          and token.end is not None
      ):
        words_with_spacing.append(
            (pending_spacing + token.text, token.start, token.end)
        )
        pending_spacing = ''

    if text is None:
      timestamps = np.array([[0.0, 0.0]], dtype=float)
      embeddings = np.array([types.LLM_NO_RESPONSE_STR], dtype=object)
    elif self._word_timestamps:
      timestamps = np.array(
          [[start, end] for _, start, end in words_with_spacing], dtype=float
      ).reshape(-1, 2)
      embeddings = np.array(
          [word_text for word_text, _, _ in words_with_spacing], dtype=object
      )
    else:
      if words_with_spacing:
        start, end = words_with_spacing[0][1], words_with_spacing[-1][2]
      else:
        # No word-level timing available (e.g. empty transcript on
        # disfluency-only audio with no_verbatim=True). Fall back to the span
        # of the waveform sent to the provider (word timestamps are relative
        # to it); `sound.context.waveform_end_second` defaults to float32.max.
        start = 0.0
        end = len(sound.waveform) / sound.context.sample_rate
      timestamps = np.array([[start, end]], dtype=float)
      embeddings = np.array([text], dtype=object)

    context = sound.context
    if keyterms or dropped:
      context = dataclasses.replace(
          sound.context,
          debug_text=json.dumps({
              'keyterms': keyterms,
              'dropped_keyterms': dropped,
              'model_response': text,
          }),
      )

    return types.SoundEmbedding(
        embedding=embeddings, timestamps=timestamps, context=context  # pyrefly: ignore[bad-argument-type]
    )
