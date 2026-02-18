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

"""Speech-to-text encoder using the OpenAI Audio Transciptions API.
"""

import io
import time
from typing import Sequence

from absl import logging
from mseb import encoder
from mseb import types
import numpy as np
import openai
import soundfile


class OpenAISpeechToTextEncoder(encoder.MultiModalEncoder):
  """Encode speech as text using the OpenAI Audio Transcriptions API."""

  def __init__(
      self,
      model_name: str,
      api_key: str,
      temperature: float = 0.0,
      word_timestamps: bool = False,
      server_url: str | None = None,
      max_num_retry: int = 1,
      wait_time: float = 1.0,
  ):
    """Initializes the OpenAI Speech-to-text encoder.

    Args:
      model_name: Name of the OpenAI Transcriptions model.
      api_key: API key for the OpenAI Transcriptions server.
      temperature: Temperature for the OpenAI Transcriptions model.
      word_timestamps: Whether to return word-level or segment-level timestamps.
      server_url: URL of the OpenAI server.
      max_num_retry: The maximum number of retries for the model.
      wait_time: The wait time in seconds between retries.
    """
    super().__init__()
    self._server_url = server_url
    self._api_key = api_key
    self._model_name = model_name
    self._temperature = temperature
    self._word_timestamps = word_timestamps
    self._max_try = max_num_retry
    self._wait_time = wait_time
    self._client = None

  def _setup(self):
    """Loads the OpenAI LLM model client."""
    if self._client is None:
      self._client = openai.OpenAI(
          api_key=self._api_key, base_url=self._server_url
      )

  def _check_input_types(self, batch: Sequence[types.MultiModalObject]) -> None:
    if not all(isinstance(x, types.Sound) for x in batch):
      raise ValueError(
          'OpenAISpeechToTextEncoder only supports a batch of all Sound inputs.'
      )

  def _encode(
      self,
      batch: Sequence[types.MultiModalObject],
  ) -> Sequence[types.SoundEmbedding]:
    """Encodes a batch of sound sources into embeddings and timestamps.

    Args:
      batch: A sequence of types.Sound objects to encode.

    Returns:
      A list of tuples, one for each input, each tuple containing:
        - waveform_embeddings (np.ndarray): A 2D array of shape
          (n, embedding_dim).
        - embedding_timestamps (np.ndarray): A 2D array of shape (m, 2),
          where each row is an [start, end] pair indicating a segment by
          sound waveform index.
          There are two common cases for the relation between embeddings (n)
          and timestamps (m):
            - Frame-Aligned (m == n): The i-th timestamp corresponds
              directly to the i-th embedding vector.
            - Utterance-Level (m == 1): A single timestamp pair represents
              the start and end of the entire audio segment from which the
              embeddings were extracted.
    """
    sound_batch: list[types.Sound] = []
    for example in batch:
      assert isinstance(example, types.Sound)
      sound_batch.append(example)
    outputs = []
    for sound in sound_batch:
      outputs.append(self._encode_sound(sound))
    return outputs

  def _encode_sound(
      self,
      sound: types.Sound,
  ) -> types.SoundEmbedding:
    """Encodes speech to text using the OpenAI Audio Transciptions API.

    The output structure of the embeddings and timestamps arrays
    depends on the word_timestamps parameter:

    1.  Sentence-level output (if word_timestamps is False):
        * timestamps: A numpy.ndarray of shape [n_segments, 2],
            where each inner element is
            [segment_start_time_seconds, segment_end_time_seconds].
        * embeddings: A numpy.ndarray of shape [n_segments] containing
            the transcribed text for each segment (strings).

    2.  Word-level output (if word_timestamps is True):
        * timestamps: A numpy.ndarray of shape [n_words, 2],
            where each inner element is
            [word_start_time_seconds, word_end_time_seconds].
        * embeddings: A numpy.ndarray of shape [n_words] containing
            the individual words (strings).

    Args:
      sound: A `Sound` object containing the audio waveform and context.

    Returns:
      A tuple (embeddings, timestamps).
      embeddings: A numpy.ndarray containing the transcribed text or words.
                  The shape is [n_segments] or [n_words] depending
                  on word_timestamps. Dtype will be object for strings.
      timestamps: A numpy.ndarray containing the start and end times.
                  The shape is [n_segments, 2] or [n_words, 2] depending
                  on word_timestamps. Dtype will be float.
    """
    assert self._client is not None

    if self._word_timestamps:
      response_format = 'verbose_json'
      timestamp_granularity = ['word']
    else:
      response_format = 'json'
      timestamp_granularity = None

    flac_file = io.BytesIO()
    flac_file.name = 'file.flac'
    soundfile.write(flac_file, sound.waveform, sound.context.sample_rate)
    transcription = None
    for n_try in range(self._max_try):
      try:
        transcription = self._client.audio.transcriptions.create(
            model=self._model_name,
            file=flac_file,
            response_format=response_format,
            temperature=self._temperature,
            timestamp_granularities=timestamp_granularity,
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception(e)
        logging.warning('Failed to get prediction, retrying %d', n_try)
        time.sleep(int(self._wait_time * 1.5 ** (n_try + 1)))
        continue

    if transcription is None:
      timestamps = np.array([[0.0, 0.0]], dtype=float)
      embeddings = np.array([types.LLM_NO_RESPONSE_STR], dtype=object)
    elif self._word_timestamps:
      timestamps_list = []
      embeddings_list = []
      for word in transcription.words:
        timestamps_list.append([word.start, word.end])
        embeddings_list.append(word.word)
      timestamps = np.array(timestamps_list, dtype=float)
      embeddings = np.array(embeddings_list, dtype=object)
    else:
      timestamps = np.array(
          [[
              sound.context.waveform_start_second,
              sound.context.waveform_end_second,
          ]],
          dtype=float,
      )
      embeddings = np.array([transcription.text], dtype=object)

    return types.SoundEmbedding(
        embedding=embeddings, timestamps=timestamps, context=sound.context
    )
