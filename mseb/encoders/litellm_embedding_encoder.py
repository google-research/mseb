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

"""Encoder using the LiteLLM Embedding API.
"""

import base64
import time
from typing import Sequence

from absl import logging
import litellm
from mseb import encoder
from mseb import types
from mseb import utils
import numpy as np


class LiteLLMEmbeddingEncoder(encoder.MultiModalEncoder):
  """Encode texts and sounds using the LiteLLM Embedding API."""

  def __init__(
      self,
      model_name: str,
      api_key: str,
      max_num_retry: int = 1,
      wait_time: float = 1.0,
  ):
    """Initializes the OpenAI Speech-to-text encoder.

    Args:
      model_name: Name of the LiteLLM Embedding model.
      api_key: API key for the LiteLLM Embedding server.
      max_num_retry: The maximum number of retries for the model.
      wait_time: The wait time in seconds between retries.
    """
    super().__init__()
    self._api_key = api_key
    self._model_name = model_name
    self._max_try = max_num_retry
    self._wait_time = wait_time
    self._client = None

  def _setup(self):
    pass

  def _check_input_types(self, batch: Sequence[types.MultiModalObject]) -> None:
    if not all(isinstance(x, types.Sound) for x in batch) and not all(
        isinstance(x, types.Text) for x in batch
    ):
      raise ValueError(
          'LiteLLMEmbeddingEncoder only supports a batch of all Sound'
          ' or all Text inputs.'
      )

  def _encode(
      self,
      batch: Sequence[types.MultiModalObject],
  ) -> Sequence[types.SoundEmbedding | types.TextEmbedding]:
    """Encodes a batch of all Sound or all Text inputs."""

    if all(isinstance(x, types.Sound) for x in batch):
      return self._encode_sound_batch(batch)
    elif all(isinstance(x, types.Text) for x in batch):
      return self._encode_text_batch(batch)
    else:
      raise ValueError(
          'LiteLLMEmbeddingEncoder only supports a batch of all Sound'
          ' or all Text inputs.'
      )

  def _encode_text_batch(
      self,
      batch: Sequence[types.Text],
  ) -> Sequence[types.TextEmbedding]:
    """Encodes a batch of Text inputs."""
    embeddings = self._encode_batch([text.text for text in batch])
    outputs = []
    for i, text in enumerate(batch):
      outputs.append(
          types.TextEmbedding(
              embedding=embeddings[i],
              spans=np.array([[0, len(text.text)]]),
              context=text.context,
          )
      )
    return outputs

  def _encode_sound_batch(
      self,
      batch: Sequence[types.Sound],
  ) -> Sequence[types.SoundEmbedding]:
    """Encodes a batch of Sound inputs."""
    inputs = []
    for sound in batch:
      wav_bytes = utils.sound_to_wav_bytes(sound)
      audio_data = base64.b64encode(wav_bytes).decode('utf-8')
      inputs.append(f'data:audio/wav;base64,{audio_data}')
    embeddings = self._encode_batch(inputs)
    outputs = []
    for i, sound in enumerate(batch):
      end_time = sound.context.length / sound.context.sample_rate
      timestamps = np.array([[0.0, end_time]])
      outputs.append(
          types.SoundEmbedding(
              embedding=embeddings[i],
              timestamps=timestamps,
              context=sound.context,
          )
      )
    return outputs

  def _encode_batch(
      self,
      input_batch: Sequence[str],
  ) -> Sequence[np.ndarray]:
    """Encodes a batch of texts or sounds into embeddings."""
    response = None
    for n_try in range(self._max_try):
      try:
        response = litellm.embedding(
            model=self._model_name,
            input=input_batch,
            api_key=self._api_key,
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception(e)
        logging.warning('Failed to get prediction, retrying %d', n_try)
        time.sleep(int(self._wait_time * 1.5 ** (n_try + 1)))
        continue

    if response is None:
      return [np.array([types.LLM_NO_RESPONSE_STR], dtype=object)] * len(
          input_batch
      )

    return [np.array(data.embedding, dtype=float) for data in response.data]
