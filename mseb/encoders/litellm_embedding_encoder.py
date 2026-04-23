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

"""Encoder using the LiteLLM Embedding API."""

import base64
import re
import time
from typing import Callable, Optional

from absl import flags
from absl import logging
import litellm
from mseb import encoder
from mseb import types
from mseb.encoders import converter
from mseb.encoders import prompt as prompt_lib
from mseb.encoders import text_encoder_with_prompt as prompt_encoder
import numpy as np

LITELLM_EMBEDDING_API_KEY = flags.DEFINE_string(
    'litellm_embedding_api_key',
    '',
    'API key for LiteLLM Embedding API.',
)

LITELLM_EMBEDDING_MODEL_NAME = flags.DEFINE_string(
    'litellm_embedding_model_name',
    'bedrock/amazon.nova-2-multimodal-embeddings-v1:0',
    'Name of the LiteLLM Embedding API model.',
)


class LiteLLMEmbeddingEncoder(prompt_encoder.TextEncoderWithPrompt):
  """Encode texts and sounds using the LiteLLM Embedding API."""

  def __init__(
      self,
      model_name: str,
      api_key: str,
      normalizer: Callable[[str], str] | None = None,
      prompt: prompt_lib.Prompt = prompt_lib.DefaultPrompt(),
      max_num_retry: int = 1,
      wait_time: float = 1.0,
      embedding_dim: int | None = None,
  ):
    """Initializes the OpenAI Speech-to-text encoder.

    Args:
      model_name: Name of the LiteLLM Embedding model.
      api_key: API key for the LiteLLM Embedding server.
      ditto_config_id: Ditto config ID to use for the LLM.
      normalizer: A function that normalizes the text before encoding. This is
        useful for removing special characters or formatting the text for better
        encoding results.
      prompt: Prompt object definining the format of the prompt to be used.
      max_num_retry: The maximum number of retries for the model.
      wait_time: The wait time in seconds between retries.
      embedding_dim: The dimension of the embedding vectors. If None, it will be
        inferred from the model.
    """
    super().__init__(normalizer, prompt=prompt)
    self._api_key = api_key
    self._model_name = model_name
    self._max_try = max_num_retry
    self._wait_time = wait_time
    self.embedding_dim = embedding_dim
    if self.embedding_dim is None:
      model_info = litellm.get_model_info(model=self._model_name)
      if 'output_vector_size' in model_info:
        self.embedding_dim = model_info['output_vector_size']
      else:
        raise ValueError(
            f'Model {model_name} does not have output_vector_size.'
        )

  def _setup(self):
    self.prompt_encode_fn = lambda prompts: np.array(
        [
            LiteLLMEmbeddingEncoder.get_embedding(
                prompt,
                model_name=self._model_name,
                api_key=self._api_key,
                embedding_dim=self.embedding_dim,
                max_try=self._max_try,
                wait_time=self._wait_time,
            )
            for prompt in prompts
        ],
        dtype=np.float32,
    )

  @staticmethod
  def get_embedding(
      request_prompt: tuple[str, Optional[bytes]],
      *,
      model_name: str,
      api_key: str,
      embedding_dim: int,
      max_try: int = 1,
      wait_time: float = 1.0,
  ) -> np.ndarray:
    """Returns the embeddings for the given request prompts."""
    input_content = []
    if request_prompt[0]:
      input_content.append(request_prompt[0])
    if request_prompt[1] is not None:
      audio_data = base64.b64encode(request_prompt[1]).decode('utf-8')
      input_content.append(f'data:audio/wav;base64,{audio_data}')
    response = None
    for n_try in range(max_try):
      try:
        response = litellm.embedding(
            model=model_name,
            input=input_content,
            api_key=api_key,
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception(e)
        logging.warning(
            'Failed to get embedding, retrying %d:  (%d)', n_try, max_try
        )
        time.sleep(int(wait_time * 1.5 ** (n_try + 1)))
        continue

    if response is None:
      logging.error('Failed to get embedding after %d retries.', max_try)
      return np.zeros(embedding_dim, dtype=np.float32)

    return np.array(response.data[0].embedding, dtype=np.float32)


def LiteLLMEmbeddingOrLiteLLMEmbeddingEncoder(
    model_name: str,
    api_key: str,
    normalizer_for_sound: Callable[[str], str] | None = None,
    prompt_for_sound: prompt_lib.Prompt | None = prompt_lib.DefaultPrompt(''),
    normalizer_for_text: Callable[[str], str] | None = lambda x: re.sub(
        r'\[\d+\]', '', x.lower()
    ),
    prompt_for_text: prompt_lib.Prompt | None = None,
) -> encoder.CollectionEncoder:
  """Pair Sound and Text encoder as for sound to text retrieval."""
  sound_encoder = LiteLLMEmbeddingEncoder(
      model_name=model_name,
      api_key=api_key,
      normalizer=normalizer_for_sound,
      prompt=prompt_for_sound,
  )
  text_encoder = LiteLLMEmbeddingEncoder(
      model_name=model_name,
      api_key=api_key,
      normalizer=normalizer_for_text,
      prompt=prompt_for_text,
  )
  return encoder.CollectionEncoder(
      encoder_by_input_type={
          types.Sound: sound_encoder,
          types.Text: text_encoder,
      }
  )


def LiteLLMEmbeddingTranscriptTruthEncoder(
    model_name: str,
    api_key: str,
    normalizer: Callable[[str], str] | None = None,
    prompt: prompt_lib.Prompt | None = None,
    embedding_dim: int | None = None,
) -> encoder.CascadeEncoder:
  """Cascaded transcript truth and LiteLLM API encoder.

  Args:
      model_name: Name of the LiteLLM Embedding model.
      api_key: API key for the LiteLLM Embedding server.
      ditto_config_id: Ditto config ID to use for the LLM.
      normalizer: A function that normalizes the text before encoding. This is
        useful for removing special characters or formatting the text for better
        encoding results.
      prompt: Prompt object definining the format of the prompt to be used.
      embedding_dim: The dimension of the embedding vectors. If None, it will be
        inferred from the model.

  Returns:
    A cascaded transcript truth and Google GenAI API encoder.
  """
  return encoder.CascadeEncoder(
      encoders=[
          converter.SoundToTextWithTitleAndContextConverter(),
          LiteLLMEmbeddingEncoder(
              model_name=model_name,
              api_key=api_key,
              normalizer=normalizer,
              prompt=prompt,
              embedding_dim=embedding_dim,
          ),
      ]
  )


def LiteLLMEmbeddingTranscriptTruthOrLiteLLMEmbeddingEncoder(
    model_name: str,
    api_key: str,
    normalizer_for_sound: Callable[[str], str] | None = None,
    prompt_for_sound: prompt_lib.Prompt | None = prompt_lib.DefaultPrompt(''),
    normalizer_for_text: Callable[[str], str] | None = lambda x: re.sub(
        r'\[\d+\]', '', x.lower()
    ),
    prompt_for_text: prompt_lib.Prompt | None = None,
) -> encoder.CollectionEncoder:
  """Pair Sound and Text encoder as for sound to text retrieval."""
  sound_encoder = LiteLLMEmbeddingTranscriptTruthEncoder(
      model_name=model_name,
      api_key=api_key,
      normalizer=normalizer_for_sound,
      prompt=prompt_for_sound,
  )
  text_encoder = LiteLLMEmbeddingEncoder(
      model_name=model_name,
      api_key=api_key,
      normalizer=normalizer_for_text,
      prompt=prompt_for_text,
  )
  return encoder.CollectionEncoder(
      encoder_by_input_type={
          types.Sound: sound_encoder,
          types.Text: text_encoder,
      }
  )
