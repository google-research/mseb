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

"""LLM encoders using the Google GenAI API."""

import logging
import re
import time
from typing import Callable, Optional, Tuple

from absl import flags
from google import genai
from mseb import encoder
from mseb import types
from mseb.encoders import converter
from mseb.encoders import prompt as prompt_lib
from mseb.encoders import text_encoder_with_prompt as prompt_encoder
import numpy as np


GENAI_LLM_ENCODER_MODEL_PATH = flags.DEFINE_string(
    'genai_llm_encoder_model_path',
    'gemma-3-27b-it',
    'Path to GenAI model.',
)

GENAI_LLM_ENCODER_GEMINI_API_KEY = flags.DEFINE_string(
    'genai_llm_encoder_gemini_api_key',
    '',
    'API key for Gemini API.',
)


class GenaiLLMTextEncoder(prompt_encoder.TextEncoderWithPrompt):
  """LLM encoder using the Google GenAI API."""

  NO_RESPONSE_STR = types.LLM_NO_RESPONSE_STR

  def __init__(
      self,
      model_path: str,
      api_key: str,
      normalizer: Callable[[str], str] | None = None,
      prompt: prompt_lib.Prompt = prompt_lib.DefaultPrompt(),
      max_num_retry: int = 1,
      wait_time: float = 1.0,
  ):
    """Initializes the client.

    Args:
      model_path: A serializable string (e.g., a GCS path or Hub ID) pointing to
        the model to be loaded in setup().
      api_key: API key for the Gemini API.
      normalizer: A function that normalizes the text before encoding. This is
        useful for removing special characters or formatting the text for better
        encoding results.
      prompt: Prompt object definining the format of the prompt to be used.
      max_num_retry: The maximum number of retries for the model.
      wait_time: The wait time in seconds between retries.
    """
    super().__init__(normalizer, prompt=prompt)
    self._model_path = model_path
    self._api_key = api_key
    self._max_try = max_num_retry
    self._wait_time = wait_time
    self._client = None

  def _setup(self):
    """Initializes the client."""
    if self._client is None:
      logging.info('Initializing client: %s', self._model_path)
      self._client = genai.Client(api_key=self._api_key)
      logging.info('Client initialized: %s', self._model_path)
    self.prompt_encode_fn = lambda prompts: np.array([
        GenaiLLMTextEncoder.get_response(
            prompt,
            client=self._client,
            model_path=self._model_path,
            max_try=self._max_try,
            wait_time=self._wait_time,
        )
        for prompt in prompts
    ])

  @staticmethod
  def get_response(
      request_prompt: Tuple[str, Optional[bytes]],
      *,
      client: genai.Client,
      model_path: str,
      max_try: int = 1,
      wait_time: float = 1.0,
  ) -> str:
    """Returns the prediction for the given question, title and context."""
    assert client is not None, 'Client is not initialized.'
    prompt_content = [request_prompt[0]]
    if request_prompt[1] is not None:
      prompt_content.append(
          genai.types.Part.from_bytes(
              data=request_prompt[1], mime_type='audio/wav'
          )
      )
    for n_try in range(max_try):
      try:
        response = client.models.generate_content(
            model=model_path, contents=prompt_content
        ).text
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning('Failed to get prediction: %s, retrying %d: ', e, n_try)
        time.sleep(int(wait_time * 1.5 ** (n_try + 1)))
        continue

      if response:
        response = response.strip()
        # Remove markdown and json formatting.
        response = re.sub(r'```json\s*|\s*```', '', response)
        response = re.sub(r'JSON\s*', '', response)

      return response

    return GenaiLLMTextEncoder.NO_RESPONSE_STR


def GenaiLLMEncoder(
    model_path: str,
    api_key: str,
    normalizer: Callable[[str], str] | None = None,
    prompt: prompt_lib.Prompt = prompt_lib.ReasoningPrompt(),
) -> encoder.CascadeEncoder:
  """Encodes sound to text using the Google GenAI API.

  Args:
    model_path: A serializable string (e.g., a GCS path or Hub ID) pointing to
      the model to be loaded in setup().
    api_key: API key for the Gemini API.
    normalizer: A function that normalizes the text before encoding. This is
      useful for removing special characters or formatting the text for better
      encoding results.
    prompt: Prompt object definining the format of the prompt to be used.

  Returns:
    An encoder that encodes sound to text using the Google GenAI API.
  """
  return encoder.CascadeEncoder(
      encoders=[
          GenaiLLMTextEncoder(
              model_path=model_path,
              api_key=api_key,
              normalizer=normalizer,
              prompt=prompt,
          ),
          converter.TextEmbeddingToTextPredictionConverter(),
      ]
  )


def GenaiLLMTranscriptTruthEncoder(
    model_path: str,
    api_key: str,
    normalizer: Callable[[str], str] | None = None,
    prompt: prompt_lib.Prompt = prompt_lib.ReasoningPrompt(),
) -> encoder.CascadeEncoder:
  """Cascaded transcript truth and Google GenAI API encoder.

  Args:
    model_path: A serializable string (e.g., a GCS path or Hub ID) pointing to
      the model to be loaded in setup().
    api_key: API key for the Gemini API.
    normalizer: A function that normalizes the text before encoding. This is
      useful for removing special characters or formatting the text for better
      encoding results.
    prompt: Prompt object definining the format of the prompt to be used.

  Returns:
    A cascaded transcript truth and Google GenAI API encoder.
  """
  return encoder.CascadeEncoder(
      encoders=[
          converter.SoundToTextWithTitleAndContextConverter(),
          GenaiLLMTextEncoder(
              model_path=model_path,
              api_key=api_key,
              normalizer=normalizer,
              prompt=prompt,
          ),
          converter.TextEmbeddingToTextPredictionConverter(),
      ]
  )
