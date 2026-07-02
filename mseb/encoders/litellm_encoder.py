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

"""LLM encoder using the LiteLLM API."""

import base64
import logging
import re
import time
from typing import Callable, Optional, Tuple

from absl import flags
import litellm
from mseb import encoder
from mseb import types
from mseb.encoders import converter
from mseb.encoders import prompt as prompt_lib
from mseb.encoders import text_encoder_with_prompt as prompt_encoder
import numpy as np


LITELLM_API_KEY = flags.DEFINE_string(
    'litellm_api_key',
    '',
    'API key for LiteLLM API.',
)

LITELLM_MODEL_NAME = flags.DEFINE_string(
    'litellm_model_name',
    '',
    'Name of the LiteLLM API model in the format of provider/model_name.'
)


class LiteLLMTextEncoder(prompt_encoder.TextEncoderWithPrompt):
  """LLM text encoder using  LiteLLM API."""

  NO_RESPONSE_STR = types.LLM_NO_RESPONSE_STR

  def __init__(
      self,
      model_name: str,
      api_key: str | None = None,
      normalizer: Callable[[str], str] | None = None,
      prompt: prompt_lib.Prompt = prompt_lib.DefaultPrompt(),
      max_num_retry: int = 1,
      wait_time: float = 1.0,
  ):
    """Initializes the client.

    Args:
      model_name: Name of the LLM model.
      api_key: API key for the OpenAI Transcriptions server.
      ditto_config_id: Ditto config ID to use for the LLM.
      normalizer: A function that normalizes the text before encoding. This is
        useful for removing special characters or formatting the text for better
        encoding results.
      prompt: Prompt object definining the format of the prompt to be used for
        the LLM.
      max_num_retry: The maximum number of retries for the model.
      wait_time: The wait time in seconds between retries.
    """
    super().__init__(normalizer, prompt=prompt)
    self._model_name = model_name
    self._api_key = api_key
    self._max_try = max_num_retry
    self._wait_time = wait_time

  def _setup(self):
    """Loads the Lite LLM client."""
    self.prompt_encode_fn = lambda prompts: np.array([
        LiteLLMTextEncoder.get_response(
            prompt,
            model_name=self._model_name,
            api_key=self._api_key,
            max_try=self._max_try,
            wait_time=self._wait_time,
        )
        for prompt in prompts
    ])

  @staticmethod
  def get_response(
      request_prompt: Tuple[str, Optional[bytes]],
      *,
      model_name: str,
      api_key: str | None = None,
      max_try: int = 1,
      wait_time: float = 1.0,
  ) -> str:
    """Returns the prediction for the given question, title and context."""
    messages = [{
        'role': 'user',
        'content': [
            {
                'type': 'text',
                'text': request_prompt[0],
            },
        ],
    }]
    if request_prompt[1] is not None:
      messages[0]['content'].append({  # pytype: disable=attribute-error
          'type': 'input_audio',
          'input_audio': {  # pyrefly: ignore[bad-assignment]
              'data': base64.b64encode(request_prompt[1]).decode('utf-8'),
              'format': 'wav',
          },
      })

    for n_try in range(max_try):
      try:
        client_response = litellm.completion(
            model=model_name,
            messages=messages,
            api_key=api_key,
        )
        response = client_response.choices[0].message.content.strip()
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning('Failed to get prediction: %s, retrying %d: ', e, n_try)
        time.sleep(int(wait_time * 1.5 ** (n_try + 1)))
        continue

      # Remove markdown and json formatting.
      response = re.sub(r'```json\s*|\s*```', '', response)
      response = re.sub(r'JSON\s*', '', response)

      print('DEBUG_response: ', response)
      return response

    return LiteLLMTextEncoder.NO_RESPONSE_STR


def LiteLLMWithTitleAndContextTranscriptTruthEncoder(
    model_name: str,
    api_key: str | None = None,
    normalizer: Callable[[str], str] | None = None,
    prompt: prompt_lib.Prompt = prompt_lib.ReasoningPrompt(),
) -> encoder.CascadeEncoder:
  """Cascaded transcript truth and LLM encoder.

  Args:
    model_name: Name of the LLM model.
    api_key: API key for the OpenAI Transcriptions server.
    ditto_config_id: Ditto config ID to use for the LLM.
    normalizer: A function that normalizes the text before encoding. This is
      useful for removing special characters or formatting the text for better
      encoding results.
    prompt: Format of the prompt to be used for LLM.

  Returns:
    A cascaded transcript truth and LLM encoder.
  """
  return encoder.CascadeEncoder(
      encoders=[
          converter.SoundToSoundEmbeddingConverter(),
          converter.SoundEmbeddingToTextConverter(),
          LiteLLMTextEncoder(
              model_name=model_name,
              api_key=api_key,
              normalizer=normalizer,
              prompt=prompt,
          ),
          converter.TextEmbeddingToTextPredictionConverter(),
      ]
  )


def LiteLLMWithTitleAndContextEncoder(
    model_name: str,
    api_key: str | None = None,
    normalizer: Callable[[str], str] | None = None,
    prompt: prompt_lib.Prompt = prompt_lib.ReasoningPrompt(),
    max_num_retry: int = 1,
    wait_time: float = 1.0,
) -> encoder.CascadeEncoder:
  """E2E LLM encoder with title and context."""

  return encoder.CascadeEncoder(
      encoders=[
          LiteLLMTextEncoder(
              model_name=model_name,
              api_key=api_key,
              normalizer=normalizer,
              prompt=prompt,
              max_num_retry=max_num_retry,
              wait_time=wait_time,
          ),
          converter.TextEmbeddingToTextPredictionConverter(),
      ]
  )
