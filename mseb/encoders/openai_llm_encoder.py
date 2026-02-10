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

"""Gemma reasoning encoder."""

import base64
import logging
import re
import time
from typing import Callable, Optional, Tuple

from mseb import encoder
from mseb.encoders import converter
from mseb.encoders import prompt as prompt_lib
from mseb.encoders import retrieval_encoder
from mseb.encoders import text_encoder_with_prompt as prompt_encoder
from mseb.encoders import whisper_encoder
from mseb.evaluators import reasoning_evaluator
import numpy as np
import openai


class OpenAILLMEncoder(prompt_encoder.TextEncoderWithPrompt):
  """Encoder with OpenAI-API-compatible remote LLM model."""

  NO_RESPONSE_STR = reasoning_evaluator.NO_RESPONSE_STR

  def __init__(
      self,
      server_url: str,
      model_name: str,
      api_key: str,
      normalizer: Callable[[str], str] | None = None,
      prompt: prompt_lib.Prompt = prompt_lib.DefaultPrompt(),
      max_num_retry: int = 1,
      wait_time: float = 1.0,
  ):
    """Initializes the LLM-based encoder.

    Args:
      server_url: URL of the LLM server.
      model_name: Name of the LLM model.
      api_key: API key of the LLM model server.
      normalizer: A function that normalizes the text before encoding. This is
        useful for removing special characters or formatting the text for better
        encoding results.
      prompt: Prompt object definining the format of the prompt to be used for
        the LLM.
      max_num_retry: The maximum number of retries for the model.
      wait_time: The wait time in seconds between retries.
    """
    super().__init__(normalizer, prompt=prompt)
    self._server_url = server_url
    self._api_key = api_key
    self._model_name = model_name
    self._max_try = max_num_retry
    self._wait_time = wait_time
    self._client = None

  def _setup(self):
    """Loads the OpenAI LLM model client."""
    if self._client is None:
      self._client = openai.OpenAI(
          api_key=self._api_key, base_url=self._server_url
      )
    self.prompt_encode_fn = lambda prompts: np.array([
        OpenAILLMEncoder.get_response(
            prompt,
            client=self._client,
            model_name=self._model_name,
            max_try=self._max_try,
            wait_time=self._wait_time,
        )
        for prompt in prompts
    ])

  @staticmethod
  def get_response(
      request_prompt: Tuple[str, Optional[bytes]],
      *,
      client: openai.OpenAI,
      model_name: str,
      max_try: int = 1,
      wait_time: float = 1.0,
  ) -> str:
    """Returns the prediction for the given question, title and context."""
    assert client is not None, 'Client is not initialized.'
    messages = [
        {'role': 'user', 'content': [
            {
                'type': 'text',
                'text': request_prompt[0],
            },
        ]}
    ]
    if request_prompt[1] is not None:
      messages[0]['content'].append({  # pytype: disable=attribute-error
          'type': 'input_audio',
          'input_audio': {
              'data': base64.b64encode(request_prompt[1]).decode('utf-8'),
              'format': 'wav',
          },
      })

    for n_try in range(max_try):
      try:
        client_response = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        response = client_response.choices[0].message.content.strip()
      except Exception as _:  # pylint: disable=broad-exception-caught
        logging.warning('Failed to get prediction, retrying %d', n_try)
        time.sleep(int(wait_time * 1.5 ** (n_try + 1)))
        continue

      # Remove markdown and json formatting.
      response = re.sub(r'```json\s*|\s*```', '', response)
      response = re.sub(r'JSON\s*', '', response)

      return response

    return OpenAILLMEncoder.NO_RESPONSE_STR


def OpenAILLMWithTitleAndContextTranscriptTruthEncoder(
    server_url: str,
    model_name: str,
    api_key: str,
    normalizer: Callable[[str], str] | None = None,
    prompt: prompt_lib.Prompt = prompt_lib.ReasoningPrompt(),
) -> encoder.CascadeEncoder:
  """Cascaded transcript truth and Gemma encoder.

  Args:
    server_url: URL of the LLM server.
    model_name: Name of the LLM model.
    api_key: API key of the LLM model server.
    normalizer: A function that normalizes the text before encoding. This is
      useful for removing special characters or formatting the text for better
      encoding results.
    prompt: Format of the prompt to be used for Gemma.

  Returns:
    A cascaded transcript truth and Gemma encoder.
  """
  return encoder.CascadeEncoder(
      encoders=[
          converter.SoundToSoundEmbeddingConverter(),
          converter.SoundEmbeddingToTextConverter(),
          OpenAILLMEncoder(
              server_url=server_url,
              model_name=model_name,
              api_key=api_key,
              normalizer=normalizer,
              prompt=prompt,
          ),
          converter.TextEmbeddingToTextPredictionConverter(),
      ]
  )


def RagOpenAILLMWithTitleAndContextTranscriptTruthEncoder(
    server_url: str,
    model_name: str,
    api_key: str,
    rag_encoder: encoder.MultiModalEncoder,
    top_k: int = 10,
    normalizer: Callable[[str], str] | None = None,
    prompt: prompt_lib.Prompt = prompt_lib.RetrievalPrompt(),
) -> encoder.CascadeEncoder:
  """Cascaded transcript truth and RAG Gemma encoder."""
  return encoder.CascadeEncoder(
      encoders=[
          converter.SoundToSoundEmbeddingConverter(),
          converter.SoundEmbeddingToTextConverter(),
          rag_encoder,
          retrieval_encoder.RetrievalEncoder(top_k=top_k),
          OpenAILLMEncoder(
              server_url=server_url,
              model_name=model_name,
              api_key=api_key,
              normalizer=normalizer,
              prompt=prompt
          ),
          converter.TextEmbeddingToTextPredictionConverter(),
      ]
  )


def OpenAILLMWithTitleAndContextWhisperEncoder(
    whisper_model_path: str,
    llm_server_url: str,
    llm_model_name: str,
    llm_api_key: str,
    normalizer: Callable[[str], str] | None = None,
    prompt: prompt_lib.Prompt = prompt_lib.ReasoningPrompt(),
    max_num_retry: int = 1,
    wait_time: float = 1.0,
) -> encoder.CascadeEncoder:
  """Cascaded Whisper and Gemma encoder with title and context.

  Args:
    whisper_model_path: A serializable string (e.g., a GCS path or Hub ID)
      pointing to the Whisper model to be loaded in setup().
    llm_server_url: URL of the LLM server.
    llm_model_name: Name of the LLM model.
    llm_api_key: API key of the LLM model server.
    normalizer: A function that normalizes the text before encoding. This is
      useful for removing special characters or formatting the text for better
      encoding results.
    prompt: Format of the prompt to be used for Gemma.
    max_num_retry: The maximum number of retries for the model.
    wait_time: The wait time in seconds between retries.

  Returns:
    A cascaded Whisper and Gemma encoder with title and context.
  """
  return encoder.CascadeEncoder(
      encoders=[
          encoder.SpeechToTextWithTitleAndContextEncoder(
              speech_to_text_encoder=whisper_encoder.SpeechToTextEncoder(
                  model_path=whisper_model_path
              )
          ),
          converter.SoundEmbeddingToTextConverter(),
          OpenAILLMEncoder(
              server_url=llm_server_url,
              model_name=llm_model_name,
              api_key=llm_api_key,
              normalizer=normalizer,
              prompt=prompt,
              max_num_retry=max_num_retry,
              wait_time=wait_time,
          ),
          converter.TextEmbeddingToTextPredictionConverter(),
      ]
  )


def RagOpenAILLMWithTitleAndContextWhisperEncoder(
    whisper_model_path: str,
    llm_server_url: str,
    llm_model_name: str,
    llm_api_key: str,
    rag_encoder: encoder.MultiModalEncoder,
    top_k: int = 10,
    normalizer: Callable[[str], str] | None = None,
    prompt: prompt_lib.Prompt = prompt_lib.RetrievalPrompt(),
    max_num_retry: int = 1,
    wait_time: float = 1.0,
) -> encoder.CascadeEncoder:
  """RAG Gemma encoder with title and context."""

  return encoder.CascadeEncoder(
      encoders=[
          encoder.SpeechToTextWithTitleAndContextEncoder(
              speech_to_text_encoder=whisper_encoder.SpeechToTextEncoder(
                  model_path=whisper_model_path
              )
          ),
          converter.SoundEmbeddingToTextConverter(),
          rag_encoder,
          retrieval_encoder.RetrievalEncoder(top_k=top_k),
          OpenAILLMEncoder(
              server_url=llm_server_url,
              model_name=llm_model_name,
              api_key=llm_api_key,
              normalizer=normalizer,
              prompt=prompt,
              max_num_retry=max_num_retry,
              wait_time=wait_time,
          ),
          converter.TextEmbeddingToTextPredictionConverter(),
      ]
  )


def OpenAILLMWithTitleAndContextEncoder(
    server_url: str,
    model_name: str,
    api_key: str,
    normalizer: Callable[[str], str] | None = None,
    prompt: prompt_lib.Prompt = prompt_lib.ReasoningPrompt(),
    max_num_retry: int = 1,
    wait_time: float = 1.0,
) -> encoder.CascadeEncoder:
  """E2E Gemma encoder with title and context."""

  return encoder.CascadeEncoder(
      encoders=[
          OpenAILLMEncoder(
              server_url=server_url,
              model_name=model_name,
              api_key=api_key,
              normalizer=normalizer,
              prompt=prompt,
              max_num_retry=max_num_retry,
              wait_time=wait_time,
          ),
          converter.SoundEmbeddingToTextConverter(),
          converter.TextToTextPredictionConverter(),
      ]
  )


def RagOpenAILLMWithTitleAndContextEncoder(
    server_url: str,
    model_name: str,
    api_key: str,
    rag_encoder: encoder.MultiModalEncoder,
    top_k: int = 10,
    normalizer: Callable[[str], str] | None = None,
    prompt: prompt_lib.Prompt = prompt_lib.RetrievalPrompt(),
    max_num_retry: int = 1,
    wait_time: float = 1.0,
) -> encoder.CascadeEncoder:
  """RAG Gemma encoder with title and context."""

  return encoder.CascadeEncoder(
      encoders=[
          rag_encoder,
          retrieval_encoder.RetrievalEncoder(top_k=top_k),
          OpenAILLMEncoder(
              server_url=server_url,
              model_name=model_name,
              api_key=api_key,
              normalizer=normalizer,
              prompt=prompt,
              max_num_retry=max_num_retry,
              wait_time=wait_time,
          ),
          converter.SoundEmbeddingToTextConverter(),
          converter.TextToTextPredictionConverter(),
      ]
  )
