# Copyright 2025 The MSEB Authors.
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

"""LLM encoder."""

import logging
import re
import time
from typing import Callable, Optional, Tuple

from absl import flags
from google import genai
from mseb import encoder
from mseb.encoders import converter
from mseb.encoders import prompt as prompt_lib
from mseb.encoders import retrieval_encoder
from mseb.encoders import text_encoder_with_prompt as prompt_encoder
from mseb.encoders import whisper_encoder
from mseb.evaluators import reasoning_evaluator
import numpy as np

_GEMINI_API_KEY = flags.DEFINE_string(
    'gemini_api_key',
    '',
    'API key for Gemini API.',
)


class GemmaTextEncoder(prompt_encoder.TextEncoderWithPrompt):
  """Text encoder with Gemma model."""

  NO_RESPONSE_STR = reasoning_evaluator.NO_RESPONSE_STR

  def __init__(
      self,
      model_path: str,
      normalizer: Callable[[str], str] | None = None,
      prompt: prompt_lib.Prompt = prompt_lib.DefaultPrompt(),
      max_num_retry: int = 1,
      wait_time: float = 1.0,
  ):
    """Initializes the Gemma model.

    Args:
      model_path: A serializable string (e.g., a GCS path or Hub ID) pointing to
        the model to be loaded in setup().
      normalizer: A function that normalizes the text before encoding. This is
        useful for removing special characters or formatting the text for better
        encoding results.
      prompt: Prompt object definining the format of the prompt to be used for
        Gemma.
      max_num_retry: The maximum number of retries for the model.
      wait_time: The wait time in seconds between retries.
    """
    super().__init__(normalizer, prompt=prompt)
    self._model_path = model_path
    self._max_try = max_num_retry
    self._wait_time = wait_time
    self._client = None

  def _setup(self):
    """Loads the Gemma model."""
    if self._client is None:
      logging.info('Connecting to Gemma at: %s', self._model_path)
      self._client = genai.Client(api_key=_GEMINI_API_KEY.value)
    self.prompt_encode_fn = lambda prompts: np.array([
        GemmaTextEncoder.get_response(
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
      except Exception as _:  # pylint: disable=broad-exception-caught
        logging.warning('Failed to get prediction, retrying %d', n_try)
        time.sleep(int(wait_time * 1.5 ** (n_try + 1)))
        continue

      if response:
        response = response.strip()
        # Remove markdown and json formatting.
        response = re.sub(r'```json\s*|\s*```', '', response)
        response = re.sub(r'JSON\s*', '', response)

      return response

    return GemmaTextEncoder.NO_RESPONSE_STR


def GemmaWithTitleAndContextTranscriptTruthEncoder(
    model_path: str,
    normalizer: Callable[[str], str] | None = None,
    prompt: prompt_lib.Prompt = prompt_lib.ReasoningPrompt(),
) -> encoder.CascadeEncoder:
  """Cascaded transcript truth and Gemma encoder.

  Args:
    model_path: BNS address of the Gemma model.
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
          GemmaTextEncoder(
              model_path=model_path,
              normalizer=normalizer,
              prompt=prompt,
          ),
          converter.TextEmbeddingToTextPredictionConverter(),
      ]
  )


def RagGemmaWithTitleAndContextTranscriptTruthEncoder(
    model_path: str,
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
          GemmaTextEncoder(
              model_path=model_path, normalizer=normalizer, prompt=prompt
          ),
          converter.TextEmbeddingToTextPredictionConverter(),
      ]
  )


def GemmaWithTitleAndContextWhisperEncoder(
    whisper_model_path: str,
    gemma_model_path: str,
    normalizer: Callable[[str], str] | None = None,
    prompt: prompt_lib.Prompt = prompt_lib.ReasoningPrompt(),
    max_num_retry: int = 1,
    wait_time: float = 1.0,
) -> encoder.CascadeEncoder:
  """Cascaded Whisper and Gemma encoder with title and context.

  Args:
    whisper_model_path: A serializable string (e.g., a GCS path or Hub ID)
      pointing to the Whisper model to be loaded in setup().
    gemma_model_path: URL of the Gemma model.
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
          GemmaTextEncoder(
              model_path=gemma_model_path,
              normalizer=normalizer,
              prompt=prompt,
              max_num_retry=max_num_retry,
              wait_time=wait_time,
          ),
          converter.TextEmbeddingToTextPredictionConverter(),
      ]
  )


def RagGemmaWithTitleAndContextWhisperEncoder(
    whisper_model_path: str,
    gemma_model_path: str,
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
          GemmaTextEncoder(
              model_path=gemma_model_path,
              normalizer=normalizer,
              prompt=prompt,
              max_num_retry=max_num_retry,
              wait_time=wait_time,
          ),
          converter.TextEmbeddingToTextPredictionConverter(),
      ]
  )


def GemmaWithTitleAndContextEncoder(
    model_path: str,
    normalizer: Callable[[str], str] | None = None,
    prompt: prompt_lib.Prompt = prompt_lib.ReasoningPrompt(),
    max_num_retry: int = 1,
    wait_time: float = 1.0,
) -> encoder.CascadeEncoder:
  """E2E Gemma encoder with title and context."""

  return encoder.CascadeEncoder(
      encoders=[
          GemmaTextEncoder(
              model_path=model_path,
              normalizer=normalizer,
              prompt=prompt,
              max_num_retry=max_num_retry,
              wait_time=wait_time,
          ),
          converter.SoundEmbeddingToTextConverter(),
          converter.TextToTextPredictionConverter(),
      ]
  )


def RagGemmaWithTitleAndContextEncoder(
    model_path: str,
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
          GemmaTextEncoder(
              model_path=model_path,
              normalizer=normalizer,
              prompt=prompt,
              max_num_retry=max_num_retry,
              wait_time=wait_time,
          ),
          converter.SoundEmbeddingToTextConverter(),
          converter.TextToTextPredictionConverter(),
      ]
  )
