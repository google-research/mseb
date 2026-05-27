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

"""LLM encoder using the LiteLLM API with Whisper."""

from typing import Callable

from mseb import encoder
from mseb.encoders import converter
from mseb.encoders import litellm_encoder
from mseb.encoders import prompt as prompt_lib
from mseb.encoders import whisper_encoder


def LiteLLMWithTitleAndContextWhisperEncoder(
    whisper_model_path: str,
    llm_model_name: str,
    api_key: str | None = None,
    normalizer: Callable[[str], str] | None = None,
    prompt: prompt_lib.Prompt = prompt_lib.ReasoningPrompt(),
    max_num_retry: int = 1,
    wait_time: float = 1.0,
    whisper_word_timestamps: bool = False,
    output_json_alignment: bool = False,
) -> encoder.CascadeEncoder:
  """Cascaded Whisper and LLM encoder with title and context.

  Args:
    whisper_model_path: A serializable string (e.g., a GCS path or Hub ID)
      pointing to the Whisper model to be loaded in setup().
    llm_model_name: Name of the LLM model.
    api_key: API key for the OpenAI Transcriptions server.
    ditto_config_id: Ditto config ID to use for the LLM.
    normalizer: A function that normalizes the text before encoding. This is
      useful for removing special characters or formatting the text for better
      encoding results.
    prompt: Format of the prompt to be used for LLM.
    max_num_retry: The maximum number of retries for the model.
    wait_time: The wait time in seconds between retries.
    whisper_word_timestamps: Whether to output word timestamps from Whisper.
    output_json_alignment: Whether to output json alignment from Whisper. This
      is used for the case when the downstream model expects json alignment.

  Returns:
    A cascaded Whisper and LLM encoder with title and context.
  """
  return encoder.CascadeEncoder(
      encoders=[
          encoder.SpeechToTextWithTitleAndContextEncoder(
              speech_to_text_encoder=whisper_encoder.SpeechToTextEncoder(
                  model_path=whisper_model_path,
                  word_timestamps=whisper_word_timestamps,
              )
          ),
          converter.SoundEmbeddingToTextConverter(
              output_json_alignment=output_json_alignment
          ),
          litellm_encoder.LiteLLMTextEncoder(
              model_name=llm_model_name,
              api_key=api_key,
              normalizer=normalizer,
              prompt=prompt,
              max_num_retry=max_num_retry,
              wait_time=wait_time,
          ),
          converter.TextEmbeddingToTextPredictionConverter(),
      ]
  )
