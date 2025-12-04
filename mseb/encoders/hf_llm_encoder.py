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

"""Hugging Face Transformers LLM encoder."""

import logging
import os
import re
import tempfile
from typing import Any, Callable, Optional, Tuple

from mseb import encoder
from mseb.encoders import converter
from mseb.encoders import prompt as prompt_lib
from mseb.encoders import retrieval_encoder
from mseb.encoders import text_encoder_with_prompt as prompt_encoder
from mseb.encoders import whisper_encoder
import numpy as np
import transformers


class HFLLMEncoder(prompt_encoder.TextEncoderWithPrompt):
  """Encoder using a HF Transformers LLM model."""

  def __init__(
      self,
      model_path: str,
      normalizer: Callable[[str], str] | None = None,
      prompt: prompt_lib.Prompt = prompt_lib.DefaultPrompt(),
      device_map: Any = 'auto',
      torch_dtype: Any = 'auto',
      max_new_tokens: int = 128,
  ):
    """Initializes the HF LLM encoder.

    Args:
      model_path: A serializable string pointing to the model to be loaded in
        setup().
      normalizer: A function that normalizes the text before encoding. This is
        useful for removing special characters or formatting the text for better
        encoding results.
      prompt: Prompt object definining the format of the prompt to be used for
        the model.
      device_map: Device map to use for the model.
      torch_dtype: Torch dtype to use for the model.
      max_new_tokens: Maximum number of new tokens to generate.
    """
    super().__init__(normalizer, prompt=prompt)
    self._model_path = model_path
    self._processor = None
    self._model = None
    self._device_map = device_map
    self._torch_dtype = torch_dtype
    self._max_new_tokens = max_new_tokens

  def _setup(self):
    """Loads the HF Transformers model."""
    if self._model is None:
      logging.info('Loading model at: %s', self._model_path)
      self._processor = transformers.AutoProcessor.from_pretrained(
          self._model_path
      )
      self._model = transformers.AutoModelForImageTextToText.from_pretrained(
          self._model_path,
          device_map=self._device_map,
          torch_dtype=self._torch_dtype,
      )

    self.prompt_encode_fn = lambda prompts: np.array([
        HFLLMEncoder.get_response(
            prompt,
            processor=self._processor,
            model=self._model,
            max_new_tokens=self._max_new_tokens,
        )
        for prompt in prompts
    ])

  @staticmethod
  def get_response(
      request_prompt: Tuple[str, Optional[bytes]],
      *,
      processor: transformers.AutoProcessor,
      model: transformers.AutoModelForImageTextToText,
      max_new_tokens: int,
  ) -> str:
    """Returns the prediction for the given question, title and context."""
    assert processor is not None, 'Processor is not initialized.'
    assert model is not None, 'Model is not initialized.'
    messages = list([
        {
            'role': 'user',
            'content': list(),
        }
    ])
    messages[0]['content'].append(  # pylint: disable=attribute-error
        {'type': 'text', 'text': request_prompt[0]}
    )
    tmp_wavefile = None
    if request_prompt[1] is not None:
      tmp_wavefile = tempfile.NamedTemporaryFile(
          mode='wb', delete=False, delete_on_close=False
      )

      try:
        tmp_wavefile.write(request_prompt[1])
      finally:
        tmp_wavefile.close()

      messages[0]['content'].append(  # pylint: disable=attribute-error
          {'type': 'audio', 'path': tmp_wavefile.name}
      )

    input_ids = processor.apply_chat_template(  # pylint: disable=attribute-error
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors='pt',
    )
    input_len = input_ids['input_ids'].shape[-1]
    input_ids = input_ids.to(model.device, dtype=model.dtype)

    if tmp_wavefile is not None:
      os.unlink(tmp_wavefile.name)

    # Generate output from the model
    outputs = model.generate(**input_ids, max_new_tokens=max_new_tokens)

    # decode and print the output as text
    text = processor.batch_decode(  # pylint: disable=attribute-error
        outputs[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    response = text[0]

    # Remove markdown and json formatting.
    response = re.sub(r'```json\s*|\s*```', '', response)
    response = re.sub(r'JSON\s*', '', response)

    return response


def HFLLMWithTitleAndContextTranscriptTruthEncoder(
    model_path: str,
    normalizer: Callable[[str], str] | None = None,
    prompt: prompt_lib.Prompt = prompt_lib.ReasoningPrompt(),
) -> encoder.CascadeEncoder:
  """Cascaded transcript truth and HF LLM encoder.

  Args:
    model_path: path to the HF LLM model.
    normalizer: A function that normalizes the text before encoding. This is
      useful for removing special characters or formatting the text for better
      encoding results.
    prompt: Format of the prompt to be used for HF LLM.

  Returns:
    A cascaded transcript truth and HF LLM encoder.
  """
  return encoder.CascadeEncoder(
      encoders=[
          converter.SoundToSoundEmbeddingConverter(),
          converter.SoundEmbeddingToTextConverter(),
          HFLLMEncoder(
              model_path=model_path,
              normalizer=normalizer,
              prompt=prompt,
          ),
          converter.TextEmbeddingToTextPredictionConverter(),
      ]
  )


def RagHFLLMWithTitleAndContextTranscriptTruthEncoder(
    model_path: str,
    rag_encoder: encoder.MultiModalEncoder,
    top_k: int = 10,
    normalizer: Callable[[str], str] | None = None,
    prompt: prompt_lib.Prompt = prompt_lib.RetrievalPrompt(),
) -> encoder.CascadeEncoder:
  """Cascaded transcript truth and RAG HF LLM encoder."""
  return encoder.CascadeEncoder(
      encoders=[
          converter.SoundToSoundEmbeddingConverter(),
          converter.SoundEmbeddingToTextConverter(),
          rag_encoder,
          retrieval_encoder.RetrievalEncoder(top_k=top_k),
          HFLLMEncoder(
              model_path=model_path, normalizer=normalizer, prompt=prompt
          ),
          converter.TextEmbeddingToTextPredictionConverter(),
      ]
  )


def HFLLMWithTitleAndContextWhisperEncoder(
    whisper_model_path: str,
    hf_llm_model_path: str,
    normalizer: Callable[[str], str] | None = None,
    prompt: prompt_lib.Prompt = prompt_lib.ReasoningPrompt(),
) -> encoder.CascadeEncoder:
  """Cascaded Whisper and HF LLM encoder with title and context.

  Args:
    whisper_model_path: A serializable string (e.g., a GCS path or Hub ID)
      pointing to the Whisper model to be loaded in setup().
    hf_llm_model_path: path to the HF LLM model.
    normalizer: A function that normalizes the text before encoding. This is
      useful for removing special characters or formatting the text for better
      encoding results.
    prompt: Format of the prompt to be used for the HF LLM.

  Returns:
    A cascaded Whisper and HF LLM encoder with title and context.
  """
  return encoder.CascadeEncoder(
      encoders=[
          encoder.SpeechToTextWithTitleAndContextEncoder(
              speech_to_text_encoder=whisper_encoder.SpeechToTextEncoder(
                  model_path=whisper_model_path
              )
          ),
          converter.SoundEmbeddingToTextConverter(),
          HFLLMEncoder(
              model_path=hf_llm_model_path,
              normalizer=normalizer,
              prompt=prompt,
          ),
          converter.TextEmbeddingToTextPredictionConverter(),
      ]
  )


def RagHFLLMWithTitleAndContextWhisperEncoder(
    whisper_model_path: str,
    hf_llm_model_path: str,
    rag_encoder: encoder.MultiModalEncoder,
    top_k: int = 10,
    normalizer: Callable[[str], str] | None = None,
    prompt: prompt_lib.Prompt = prompt_lib.RetrievalPrompt(),
) -> encoder.CascadeEncoder:
  """RAG HF LLM encoder with title and context."""

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
          HFLLMEncoder(
              model_path=hf_llm_model_path,
              normalizer=normalizer,
              prompt=prompt,
          ),
          converter.TextEmbeddingToTextPredictionConverter(),
      ]
  )


def HFLLMWithTitleAndContextEncoder(
    model_path: str,
    normalizer: Callable[[str], str] | None = None,
    prompt: prompt_lib.Prompt = prompt_lib.ReasoningPrompt(),
) -> encoder.CascadeEncoder:
  """E2E HF LLM encoder with title and context."""

  return encoder.CascadeEncoder(
      encoders=[
          HFLLMEncoder(
              model_path=model_path,
              normalizer=normalizer,
              prompt=prompt,
          ),
          converter.SoundEmbeddingToTextConverter(),
          converter.TextToTextPredictionConverter(),
      ]
  )


def RagHFLLMWithTitleAndContextEncoder(
    model_path: str,
    rag_encoder: encoder.MultiModalEncoder,
    top_k: int = 10,
    normalizer: Callable[[str], str] | None = None,
    prompt: prompt_lib.Prompt = prompt_lib.RetrievalPrompt(),
) -> encoder.CascadeEncoder:
  """RAG HF LLM encoder with title and context."""

  return encoder.CascadeEncoder(
      encoders=[
          rag_encoder,
          retrieval_encoder.RetrievalEncoder(top_k=top_k),
          HFLLMEncoder(
              model_path=model_path,
              normalizer=normalizer,
              prompt=prompt,
          ),
          converter.SoundEmbeddingToTextConverter(),
          converter.TextToTextPredictionConverter(),
      ]
  )
