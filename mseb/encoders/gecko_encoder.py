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

"""MSEB Encoder base class."""

import re
from typing import Callable

from mseb import encoder
from mseb import types
from mseb.encoders import converter
from mseb.encoders import normalized_text_encoder_with_prompt as text_encoder_lib
from mseb.encoders import whisper_encoder


class GeckoTranscriptTruthEncoder(encoder.CascadeEncoder):
  """Transcript truth encoder with Gecko model."""

  def __init__(
      self,
      model_path: str,
      normalizer: Callable[[str], str] | None = None,
      prompt_template: str = 'task: search result | query: {text}',
  ):
    """Initializes the transcript truth and Gecko models.

    Args:
      model_path: A serializable string (e.g., a GCS path or Hub ID) pointing to
        the model to be loaded in setup().
      normalizer: A function that normalizes the text before encoding. This is
        useful for removing special characters or formatting the text for better
        encoding results.
      prompt_template: Format of the prompt to be used for Gecko. Typically, the
        prompt is of the form: 'task: search result | query: {text}' for queries
        and 'title: {title} | text: {text}' for documents".
    """
    super().__init__(
        encoders=[
            converter.SoundToSoundEmbeddingConverter(),
            converter.SoundEmbeddingToTextConverter(),
            text_encoder_lib.GeckoTextEncoder(
                model_path=model_path,
                normalizer=normalizer,
                prompt_template=prompt_template,
            ),
        ]
    )


class GeckoTranscriptTruthOrGeckoEncoder(encoder.CollectionEncoder):
  """Pair Sound and Text encoder as for sound to text retrieval."""

  def __init__(
      self,
      gecko_model_path: str,
      query_normalizer: Callable[[str], str] | None = None,
      query_prompt_template: str = 'task: search result | query: {text}',
      document_normalizer: Callable[[str], str] | None = lambda x: re.sub(
          r'\[\d+\]', '', x.lower()
      ),
      document_prompt_template: str | None = 'title: {title} | text: {text}',
  ):
    sound_encoder = (
        GeckoTranscriptTruthEncoder(
            model_path=gecko_model_path,
            normalizer=query_normalizer,
            prompt_template=query_prompt_template,
        ),
    )
    text_encoder = text_encoder_lib.GeckoTextEncoder(
        model_path=gecko_model_path,
        normalizer=document_normalizer,
        prompt_template=document_prompt_template,
    )
    super().__init__(
        encoder_by_input_type={
            types.Sound: sound_encoder,
            types.Text: text_encoder,
        }
    )


class GeckoWhisperEncoder(encoder.CascadeEncoder):
  """Cascaded Whisper and Gecko encoder."""

  def __init__(
      self,
      whisper_model_path: str,
      gecko_model_path: str,
      normalizer: Callable[[str], str] | None = None,
      prompt_template: str = 'task: search result | query: {text}',
  ):
    """Initializes the Whisper and Gecko models.

    Args:
      whisper_model_path: A serializable string (e.g., a GCS path or Hub ID)
        pointing to the Whisper model to be loaded in setup().
      gecko_model_path: A serializable string (e.g., a GCS path or Hub ID)
        pointing to the Gecko model to be loaded in setup().
      normalizer: A function that normalizes the text before encoding. This is
        useful for removing special characters or formatting the text for better
        encoding results.
      prompt_template: Format of the prompt to be used for Gecko. Typically, the
        prompt is of the form: 'task: search result | query: {text}' for queries
        and 'title: {title} | text: {text}' for documents".
    """
    super().__init__(
        encoders=[
            whisper_encoder.SpeechToTextEncoder(model_path=whisper_model_path),
            converter.SoundEmbeddingToTextConverter(),
            text_encoder_lib.GeckoTextEncoder(
                model_path=gecko_model_path,
                normalizer=normalizer,
                prompt_template=prompt_template,
            ),
        ]
    )


class GeckoWhisperOrGeckoEncoder(encoder.CollectionEncoder):
  """Pair Sound and Text encoder as for sound to text retrieval."""

  def __init__(
      self,
      whisper_model_path: str,
      gecko_model_path: str,
      query_normalizer: Callable[[str], str] | None = None,
      query_prompt_template: str = 'task: search result | query: {text}',
      document_normalizer: Callable[[str], str] | None = lambda x: re.sub(
          r'\[\d+\]', '', x.lower()
      ),
      document_prompt_template: str | None = 'title: {title} | text: {text}',
  ):
    sound_encoder = GeckoWhisperEncoder(
        whisper_model_path=whisper_model_path,
        gecko_model_path=gecko_model_path,
        normalizer=query_normalizer,
        prompt_template=query_prompt_template,
    )
    text_encoder = text_encoder_lib.GeckoTextEncoder(
        model_path=gecko_model_path,
        normalizer=document_normalizer,
        prompt_template=document_prompt_template,
    )
    super().__init__(
        encoder_by_input_type={
            types.Sound: sound_encoder,
            types.Text: text_encoder,
        }
    )


class GeckoWithTitleAndContextWhisperEncoder(encoder.CascadeEncoder):
  """Cascaded Whisper with title and context and Gecko encoder."""

  def __init__(
      self,
      whisper_model_path: str,
      gecko_model_path: str,
      normalizer: Callable[[str], str] | None = None,
      prompt_template: str = 'task: search result | query: {text}',
  ):
    """Initializes the Whisper and Gecko models.

    Args:
      whisper_model_path: A serializable string (e.g., a GCS path or Hub ID)
        pointing to the Whisper model to be loaded in setup().
      gecko_model_path: A serializable string (e.g., a GCS path or Hub ID)
        pointing to the Gecko model to be loaded in setup().
      normalizer: A function that normalizes the text before encoding. This is
        useful for removing special characters or formatting the text for better
        encoding results.
      prompt_template: Format of the prompt to be used for Gecko. Typically, the
        prompt is of the form: 'task: search result | query: {text}' for queries
        and 'title: {title} | text: {text}' for documents".
    """
    super().__init__(
        encoders=[
            encoder.SpeechToTextWithTitleAndContextEncoder(
                speech_to_text_encoder=whisper_encoder.SpeechToTextEncoder(
                    model_path=whisper_model_path
                )
            ),
            converter.SoundEmbeddingToTextConverter(),
            text_encoder_lib.GeckoTextEncoder(
                model_path=gecko_model_path,
                normalizer=normalizer,
                prompt_template=prompt_template,
            ),
        ]
    )


class GeckoWithTitleAndContextWhisperOrGeckoEncoder(encoder.CollectionEncoder):
  """Pair Sound and Text encoder as for sound to text retrieval."""

  def __init__(
      self,
      whisper_model_path: str,
      gecko_model_path: str,
      query_normalizer: Callable[[str], str] | None = None,
      query_prompt_template: str = 'task: search result | query: {text}',
      document_normalizer: Callable[[str], str] | None = lambda x: re.sub(
          r'\[\d+\]', '', x.lower()
      ),
      document_prompt_template: str | None = 'title: {title} | text: {text}',
  ):
    sound_encoder = GeckoWithTitleAndContextWhisperEncoder(
        whisper_model_path=whisper_model_path,
        gecko_model_path=gecko_model_path,
        normalizer=query_normalizer,
        prompt_template=query_prompt_template,
    )
    text_encoder = text_encoder_lib.GeckoTextEncoder(
        model_path=gecko_model_path,
        normalizer=document_normalizer,
        prompt_template=document_prompt_template,
    )
    super().__init__(
        encoder_by_input_type={
            types.SoundWithTitleAndContext: sound_encoder,
            types.Text: text_encoder,
        }
    )
