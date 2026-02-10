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

"""MSEB Encoder base class."""

import re
from typing import Callable

from mseb import encoder
from mseb import types
from mseb.encoders import converter
from mseb.encoders import text_encoder_with_prompt as prompt_encoder_lib
from mseb.encoders import whisper_encoder


def GeckoTranscriptTruthEncoder(
    model_path: str,
    normalizer: Callable[[str], str] | None = None,
    prompt_template: str = 'task: search result | query: {text}',
) -> encoder.CascadeEncoder:
  """Transcript truth encoder with Gecko model.

  Args:
    model_path: A serializable string (e.g., a GCS path or Hub ID) pointing to
      the model to be loaded in setup().
    normalizer: A function that normalizes the text before encoding. This is
      useful for removing special characters or formatting the text for better
      encoding results.
    prompt_template: Format of the prompt to be used for Gecko. Typically, the
      prompt is of the form: 'task: search result | query: {text}' for queries
      and 'title: {title} | text: {text}' for documents".

  Returns:
    A CascadeEncoder that encodes Sound to text to embedding.
  """
  return encoder.CascadeEncoder(
      encoders=[
          converter.SoundToSoundEmbeddingConverter(),
          converter.SoundEmbeddingToTextConverter(),
          prompt_encoder_lib.GeckoTextEncoder(
              model_path=model_path,
              normalizer=normalizer,
              prompt_template=prompt_template,
          ),
      ]
  )


def GeckoTranscriptTruthOrGeckoEncoder(
    gecko_model_path: str,
    query_normalizer: Callable[[str], str] | None = None,
    query_prompt_template: str | None = 'task: search result | query: {text}',
    document_normalizer: Callable[[str], str] | None = lambda x: re.sub(
        r'\[\d+\]', '', x.lower()
    ),
    document_prompt_template: str | None = 'title: {title} | text: {text}',
) -> encoder.CollectionEncoder:
  """Pair Sound and Text encoder as for sound to text retrieval."""
  sound_encoder = GeckoTranscriptTruthEncoder(
      model_path=gecko_model_path,
      normalizer=query_normalizer,
      prompt_template=query_prompt_template,
  )
  text_encoder = prompt_encoder_lib.GeckoTextEncoder(
      model_path=gecko_model_path,
      normalizer=document_normalizer,
      prompt_template=document_prompt_template,
  )
  return encoder.CollectionEncoder(
      encoder_by_input_type={
          types.Sound: sound_encoder,
          types.Text: text_encoder,
      }
  )


def GeckoWithTitleAndContextTranscriptTruthOrGeckoEncoder(
    gecko_model_path: str,
    query_normalizer: Callable[[str], str] | None = None,
    query_prompt_template: str = 'task: search result | query: {text}',
    document_normalizer: Callable[[str], str] | None = lambda x: re.sub(
        r'\[\d+\]', '', x.lower()
    ),
    document_prompt_template: str | None = 'title: {title} | text: {text}',
) -> encoder.CollectionEncoder:
  """Pair SoundWithTitleAndContext and Text encoder as for sound to text retrieval."""
  sound_encoder = GeckoTranscriptTruthEncoder(
      model_path=gecko_model_path,
      normalizer=query_normalizer,
      prompt_template=query_prompt_template,
  )
  text_encoder = prompt_encoder_lib.GeckoTextEncoder(
      model_path=gecko_model_path,
      normalizer=document_normalizer,
      prompt_template=document_prompt_template,
  )
  return encoder.CollectionEncoder(
      encoder_by_input_type={
          types.SoundWithTitleAndContext: sound_encoder,
          types.Text: text_encoder,
      }
  )


def GeckoWhisperEncoder(
    whisper_model_path: str,
    gecko_model_path: str,
    normalizer: Callable[[str], str] | None = None,
    prompt_template: str = 'task: search result | query: {text}',
) -> encoder.CascadeEncoder:
  """Cascaded Whisper and Gecko encoder.

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

  Returns:
    A CascadeEncoder that encodes Sound to text to embedding.
  """
  return encoder.CascadeEncoder(
      encoders=[
          whisper_encoder.SpeechToTextEncoder(model_path=whisper_model_path),
          converter.SoundEmbeddingToTextConverter(),
          prompt_encoder_lib.GeckoTextEncoder(
              model_path=gecko_model_path,
              normalizer=normalizer,
              prompt_template=prompt_template,
          ),
      ]
  )


def GeckoWhisperOrGeckoEncoder(
    whisper_model_path: str,
    gecko_model_path: str,
    query_normalizer: Callable[[str], str] | None = None,
    query_prompt_template: str | None = 'task: search result | query: {text}',
    document_normalizer: Callable[[str], str] | None = lambda x: re.sub(
        r'\[\d+\]', '', x.lower()
    ),
    document_prompt_template: str | None = 'title: {title} | text: {text}',
) -> encoder.CollectionEncoder:
  """Pair Sound and Text encoder as for sound to text retrieval."""
  sound_encoder = GeckoWhisperEncoder(
      whisper_model_path=whisper_model_path,
      gecko_model_path=gecko_model_path,
      normalizer=query_normalizer,
      prompt_template=query_prompt_template,
  )
  text_encoder = prompt_encoder_lib.GeckoTextEncoder(
      model_path=gecko_model_path,
      normalizer=document_normalizer,
      prompt_template=document_prompt_template,
  )
  return encoder.CollectionEncoder(
      encoder_by_input_type={
          types.Sound: sound_encoder,
          types.Text: text_encoder,
      }
  )


def GeckoWithTitleAndContextWhisperEncoder(
    whisper_model_path: str,
    gecko_model_path: str,
    normalizer: Callable[[str], str] | None = None,
    prompt_template: str = 'task: search result | query: {text}',
) -> encoder.CascadeEncoder:
  """Cascaded Whisper with title and context and Gecko encoder.

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

  Returns:
    A CascadeEncoder that encodes sound to text to embedding.
  """
  return encoder.CascadeEncoder(
      encoders=[
          encoder.SpeechToTextWithTitleAndContextEncoder(
              speech_to_text_encoder=whisper_encoder.SpeechToTextEncoder(
                  model_path=whisper_model_path
              )
          ),
          converter.SoundEmbeddingToTextConverter(),
          prompt_encoder_lib.GeckoTextEncoder(
              model_path=gecko_model_path,
              normalizer=normalizer,
              prompt_template=prompt_template,
          ),
      ]
  )


def GeckoWithTitleAndContextWhisperOrGeckoEncoder(
    whisper_model_path: str,
    gecko_model_path: str,
    query_normalizer: Callable[[str], str] | None = None,
    query_prompt_template: str = 'task: search result | query: {text}',
    document_normalizer: Callable[[str], str] | None = lambda x: re.sub(
        r'\[\d+\]', '', x.lower()
    ),
    document_prompt_template: str | None = 'title: {title} | text: {text}',
) -> encoder.CollectionEncoder:
  """Pair Sound and Text encoder as for sound to text retrieval."""
  sound_encoder = GeckoWithTitleAndContextWhisperEncoder(
      whisper_model_path=whisper_model_path,
      gecko_model_path=gecko_model_path,
      normalizer=query_normalizer,
      prompt_template=query_prompt_template,
  )
  text_encoder = prompt_encoder_lib.GeckoTextEncoder(
      model_path=gecko_model_path,
      normalizer=document_normalizer,
      prompt_template=document_prompt_template,
  )
  return encoder.CollectionEncoder(
      encoder_by_input_type={
          types.SoundWithTitleAndContext: sound_encoder,
          types.Text: text_encoder,
      }
  )
