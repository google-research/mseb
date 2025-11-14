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

"""Gemini embedding encoders."""

import functools
import logging
import os
import re
from typing import Callable, Optional, Sequence, Tuple

from google import genai
from mseb import encoder
from mseb import types
from mseb.encoders import converter
from mseb.encoders import prompt as prompt_lib
from mseb.encoders import text_encoder_with_prompt as prompt_encoder_lib
from mseb.encoders import whisper_encoder
import numpy as np
import tensorflow as tf



def encoder_genai(  # pylint: disable=invalid-name
    inputs: Sequence[Tuple[str | Optional[bytes]]],
    *,
    model_name: str,
    client: genai.Client,
    task_type: str | None,
) -> np.ndarray:
  """Encodes input texts using Gemini embedding model via GenAI API."""
  response = client.models.embed_content(
      model=model_name,
      contents=[input[0] for input in inputs],
      config=genai.types.EmbedContentConfig(task_type=task_type),
  )
  embeddings = [x.values for x in response.embeddings]
  return np.array(embeddings)


class GeminiEmbeddingTextEncoder(prompt_encoder_lib.TextEncoderWithPrompt):
  """Gemini embedding text encoder."""

  def __init__(
      self,
      model_path: str,
      normalizer: Callable[[str], str] | None = lambda x: re.sub(
          r'\[\d+\]', '', x.lower()
      ),
      prompt_template: str | None = 'title: {title} | text: {text}',
      task_type: str | None = None,
  ):
    """Initializes the Gemini embedding model.

    Args:
      model_path: The Gemini embedding model name or path.
      normalizer: A function that normalizes the text before encoding. This is
        useful for removing special characters or formatting the text for better
        encoding results.
      prompt_template: Format of the prompt to be used for Gemini embedding, see
        go/gem-embed-text-user-card ("How to Use") for more details. For
        example, 'task: search result | query: {text}' for queries and 'title:
        {title} | text: {text}' for documents".
      task_type: Task type for Gemini embedding model. One of: None,
        RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, SEMANTIC_SIMILARITY,
        CLASSIFICATION, CLUSTERING.
    """
    super().__init__(
        normalizer, prompt=prompt_lib.DefaultPrompt(prompt_template)
    )
    self.model_path = model_path
    self.task_type = task_type

  def _setup(self):
    logging.info('Connecting to Gemini embedding at: %s', self.model_path)

    if not os.environ.get('GEMINI_API_KEY'):
      raise ValueError('Environment variable GEMINI_API_KEY is not set.')
    self.prompt_encode_fn = functools.partial(
        encoder_genai,
        model_name=self.model_path,
        client=genai.Client(),
        task_type=self.task_type,
    )


def GeminiEmbeddingTranscriptTruthEncoder(
    model_path: str,
    normalizer: Callable[[str], str] | None = None,
    prompt_template: str = 'task: search result | query: {text}',
    task_type: str | None = None,
) -> encoder.CascadeEncoder:
  """Cascaded transcript truth and Gemini embedding encoder.

  Args:
    model_path: The Gemini embedding model name or path.
    normalizer: A function that normalizes the text before encoding. This is
      useful for removing special characters or formatting the text for better
      encoding results.
    prompt_template: Format of the prompt to be used for Gemini embedding, see
      go/gem-embed-text-user-card ("How to Use") for more details. For example,
      'task: search result | query: {text}' for queries and
      'title: {title} | text: {text}' for documents".
    task_type: Task type for Gemini embedding model. One of: None,
      RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, SEMANTIC_SIMILARITY, CLASSIFICATION,
      CLUSTERING.

  Returns:
    A cascaded transcript truth and Gemini embedding encoder.
  """
  return encoder.CascadeEncoder(
      encoders=[
          converter.SoundToSoundEmbeddingConverter(),
          converter.SoundEmbeddingToTextConverter(),
          GeminiEmbeddingTextEncoder(
              model_path=model_path,
              normalizer=normalizer,
              prompt_template=prompt_template,
              task_type=task_type,
          ),
      ]
  )


def GeminiEmbeddingTranscriptTruthOrGeminiEmbeddingEncoder(
    model_path: str,
    query_normalizer: Callable[[str], str] | None = None,
    query_prompt_template: str | None = 'task: search result | query: {text}',
    query_task_type: str | None = None,
    document_normalizer: Callable[[str], str] | None = lambda x: re.sub(
        r'\[\d+\]', '', x.lower()
    ),
    document_prompt_template: str | None = 'title: {title} | text: {text}',
    document_task_type: str | None = None,
) -> encoder.CollectionEncoder:
  """Pair Sound and Text encoder as for sound to text retrieval."""
  sound_encoder = GeminiEmbeddingTranscriptTruthEncoder(
      model_path=model_path,
      normalizer=query_normalizer,
      prompt_template=query_prompt_template,
      task_type=query_task_type,
  )
  text_encoder = GeminiEmbeddingTextEncoder(
      model_path=model_path,
      normalizer=document_normalizer,
      prompt_template=document_prompt_template,
      task_type=document_task_type,
  )
  return encoder.CollectionEncoder(
      encoder_by_input_type={
          types.Sound: sound_encoder,
          types.Text: text_encoder,
      }
  )


def GeminiEmbeddingWithTitleAndContextTranscriptTruthOrGeminiEmbeddingEncoder(
    model_path: str,
    query_normalizer: Callable[[str], str] | None = None,
    query_prompt_template: str = 'task: search result | query: {text}',
    query_task_type: str | None = None,
    document_normalizer: Callable[[str], str] | None = lambda x: re.sub(
        r'\[\d+\]', '', x.lower()
    ),
    document_prompt_template: str | None = 'title: {title} | text: {text}',
    document_task_type: str | None = None,
) -> encoder.CollectionEncoder:
  """Pair Sound and Text encoder as for sound to text retrieval."""
  sound_encoder = GeminiEmbeddingTranscriptTruthEncoder(
      model_path=model_path,
      normalizer=query_normalizer,
      prompt_template=query_prompt_template,
      task_type=query_task_type,
  )
  text_encoder = GeminiEmbeddingTextEncoder(
      model_path=model_path,
      normalizer=document_normalizer,
      prompt_template=document_prompt_template,
      task_type=document_task_type,
  )
  return encoder.CollectionEncoder(
      encoder_by_input_type={
          types.SoundWithTitleAndContext: sound_encoder,
          types.Text: text_encoder,
      }
  )


def GeminiEmbeddingWhisperEncoder(
    whisper_model_path: str,
    gemini_embedding_model_path: str,
    normalizer: Callable[[str], str] | None = None,
    prompt_template: str = 'task: search result | query: {text}',
    gemini_embedding_task_type: str | None = None,
) -> encoder.CascadeEncoder:
  """Cascaded Whisper and Gemini embedding encoder.

  Args:
    whisper_model_path: A serializable string (e.g., a GCS path or Hub ID)
      pointing to the Whisper model to be loaded in setup().
    gemini_embedding_model_path: The Gemini embedding model name or path.
    normalizer: A function that normalizes the text before encoding. This is
      useful for removing special characters or formatting the text for better
      encoding results.
    prompt_template: Format of the prompt to be used for Gemini embedding, see
      go/gem-embed-text-user-card (How to Use) for more details.
    gemini_embedding_task_type: Task type for Gemini embedding model. One of:
      None, RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, SEMANTIC_SIMILARITY,
      CLASSIFICATION, CLUSTERING.

  Returns:
    A cascaded Whisper and Gemini embedding encoder.
  """
  return encoder.CascadeEncoder(
      encoders=[
          whisper_encoder.SpeechToTextEncoder(model_path=whisper_model_path),
          converter.SoundEmbeddingToTextConverter(),
          GeminiEmbeddingTextEncoder(
              model_path=gemini_embedding_model_path,
              normalizer=normalizer,
              prompt_template=prompt_template,
              task_type=gemini_embedding_task_type,
          ),
      ]
  )


def GeminiEmbeddingWithTitleAndContextWhisperEncoder(
    whisper_model_path: str,
    gemini_embedding_model_path: str,
    normalizer: Callable[[str], str] | None = None,
    prompt_template: str = 'task: search result | query: {text}',
    gemini_embedding_task_type: str | None = None,
) -> encoder.CascadeEncoder:
  """Cascaded Whisper and Gemini embedding encoder with title and context.

  Args:
    whisper_model_path: A serializable string (e.g., a GCS path or Hub ID)
      pointing to the Whisper model to be loaded in setup().
    gemini_embedding_model_path: The Gemini embedding model name or path.
    normalizer: A function that normalizes the text before encoding. This is
      useful for removing special characters or formatting the text for better
      encoding results.
    prompt_template: Format of the prompt to be used for Gemini embedding, see
      go/gem-embed-text-user-card (How to Use) for more details.
    gemini_embedding_task_type: Task type for Gemini embedding model. One of:
      None, RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, SEMANTIC_SIMILARITY,
      CLASSIFICATION, CLUSTERING.

  Returns:
    A cascaded Whisper and Gemini embedding encoder with title and context.
  """
  return encoder.CascadeEncoder(
      encoders=[
          encoder.SpeechToTextWithTitleAndContextEncoder(
              speech_to_text_encoder=whisper_encoder.SpeechToTextEncoder(
                  model_path=whisper_model_path
              )
          ),
          converter.SoundEmbeddingToTextConverter(),
          GeminiEmbeddingTextEncoder(
              model_path=gemini_embedding_model_path,
              normalizer=normalizer,
              prompt_template=prompt_template,
              task_type=gemini_embedding_task_type,
          ),
      ]
  )


def GeminiEmbeddingWhisperOrGeminiEmbeddingEncoder(
    whisper_model_path: str,
    gemini_embedding_model_path: str,
    query_normalizer: Callable[[str], str] | None = None,
    query_prompt_template: str | None = 'task: search result | query: {text}',
    query_task_type: str | None = None,
    document_normalizer: Callable[[str], str] | None = lambda x: re.sub(
        r'\[\d+\]', '', x.lower()
    ),
    document_prompt_template: str | None = 'title: {title} | text: {text}',
    document_task_type: str | None = None,
) -> encoder.CollectionEncoder:
  """Pair Sound and Text encoder as for sound to text retrieval."""
  sound_encoder = GeminiEmbeddingWhisperEncoder(
      whisper_model_path=whisper_model_path,
      gemini_embedding_model_path=gemini_embedding_model_path,
      normalizer=query_normalizer,
      prompt_template=query_prompt_template,
      gemini_embedding_task_type=query_task_type,
  )
  text_encoder = GeminiEmbeddingTextEncoder(
      model_path=gemini_embedding_model_path,
      normalizer=document_normalizer,
      prompt_template=document_prompt_template,
      task_type=document_task_type,
  )
  return encoder.CollectionEncoder(
      encoder_by_input_type={
          types.Sound: sound_encoder,
          types.Text: text_encoder,
      }
  )


def GeminiEmbeddingWithTitleAndContextWhisperOrGeminiEmbeddingEncoder(
    whisper_model_path: str,
    gemini_embedding_model_path: str,
    query_normalizer: Callable[[str], str] | None = None,
    query_prompt_template: str = 'task: search result | query: {text}',
    query_task_type: str | None = None,
    document_normalizer: Callable[[str], str] | None = lambda x: re.sub(
        r'\[\d+\]', '', x.lower()
    ),
    document_prompt_template: str | None = 'title: {title} | text: {text}',
    document_task_type: str | None = None,
) -> encoder.CollectionEncoder:
  """Pair Sound and Text encoder as for sound to text retrieval."""
  sound_encoder = GeminiEmbeddingWithTitleAndContextWhisperEncoder(
      whisper_model_path=whisper_model_path,
      gemini_embedding_model_path=gemini_embedding_model_path,
      normalizer=query_normalizer,
      prompt_template=query_prompt_template,
      gemini_embedding_task_type=query_task_type,
  )
  text_encoder = GeminiEmbeddingTextEncoder(
      model_path=gemini_embedding_model_path,
      normalizer=document_normalizer,
      prompt_template=document_prompt_template,
      task_type=document_task_type,
  )
  return encoder.CollectionEncoder(
      encoder_by_input_type={
          types.SoundWithTitleAndContext: sound_encoder,
          types.Text: text_encoder,
      }
  )
