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
from typing import Callable, final, Sequence

import jaxtyping
from mseb import encoder
from mseb import types
from mseb.encoders import normalized_text_encoder_with_prompt as text_encoder_lib
from mseb.encoders import whisper_encoder
import numpy as np


class CascadeEncoder(encoder.MultiModalEncoder):
  """Cascaded encoder for speech into text into embeddings.

  This class provides a standardized structure for cascaded encoders
  consisting of a speech-to-text encoder (ASR) followed by a text-to-embedding
  encoder, within the MSEB benchmark. If the ASR encoder is not provided, the
  transcript is taken from params.text.

  Subclasses are responsible for setting the two encoders in the constructor.
  """

  def __init__(
      self,
      text_encoder: encoder.MultiModalEncoder,
      speech_to_text_encoder: encoder.MultiModalEncoder | None = None,
  ):
    """Initializes the sound and text encoders from configurations.

    Args:
      text_encoder: The text encoder to use.
      speech_to_text_encoder: The speech-to-text encoder to use. If not
        provided, the transcript is taken from params.text.
    """
    super().__init__()
    self.text_encoder = text_encoder
    self.speech_to_text_encoder = speech_to_text_encoder

  @final
  def _setup(self):
    """Loads the models into memory."""
    if self.speech_to_text_encoder is not None:
      self.speech_to_text_encoder.setup()
    self.text_encoder.setup()

  def _check_input_types(self, batch: Sequence[types.MultiModalObject]) -> None:
    if not all(isinstance(x, types.Sound) for x in batch):
      raise ValueError(
          'CascadeEncoder only supports a batch of all Sound inputs.'
      )

  def _convert_sound_embedding_to_text(
      self, sound_embedding: types.SoundEmbedding
  ) -> types.Text:
    """Converts a SoundEmbedding object to a TextEmbeddings object."""
    embedding: jaxtyping.Shaped[np.ndarray, '1'] = sound_embedding.embedding
    if isinstance(sound_embedding, types.SoundEmbeddingWithTitleAndContext):
      return types.TextWithTitleAndContext(
          text=str(embedding[0]),
          title_text=sound_embedding.title_text,
          context_text=sound_embedding.context_text,
          context=types.TextContextParams(id=sound_embedding.context.id),
      )
    else:
      return types.Text(
          text=str(embedding[0]),
          context=types.TextContextParams(id=sound_embedding.context.id),
      )

  @final
  def _encode(
      self, sound_batch: Sequence[types.MultiModalObject]
  ) -> Sequence[types.SoundEmbedding]:
    """Encodes a batch of sound sources.

    Args:
      sound_batch: A sequence of sound sources to encode.

    Returns:
      A list of types.SoundEmbedding objects, one for each input:
        - embeddings (np.ndarray): A 2D array of shape (n, embedding_dim).
        - timestamps (np.ndarray): A 2D array of shape (m, 2), where each row is
          an [start, end] pair indicating a segment where start and end are in
          seconds.
          There are two common cases for the relation between embeddings (n)
          and timestamps (m):
            - Frame-Aligned (m == n): The i-th timestamp corresponds
              directly to the i-th embedding vector.
            - Utterance-Level (m == 1): A single timestamp pair represents
              the start and end of the entire audio segment from which the
              embeddings were extracted.
    """
    if self.speech_to_text_encoder is not None:
      transcripts_batch = self.speech_to_text_encoder.encode(sound_batch)
    else:
      transcripts_batch = []
      for sound in sound_batch:
        assert isinstance(sound, types.Sound)
        context = sound.context
        transcripts_batch.append(
            types.SoundEmbedding(
                embedding=np.array([context.text], dtype=object),
                timestamps=np.array([[
                    context.waveform_start_second,
                    context.waveform_end_second,
                ]]),
                context=context,
            )
        )
    text_batch = []
    sound_embeddings_batch = []
    for transcripts in transcripts_batch:
      if not isinstance(transcripts, types.SoundEmbedding):
        raise TypeError(
            'CascadeEncoder expects sound_encoder to return SoundEmbedding, '
            f'but got {type(transcripts)}.'
        )
      sound_embeddings_batch.append(transcripts)
      text_batch.append(self._convert_sound_embedding_to_text(transcripts))
    text_embeddings_batch = self.text_encoder.encode(text_batch)

    outputs = []
    for text_embeddings, sound_embedding in zip(
        text_embeddings_batch, sound_embeddings_batch
    ):
      assert isinstance(text_embeddings, types.TextEmbeddings)
      outputs.append(
          types.SoundEmbedding(
              embedding=text_embeddings.embeddings,
              timestamps=sound_embedding.timestamps,
              context=sound_embedding.context,
          )
      )

    return outputs


class GeckoTranscriptTruthEncoder(CascadeEncoder):
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
        text_encoder=text_encoder_lib.GeckoTextEncoder(
            model_path=model_path,
            normalizer=normalizer,
            prompt_template=prompt_template),
    )


class GeckoTranscriptTruthOrGeckoEncoder(encoder.SoundOrTextEncoder):
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
    super().__init__(
        sound_encoder=GeckoTranscriptTruthEncoder(
            model_path=gecko_model_path,
            normalizer=query_normalizer,
            prompt_template=query_prompt_template,
        ),
        text_encoder=text_encoder_lib.GeckoTextEncoder(
            model_path=gecko_model_path,
            normalizer=document_normalizer,
            prompt_template=document_prompt_template,
        ),
    )


class GeckoWhisperEncoder(CascadeEncoder):
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
        speech_to_text_encoder=whisper_encoder.SpeechToTextEncoder(
            model_path=whisper_model_path
        ),
        text_encoder=text_encoder_lib.GeckoTextEncoder(
            model_path=gecko_model_path,
            normalizer=normalizer,
            prompt_template=prompt_template,
        ),
    )


class GeckoWhisperOrGeckoEncoder(encoder.SoundOrTextEncoder):
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
    super().__init__(
        sound_encoder=GeckoWhisperEncoder(
            whisper_model_path=whisper_model_path,
            gecko_model_path=gecko_model_path,
            normalizer=query_normalizer,
            prompt_template=query_prompt_template,
        ),
        text_encoder=text_encoder_lib.GeckoTextEncoder(
            model_path=gecko_model_path,
            normalizer=document_normalizer,
            prompt_template=document_prompt_template,
        ),
    )


class GeckoWithTitleAndContextWhisperEncoder(CascadeEncoder):
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
        speech_to_text_encoder=encoder.SpeechToTextWithTitleAndContextEncoder(
            speech_to_text_encoder=whisper_encoder.SpeechToTextEncoder(
                model_path=whisper_model_path
            )
        ),
        text_encoder=text_encoder_lib.GeckoTextEncoder(
            model_path=gecko_model_path,
            normalizer=normalizer,
            prompt_template=prompt_template,
        ),
    )


class GeckoWithTitleAndContextWhisperOrGeckoEncoder(encoder.SoundOrTextEncoder):
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
    super().__init__(
        sound_encoder=GeckoWithTitleAndContextWhisperEncoder(
            whisper_model_path=whisper_model_path,
            gecko_model_path=gecko_model_path,
            normalizer=query_normalizer,
            prompt_template=query_prompt_template,
        ),
        text_encoder=text_encoder_lib.GeckoTextEncoder(
            model_path=gecko_model_path,
            normalizer=document_normalizer,
            prompt_template=document_prompt_template,
        ),
    )
