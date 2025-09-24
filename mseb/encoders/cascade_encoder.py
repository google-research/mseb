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

from typing import Any, Callable, final, Sequence

import jaxtyping
from mseb import encoder
from mseb import types
from mseb.encoders import normalized_text_encoder_with_prompt as text_encoder
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
      model_path: str,
      text_encoder_cls: type[encoder.MultiModalEncoder],
      text_encoder_kwargs: dict[str, Any],
      speech_to_text_encoder_cls: type[encoder.MultiModalEncoder] | None = None,
      speech_to_text_encoder_kwargs: dict[str, Any] | None = None,
  ):
    """Initializes the sound and text encoders from configurations.

    Args:
      model_path: Not used.
      text_encoder_cls: The class of the text encoder to use.
      text_encoder_kwargs: The keyword arguments to pass to the text encoder
        constructor.
      speech_to_text_encoder_cls: The class of the speech-to-text encoder to
        use. If not provided, the transcript is taken from params.text.
      speech_to_text_encoder_kwargs: The keyword arguments to pass to the
        speech-to-text encoder constructor.
    """
    super().__init__()
    self.text_encoder = text_encoder_cls(**text_encoder_kwargs)
    if speech_to_text_encoder_cls is not None:
      speech_to_text_encoder_kwargs = speech_to_text_encoder_kwargs or {}
      self.speech_to_text_encoder = speech_to_text_encoder_cls(
          **speech_to_text_encoder_kwargs
      )
    else:
      self.speech_to_text_encoder = None

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
      embedding: jaxtyping.Shaped[np.ndarray, '1'] = transcripts.embedding
      text = str(embedding[0])
      if isinstance(transcripts.context.text, types.Text):
        context = transcripts.context.text.context
      else:
        context = types.TextContextParams(id=transcripts.context.id)
      text_batch.append(types.Text(text=text, context=context))
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
      **kwargs: Any,
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
      **kwargs: Model-specific initialization arguments that will be stored in
        `self._kwargs` for use in `setup()`.
    """
    super().__init__(
        'not used',
        text_encoder_cls=text_encoder.GeckoTextEncoder,
        text_encoder_kwargs={
            'model_path': model_path,
            'normalizer': normalizer,
            'prompt_template': prompt_template,
        },
        **kwargs,
    )


class GeckoWhisperEncoder(CascadeEncoder):
  """Cascaded Whisper and Gecko encoder."""

  def __init__(
      self,
      model_path: str,
      gecko_model_path: str,
      normalizer: Callable[[str], str] | None = None,
      prompt_template: str = 'task: search result | query: {text}',
      **kwargs: Any,
  ):
    """Initializes the Whisper and Gecko models.

    Args:
      model_path: A serializable string (e.g., a GCS path or Hub ID) pointing to
        the Whisper model to be loaded in setup().
      gecko_model_path: A serializable string (e.g., a GCS path or Hub ID)
        pointing to the Gecko model to be loaded in setup().
      normalizer: A function that normalizes the text before encoding. This is
        useful for removing special characters or formatting the text for better
        encoding results.
      prompt_template: Format of the prompt to be used for Gecko. Typically, the
        prompt is of the form: 'task: search result | query: {text}' for queries
        and 'title: {title} | text: {text}' for documents".
      **kwargs: Model-specific initialization arguments that will be stored in
        `self._kwargs` for use in `setup()`.
    """
    super().__init__(
        'not_used',
        speech_to_text_encoder_cls=whisper_encoder.SpeechToTextEncoder,
        speech_to_text_encoder_kwargs={
            'model_path': model_path,
        },
        text_encoder_cls=text_encoder.GeckoTextEncoder,
        text_encoder_kwargs={
            'model_path': gecko_model_path,
            'normalizer': normalizer,
            'prompt_template': prompt_template,
        },
    )
