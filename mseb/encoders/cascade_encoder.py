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
import numpy as np


class CascadeEncoder(encoder.SoundEncoder):
  """Defines the interface for cascaded encoding audio into text into embeddings.

  This class provides a standardized structure for cascaded sound encoders
  consisting of a speech-to-text encoder (ASR) followed by a text-to-embedding
  encoder, within the MSEB benchmark. If the ASR encoder is not provided, the
  text is taken from params.text.

  Subclasses are responsible for setting the two encoders in the constructor.
  """

  def __init__(
      self,
      model_path: str,
      text_encoder_cls: type[encoder.TextEncoder],
      text_encoder_kwargs: dict[str, Any],
      sound_encoder_cls: type[encoder.SoundEncoder] | None = None,
      sound_encoder_kwargs: dict[str, Any] | None = None,
      **kwargs: Any,
  ):
    """Initializes the sound and text encoders from configurations.

    Args:
      model_path: Not used.
      text_encoder_cls: The class of the text encoder to use.
      text_encoder_kwargs: The keyword arguments to pass to the text encoder
        constructor.
      sound_encoder_cls: The class of the sound encoder to use. If not provided,
        the text is taken from params.text.
      sound_encoder_kwargs: The keyword arguments to pass to the sound encoder
        constructor.
      **kwargs: Model-specific initialization arguments that will be stored in
        `self._kwargs` for use in `setup()`.
    """
    super().__init__(model_path=model_path, **kwargs)
    self.text_encoder: encoder.TextEncoder = text_encoder_cls(
        **text_encoder_kwargs
    )
    if sound_encoder_cls is not None:
      sound_encoder_kwargs = sound_encoder_kwargs or {}
      self.sound_encoder: encoder.SoundEncoder = sound_encoder_cls(
          **sound_encoder_kwargs
      )
    else:
      self.sound_encoder = None
    self._model_loaded = False
    self._kwargs = kwargs

  @final
  def setup(self):
    """Loads the models into memory."""
    if self.sound_encoder is not None:
      self.sound_encoder.setup()
    self.text_encoder.setup()
    self._model_loaded = True

  @final
  def _encode_batch(
      self, sound_batch: Sequence[types.Sound], **kwargs: Any
  ) -> Sequence[types.SoundEmbedding]:
    """Encodes a batch of sound sources.

    Args:
      sound_batch: A sequence of sound sources to encode.
      **kwargs: Any additional parameters required for encoding.

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
    if self.sound_encoder is not None:
      transcripts_batch = self.sound_encoder.encode_batch(sound_batch, **kwargs)
    else:
      transcripts_batch = []
      for sound in sound_batch:
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
    for transcripts in transcripts_batch:
      embedding: jaxtyping.Shaped[np.ndarray, '1'] = transcripts.embedding
      test = str(embedding[0])
      text_batch.append(
          types.Text(text=test, context=types.TextContextParams(id=''))
      )
    text_embeddings_batch = self.text_encoder.encode_batch(text_batch, **kwargs)

    outputs = [
        types.SoundEmbedding(
            embedding=text_embeddings.embeddings,
            timestamps=transcripts.timestamps,
            context=transcripts.context,
        )
        for text_embeddings, transcripts in zip(
            text_embeddings_batch, transcripts_batch
        )
    ]

    return outputs


class GeckoTranscriptTruthEncoderV2(CascadeEncoder):
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
    self.text_encoder = text_encoder.GeckoTextEncoder(
        model_path, normalizer, prompt_template
    )
