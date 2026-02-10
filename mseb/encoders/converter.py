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

"""Converter between different types of MultiModalObjects for MSEB encoders."""

from collections.abc import Sequence
from typing import final

import jaxtyping
from mseb import encoder
from mseb import types
import numpy as np


class Converter(encoder.MultiModalEncoder):
  """Base class for converters."""

  def _setup(self) -> None:
    return


class SoundToSoundEmbeddingConverter(Converter):
  """Converter between Sound and SoundEmbedding objects for MSEB encoders."""

  @final
  def _check_input_types(self, batch: Sequence[types.MultiModalObject]) -> None:
    if not all(isinstance(x, types.Sound) for x in batch):
      raise ValueError(
          'SoundToSoundEmbeddingConverter only supports a batch of all'
          ' Sound inputs.'
      )

  @final
  def _encode(
      self, batch: Sequence[types.MultiModalObject]
  ) -> Sequence[types.SoundEmbedding]:
    """Converts a batch of Sound objects to SoundEmbedding objects."""
    outputs = []
    for sound in batch:
      if isinstance(sound, types.SoundWithTitleAndContext):
        sound_embedding = types.SoundEmbeddingWithTitleAndContext(
            embedding=np.array([sound.context.text], dtype=object),
            timestamps=np.array([[
                sound.context.waveform_start_second,
                sound.context.waveform_end_second,
            ]]),
            context=sound.context,
            title_text=sound.title_text,
            context_text=sound.context_text,
        )
      elif isinstance(sound, types.Sound):
        sound_embedding = types.SoundEmbedding(
            embedding=np.array([sound.context.text], dtype=object),
            timestamps=np.array([[
                sound.context.waveform_start_second,
                sound.context.waveform_end_second,
            ]]),
            context=sound.context,
        )
      else:
        raise ValueError(
            'SoundToSoundEmbeddingConverter only supports Sound or'
            ' SoundWithTitleAndContext inputs.'
        )
      outputs.append(sound_embedding)
    return outputs


class SoundEmbeddingToTextConverter(Converter):
  """Converter between different types of MultiModalObjects for MSEB encoders."""

  def _check_input_types(self, batch: Sequence[types.MultiModalObject]) -> None:
    if not all(isinstance(x, types.SoundEmbedding) for x in batch):
      raise ValueError(
          'SoundEmbeddingToTextConverter only supports a batch of all'
          ' SoundEmbedding inputs.'
      )

  @final
  def _encode(
      self, batch: Sequence[types.MultiModalObject]
  ) -> Sequence[types.Text]:
    """Converts a batch of SoundEmbedding objects to Text objects."""
    outputs = []
    for sound_embedding in batch:
      assert isinstance(sound_embedding, types.SoundEmbedding)
      embedding: jaxtyping.Shaped[np.ndarray, '1'] = sound_embedding.embedding
      if isinstance(sound_embedding, types.SoundEmbeddingWithTitleAndContext):
        text = types.TextWithTitleAndContext(
            text=str(embedding[0]),
            title_text=sound_embedding.title_text,
            context_text=sound_embedding.context_text,
            context=types.TextContextParams(
                id=sound_embedding.context.id,
                debug_text=sound_embedding.context.debug_text,
            ),
        )
      else:
        text = types.Text(
            text=str(embedding[0]),
            context=types.TextContextParams(
                id=sound_embedding.context.id,
                debug_text=sound_embedding.context.debug_text,
            ),
        )
      outputs.append(text)
    return outputs


class TextEmbeddingToTextPredictionConverter(Converter):
  """Converter between TextEmbedding and TextPrediction objects."""

  def _check_input_types(self, batch: Sequence[types.MultiModalObject]) -> None:
    if not all(
        isinstance(x, (types.TextEmbedding, types.SoundEmbedding))
        for x in batch
    ):
      raise ValueError(
          'TextEmbeddingToTextPredictionConverter only supports a batch of'
          ' all TextEmbedding or SoundEmbedding inputs.'
      )

  @final
  def _encode(
      self, batch: Sequence[types.MultiModalObject]
  ) -> Sequence[types.TextPrediction]:
    """Converts a batch of TextEmbedding objects to TextPrediction objects."""
    outputs = []
    for text_embedding in batch:
      assert isinstance(
          text_embedding, (types.TextEmbedding, types.SoundEmbedding)
      )
      embedding: jaxtyping.Shaped[np.ndarray, '1'] = text_embedding.embedding
      outputs.append(
          types.TextPrediction(
              prediction=str(embedding[0]),
              context=types.PredictionContextParams(
                  id=text_embedding.context.id,
                  debug_text=text_embedding.context.debug_text,
              ),
          )
      )
    return outputs


class TextToTextPredictionConverter(Converter):
  """Converter between Text and TextPrediction objects."""

  def _check_input_types(self, batch: Sequence[types.MultiModalObject]) -> None:
    if not all(isinstance(x, types.Text) for x in batch):
      raise ValueError(
          'TextEmbeddingToTextPredictionConverter only supports a batch of'
          ' all Text inputs.'
      )

  @final
  def _encode(
      self, batch: Sequence[types.MultiModalObject]
  ) -> Sequence[types.TextPrediction]:
    """Converts a batch of TextEmbedding objects to TextPrediction objects."""
    outputs = []
    for text in batch:
      assert isinstance(text, types.Text)
      outputs.append(
          types.TextPrediction(
              prediction=text.text,
              context=types.PredictionContextParams(
                  id=text.context.id, debug_text=text.context.debug_text
              ),
          )
      )
    return outputs


class SoundToTextWithTitleAndContextConverter(Converter):
  """Converter between Sound and TextWithTitleAndContext objects."""

  @final
  def _check_input_types(self, batch: Sequence[types.MultiModalObject]) -> None:
    if not all(isinstance(x, types.Sound) for x in batch):
      raise ValueError(
          'SoundToSoundEmbeddingConverter only supports a batch of all'
          ' Sound inputs.'
      )

  @final
  def _encode(
      self, batch: Sequence[types.MultiModalObject]
  ) -> Sequence[types.TextWithTitleAndContext]:
    """Converts a batch of Sound objects to TextWithTitleAndContext objects."""
    outputs = []
    for sound in batch:
      if isinstance(sound, types.SoundWithTitleAndContext):
        text = types.TextWithTitleAndContext(
            text=sound.context.text,
            title_text=sound.title_text,
            context_text=sound.context_text,
            context=types.TextContextParams(
                id=sound.context.id,
                text=sound.context.text,
                debug_text=sound.context.debug_text,
            ),
        )
      elif isinstance(sound, types.Sound):
        text = types.TextWithTitleAndContext(
            text=sound.context.text,
            title_text=None,
            context_text=None,
            context=types.TextContextParams(
                id=sound.context.id,
                text=sound.context.text,
                debug_text=sound.context.debug_text,
            ),
        )
      else:
        raise ValueError(
            'SoundToTextWithTitleAndContextConverter only supports Sound or'
            ' SoundWithTitleAndContext inputs.'
        )
      outputs.append(text)
    return outputs
