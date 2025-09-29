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

"""Converter between different types of MultiModalObjects for MSEB encoders."""

from collections.abc import Sequence
from typing import final

import jaxtyping
from mseb import encoder
from mseb import types
import numpy as np


class SoundToSoundEmbeddingConverter(encoder.MultiModalEncoder):
  """Converter between Sound and SoundEmbedding objects for MSEB encoders."""

  @final
  def _setup(self):
    return

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


class SoundEmbeddingToTextConverter(encoder.MultiModalEncoder):
  """Converter between different types of MultiModalObjects for MSEB encoders."""

  @final
  def _setup(self):
    return

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
            context=types.TextContextParams(id=sound_embedding.context.id),
        )
      else:
        text = types.Text(
            text=str(embedding[0]),
            context=types.TextContextParams(id=sound_embedding.context.id),
        )
      outputs.append(text)
    return outputs
