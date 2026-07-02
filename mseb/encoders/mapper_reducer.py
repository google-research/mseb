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

"""Mapper and reducer encoders for SoundEmbeddingCollections."""

from collections.abc import Callable, Sequence
from mseb import encoder as encoder_lib
from mseb import types


class SoundEmbeddingCollectionMapper(encoder_lib.MultiModalEncoder):
  """Applies an encoder to each SoundEmbedding in a SoundEmbeddingCollection."""

  def __init__(self, encoder: encoder_lib.MultiModalEncoder):
    super().__init__()
    self._encoder = encoder

  def _setup(self):
    self._encoder.setup()

  def _check_input_types(self, batch: Sequence[types.MultiModalObject]):
    if not all(isinstance(x, types.SoundEmbeddingCollection) for x in batch):
      raise ValueError(
          "SoundEmbeddingCollectionMapper only supports "
          "SoundEmbeddingCollection inputs."
      )

  def _encode(
      self,
      batch: Sequence[types.MultiModalObject],
  ) -> Sequence[types.SoundEmbeddingCollection]:
    """Encodes a batch of SoundEmbeddingCollections."""
    outputs = []
    for collection in batch:
      assert isinstance(collection, types.SoundEmbeddingCollection)
      outputs.append(
          types.SoundEmbeddingCollection(
              embeddings={  # pyrefly: ignore[bad-argument-type]
                  k: self._encoder.encode([v])[0]
                  for k, v in collection.embeddings.items()
              },
              context=collection.context,
          )
      )
    return outputs


class SoundEmbeddingCollectionReducer(encoder_lib.MultiModalEncoder):
  """Reduces a SoundEmbeddingCollection to a SoundEmbedding."""

  def __init__(
      self,
      combine_fn: Callable[
          [Sequence[types.SoundEmbedding], types.SoundContextParams],
          types.SoundEmbedding,
      ],
  ):
    super().__init__()
    self._combine_fn = combine_fn

  def _setup(self):
    pass

  def _check_input_types(self, batch: Sequence[types.MultiModalObject]):
    if not all(isinstance(x, types.SoundEmbeddingCollection) for x in batch):
      raise ValueError(
          "SoundEmbeddingCollectionReducer only supports "
          "SoundEmbeddingCollection inputs."
      )

  def _encode(
      self,
      batch: Sequence[types.MultiModalObject],
  ) -> Sequence[types.SoundEmbedding]:
    """Encodes a batch of SoundEmbeddingCollections."""
    outputs = []
    for collection in batch:
      assert isinstance(collection, types.SoundEmbeddingCollection)
      outputs.append(self._combine_fn(list(collection.embeddings.values()),  # pyrefly: ignore[bad-argument-type]
                                      collection.context))
    return outputs
