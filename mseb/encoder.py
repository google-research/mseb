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

import abc
from typing import final, Sequence

from mseb import types


class MultiModalEncoder(abc.ABC):
  """Multi-modal encoder interface.

  Subclasses are responsible for implementing model loading logic (`_setup`)
  and batch encoding (`_encode`).

  The `encode` method takes an arbitrary batch of `types.Sound` and
  `types.Text` objects. The encoder implementation must check at run-time that
  the sequence of objects provided as args is valid for that model
  implementation.

  The __init__ method is lightweight and only stores configuration. All
  heavy model loading is deferred to the `_setup()` method.
  """

  def __init__(self):
    self._is_setup = False

  @abc.abstractmethod
  def _setup(self):
    """Loads the encoder and prepares for use."""

  @final
  def setup(self):
    """Loads the encoder and prepares for use."""
    if not self._is_setup:
      self._setup()
      self._is_setup = True

  @abc.abstractmethod
  def _check_input_types(
      self, batch: Sequence[types.MultiModalObject]
  ) -> None:
    """Validates the modality for the specific encoder.

    Subclasses must implement this method to verify that the input modality is
    supported by the underlying model.

    Args:
      batch: A batch of MultiModelInput example to check.
    Raises:
      ValueError: If the sequence of input types is not a valid combination
        for this encoder.
    """

  @abc.abstractmethod
  def _encode(
      self, batch: Sequence[types.MultiModalObject]
  ) -> Sequence[types.MultiModalObject]:
    """Encodes a batch of multi-modal example."""

  @final
  def encode(
      self, batch: Sequence[types.MultiModalObject]
  ) -> Sequence[types.MultiModalObject]:
    """Encodes a batch of multi-modal examples."""
    self._check_input_types(batch)
    embeddings = self._encode(batch)
    for features, embedding in zip(batch, embeddings):
      embedding.encoding_stats = types.EncodingStats(
          input_size_bytes=features.size_bytes,
          embedding_size_bytes=embedding.size_bytes,
          flops=self.get_encode_flops(features),
      )
    return embeddings

  def get_encode_flops(self, data: types.MultiModalObject) -> int | None:
    """Returns total flops used to encode or None if not yet implemented."""
    del data
    return None


class SoundOrTextEncoder(MultiModalEncoder):
  """Pair Sound and Text encoder as for sound to text retrieval."""

  def __init__(
      self, sound_encoder: MultiModalEncoder, text_encoder: MultiModalEncoder
  ):
    super().__init__()
    self._sound_encoder = sound_encoder
    self._text_encoder = text_encoder

  def _setup(self):
    self._sound_encoder.setup()
    self._text_encoder.setup()

  def _check_input_types(self, batch: Sequence[types.MultiModalObject]) -> None:
    if not (
        all(isinstance(x, types.Sound) for x in batch)
        or all(isinstance(x, types.Text) for x in batch)
    ):
      raise ValueError(
          "SoundOrTextEncoder only supports a batch of all Sound or all Text"
          " inputs."
      )

  def _encode(
      self, batch: Sequence[types.MultiModalObject]
  ) -> Sequence[types.MultiModalObject]:
    if isinstance(batch[0], types.Sound):
      return self._sound_encoder.encode(batch)
    else:
      return self._text_encoder.encode(batch)


class SpeechToTextWithTitleAndContextEncoder(MultiModalEncoder):
  """Represents speech with its transcription derived by Whisper model."""

  def __init__(self, speech_to_text_encoder: MultiModalEncoder):
    super().__init__()
    self.speech_to_text_encoder = speech_to_text_encoder

  def _check_input_types(self, batch: Sequence[types.MultiModalObject]) -> None:
    if not all(isinstance(x, types.SoundWithTitleAndContext) for x in batch):
      raise ValueError(
          "SpeechToTextWithTitleAndContextEncoder only supports a batch of all"
          " SoundWithTitleAndContext inputs."
      )

  def _setup(self):
    self.speech_to_text_encoder.setup()

  def _encode(
      self, batch: Sequence[types.MultiModalObject]
  ) -> Sequence[types.SoundEmbeddingWithTitleAndContext]:
    """Encodes a batch of sound sources into embeddings and timestamps."""
    sound_batch = []
    for example in batch:
      assert isinstance(example, types.SoundWithTitleAndContext)
      sound_batch.append(types.Sound(example.waveform, example.context))
    sound_embeddings = self.speech_to_text_encoder.encode(sound_batch)
    outputs = []
    for sound_embedding, example in zip(sound_embeddings, batch):
      assert isinstance(sound_embedding, types.SoundEmbedding)
      assert isinstance(example, types.SoundWithTitleAndContext)
      outputs.append(
          types.SoundEmbeddingWithTitleAndContext(
              embedding=sound_embedding.embedding,
              timestamps=sound_embedding.timestamps,
              title_text=example.title_text,
              context_text=example.context_text,
              context=sound_embedding.context,
          )
      )
    return outputs
