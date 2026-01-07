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
import dataclasses
from typing import Mapping, Sequence, Type, final

import librosa
from mseb import types
import numpy as np


INVALID_ANSWER_STR = ""
NO_RESPONSE_STR = "NO_RESPONSE"


def resample_sound(
    sound: types.Sound,
    target_sr: int,
    target_dtype: (
        type[np.float32]
        | type[np.float64]
        | type[np.int16]
        | type[np.int32]
        | type[np.int8]
    ) = np.float32,
) -> types.Sound:
  """Resamples a Sound object to a target sample rate if necessary."""
  supported_dtypes = (np.float32, np.float64, np.int16, np.int32, np.int8)
  if sound.waveform.dtype.type not in supported_dtypes:
    raise ValueError(
        f"Unsupported input waveform dtype: {sound.waveform.dtype}"
    )
  if target_dtype not in supported_dtypes:
    raise ValueError(f"Unsupported target_dtype: {target_dtype}")

  if (
      sound.context.sample_rate == target_sr
      and sound.waveform.dtype == np.dtype(target_dtype)
  ):
    return sound

  if np.issubdtype(sound.waveform.dtype, np.integer):
    info = np.iinfo(sound.waveform.dtype)
    d = -info.min if info.min != info.max else 1.0
    waveform_float = sound.waveform.astype(np.float32) / d
  else:
    waveform_float = sound.waveform.astype(np.float32)

  if sound.context.sample_rate == target_sr:
    resampled_waveform = waveform_float
    new_context = sound.context
  else:
    resampled_waveform = librosa.resample(
        waveform_float,
        orig_sr=sound.context.sample_rate,
        target_sr=target_sr,
    )
    new_context = dataclasses.replace(
        sound.context,
        sample_rate=target_sr,
        length=len(resampled_waveform),
    )

  if target_dtype == np.float32:
    output_waveform = resampled_waveform
  elif target_dtype == np.float64:
    output_waveform = resampled_waveform.astype(np.float64)
  else:
    info = np.iinfo(target_dtype)
    output_waveform = np.clip(
        resampled_waveform * -info.min, info.min, info.max
    ).astype(target_dtype)

  return types.Sound(waveform=output_waveform, context=new_context)


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
  def _check_input_types(self, batch: Sequence[types.MultiModalObject]) -> None:
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

  def output_type(self) -> type[types.MultiModalObject]:
    """The type of the output of the encoder."""
    assert self._encode is not None
    return self._encode.__annotations__["return"].__args__[0]


class CascadeEncoder(MultiModalEncoder):
  """Sequence encoder interface.

  A wrapper around a sequence of encoders (including converters to match the
  output type of the previous encoder with the input type of the next encoder)
  that are applied in sequence.

  Example: A cascade encoder consisting of a speech-to-text encoder (ASR)
  followed by a text-to-embedding encoder. A converter is used to convert the
  output of the ASR encoder (SoundEmbedding) to the input of the text-to-
  embedding encoder (Text).

  Attributes:
    _encoders: The sequence of encoders to apply.
  """

  def __init__(self, encoders: Sequence[MultiModalEncoder]):
    super().__init__()
    self._encoders = encoders

  @final
  def _setup(self):
    for encoder in self._encoders:
      encoder.setup()

  @final
  def _check_input_types(self, batch: Sequence[types.MultiModalObject]) -> None:
    # CascadeEncoder checks input types in _encode.
    return

  @final
  def _encode(
      self, batch: Sequence[types.MultiModalObject]
  ) -> Sequence[types.MultiModalObject]:
    outputs = batch
    for encoder in self._encoders:
      encoder._check_input_types(outputs)  # pylint: disable=protected-access
      outputs = encoder.encode(outputs)
    return outputs

  def output_type(self) -> type[types.MultiModalObject]:
    """The type of the output of the encoder."""
    return self._encoders[-1].output_type()


class CollectionEncoder(MultiModalEncoder):
  """Collection encoder interface.

  A wrapper around a collection of independent encoders each of different input
  type that are used for a task. The batches of inputs are assumed to be all of
  the same type.

  Example (retrieval task): A collection encoder consisting of a sound encoder
  (for the audio query) and a text encoder (for generating the index of text
  documents).

  Attributes:
    _encoder_by_input_type: A mapping of input type to encoder.
  """

  def __init__(
      self,
      encoder_by_input_type: Mapping[
          Type[types.MultiModalObject], MultiModalEncoder
      ],
  ):
    super().__init__()
    self._encoder_by_input_type = encoder_by_input_type

  @final
  def _setup(self):
    for encoder in self._encoder_by_input_type.values():
      encoder.setup()

  @final
  def _check_input_types(self, batch: Sequence[types.MultiModalObject]) -> None:
    if not all(isinstance(x, type(batch[0])) for x in batch):
      raise ValueError(
          "CollectionEncoder only supports a batch of all inputs of the same"
          " type, type must be one of:"
          f" {tuple(self._encoder_by_input_type.keys())}."
      )

  def _encode(
      self, batch: Sequence[types.MultiModalObject]
  ) -> Sequence[types.MultiModalObject]:
    return self._encoder_by_input_type[type(batch[0])].encode(batch)


class SpeechToTextWithTitleAndContextEncoder(MultiModalEncoder):
  """Represents speech with its transcription derived by Whisper model."""

  def __init__(self, speech_to_text_encoder: MultiModalEncoder):
    super().__init__()
    self.speech_to_text_encoder = speech_to_text_encoder

  def _check_input_types(self, batch: Sequence[types.MultiModalObject]) -> None:
    if not all(
        isinstance(x, types.SoundWithTitleAndContext) for x in batch
    ) and not all(isinstance(x, types.Sound) for x in batch):
      raise ValueError(
          "SpeechToTextWithTitleAndContextEncoder only supports a batch of all"
          " SoundWithTitleAndContext or all Sound inputs."
      )

  def _setup(self):
    self.speech_to_text_encoder.setup()

  def _encode(
      self, batch: Sequence[types.MultiModalObject]
  ) -> Sequence[types.SoundEmbeddingWithTitleAndContext | types.SoundEmbedding]:
    """Encodes a batch of sound sources into embeddings and timestamps."""
    sound_batch = []
    if not all(isinstance(x, types.SoundWithTitleAndContext) for x in batch):
      for x in self.speech_to_text_encoder.encode(batch):
        assert isinstance(x, types.SoundEmbedding)
        sound_batch.append(x)
      return sound_batch
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
