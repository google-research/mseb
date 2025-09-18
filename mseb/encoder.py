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
from typing import Any, final, Sequence

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
      self, batch: Sequence[types.MultiModalInput]
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
      self, batch: Sequence[types.MultiModalInput]
  ) -> Sequence[types.SoundEmbedding | types.TextEmbeddings]:
    """Encodes a batch of multi-modal example."""

  @final
  def encode(
      self, batch: Sequence[types.MultiModalInput]
  ) -> Sequence[types.SoundEmbedding | types.TextEmbeddings]:
    """Encodes a batch of multi-modal examples."""
    self._check_input_types(batch)
    return self._encode(batch)


class SoundEncoder(abc.ABC):
  """Defines the interface for encoding audio into embeddings.

  This abstract class provides a standardized structure for sound encoders
  within the MSEB benchmark. It's designed for lazy loading of models, making it
  suitable for large-scale, distributed processing environments.

  Subclasses are responsible for implementing the model loading logic (`setup`)
  and the core single-item processing (`encode`). For performance, it is
  highly recommended that subclasses also override the default `encode_batch`
  method with a more efficient, truly vectorized implementation.
  """

  def __init__(self, model_path: str, **kwargs: Any):
    """Initializes the encoder with configuration.

    Note: This method is lightweight and only stores configuration. The
    heavy model loading is deferred to the `setup()` method.

    Args:
      model_path: A serializable string (e.g., a GCS path or Hub ID)
        pointing to the model to be loaded in setup().
      **kwargs: Model-specific initialization arguments that will be stored
        in `self._kwargs` for use in `setup()`.
    """
    self.model_path = model_path
    self._model_loaded = False
    self._kwargs = kwargs

  @abc.abstractmethod
  def setup(self):
    """Loads the model into memory.

    This method is intended to be called once on each worker before any data
    is processed. It should use `self.model_path` and `self._kwargs` to load
    the model and any necessary assets.
    """
    # In the subclass implementation, you would set this to True
    # self._model_loaded = True
    ...

  def _ensure_model_loaded(self):
    """Checks if the model is loaded and calls setup() if not."""
    if not self._model_loaded:
      self.setup()
      if not self._model_loaded:
        raise RuntimeError(
            "The 'setup()' method was called but '_model_loaded' "
            "was not set to True."
        )

  @abc.abstractmethod
  def _encode_batch(
      self,
      sound_batch: Sequence[types.Sound],
      **kwargs: Any,
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
    ...

  def encode(
      self,
      sound: types.Sound,
      **kwargs: Any,
  ) -> types.SoundEmbedding:
    """Ensures the model is loaded, then encodes a single audio source.

    This method acts as a template, handling the model loading check before
    delegating the core encoding logic to the `_encode` method.
    Subclasses should not override this method.

    Args:
      sound: The sound source to encode.
      **kwargs: Any additional, model-specific runtime parameters required for
        this specific encoding call.

    Returns:
      A tuple containing:
        - waveform_embeddings (np.ndarray): A 2D array of shape
          (n, embedding_dim).
        - embedding_timestamps (np.ndarray): A 2D array of shape (m, 2),
          where each row is an [start, end] pair indicating a segment by
          sound waveform index.
          There are two common cases for the relation between embeddings (n)
          and timestamps (m):
            - Frame-Aligned (m == n): The i-th timestamp corresponds
              directly to the i
    """
    return self.encode_batch([sound], **kwargs)[0]

  @final
  def encode_batch(
      self,
      sound_batch: Sequence[types.Sound],
      **kwargs: Any,
  ) -> Sequence[types.SoundEmbedding]:
    """Encodes a batch of sound sources.

    This is a default, non-performant implementation that processes items
    serially. For optimal performance, subclasses SHOULD override this method
    with a truly batched implementation that processes the entire batch in a
    single model call.

    Args:
      sound_batch: A sequence of sound sources to encode.
      **kwargs: Any additional parameters required for encoding.

    Returns:
      A list of SoundEmbedding, one for each input.
    """
    self._ensure_model_loaded()
    return self._encode_batch(sound_batch, **kwargs)


class SoundEncoderAsMultiModalEncoder(MultiModalEncoder):
  """A wrapper for SoundEncoder to implement MultiModalEncoder."""

  def __init__(self, sound_encoder: SoundEncoder):
    super().__init__()
    self._sound_encoder = sound_encoder

  def _setup(self):
    self._sound_encoder.setup()

  def _check_input_types(
      self, batch: Sequence[types.MultiModalInput]
  ) -> None:
    if not all(isinstance(x, types.Sound) for x in batch):
      raise ValueError(
          "SoundEncoderAsMultiModalEncoder only supports a batch of all Sound"
          " inputs."
      )

  def _encode(
      self, batch: Sequence[types.MultiModalInput]
  ) -> Sequence[types.SoundEmbedding | types.TextEmbeddings]:
    sound_batch: list[types.Sound] = []
    for example in batch:
      assert isinstance(example, types.Sound)
      sound_batch.append(example)
    return self._sound_encoder.encode_batch(sound_batch)


class TextEncoder(abc.ABC):
  """Defines the interface for encoding text into embeddings.

  This abstract class provides a standardized structure for text encoders
  within the MSEB benchmark. It's designed for lazy loading of models, making it
  suitable for large-scale, distributed processing environments.

  Subclasses are responsible for implementing the model loading logic (`setup`)
  and the core single-item processing (`encode`).
  """

  def __init__(self, **kwargs: Any):
    """Initializes the encoder with configuration.

    Note: This method is lightweight and only stores configuration. The
    heavy model loading is deferred to the `setup()` method.

    Args:
      **kwargs: Model-specific initialization arguments that will be stored in
        `self._kwargs` for use in `setup()`.
    """
    self._model_loaded = False
    self._kwargs = kwargs

  @abc.abstractmethod
  def setup(self):
    """Loads the model into memory.

    This method is intended to be called once on each worker before any data
    is processed. It should use `self.model_path` and `self._kwargs` to load
    the model and any necessary assets.
    """
    # In the subclass implementation, you would set this to True
    # self._model_loaded = True
    ...

  def _ensure_model_loaded(self):
    """Checks if the model is loaded and calls setup() if not."""
    if not self._model_loaded:
      self.setup()
      if not self._model_loaded:
        raise RuntimeError(
            "The 'setup()' method was called but '_model_loaded' "
            "was not set to True."
        )

  @abc.abstractmethod
  def _encode_batch(
      self, text_batch: Sequence[types.Text], **kwargs: Any
  ) -> Sequence[types.TextEmbeddings]:
    """Encodes a batch of text sources.

    Args:
      text_batch: A sequence of text sources to encode.
      **kwargs: Any additional parameters required for encoding.

    Returns:
      A list of types.TextEmbeddings objects, one for each input:
        - embeddings (np.ndarray): A 2D array of shape (n, embedding_dim).
        - spans (np.ndarray of ints): A 2D array of shape (m, 2),
          where each row indicates a span, i.e., a tuple of the start
          (inclusive) and end (exclusive) index of the span in the text (in
          characters).

      There are two common cases for the relation between embeddings (n)
      and spans (m):
        - Frame-Aligned (m == n): The i-th span corresponds directly to the
          i-th embedding vector.
        - Utterance-Level (m == 1): A single span represents the entire text
          segment from which the embeddings were extracted.
    """
    ...

  def encode(self, text: types.Text, **kwargs: Any) -> types.TextEmbeddings:
    """Encodes a single text source.

    This method is a convenience wrapper around `encode_batch` that handles
    single-item encoding. Subclasses should not override this method.

    Args:
      text: The text source to encode.
      **kwargs: Any additional, model-specific runtime parameters required for
        this specific encoding call.

    Returns:
      A TextEmbeddings object.
    """
    return self.encode_batch([text], **kwargs)[0]

  @final
  def encode_batch(
      self,
      text_batch: Sequence[types.Text],
      **kwargs: Any,
  ) -> Sequence[types.TextEmbeddings]:
    """Ensures the model is loaded, then encodes a batch of text sources.

    This method acts as a template, handling the model loading check before
    delegating the core encoding logic to the `_encode_batch` method.
    Subclasses should not override this method.

    Args:
      text_batch: A sequence of text sources to encode.
      **kwargs: Any additional parameters required for encoding.

    Returns:
      A list of types.TextEmbeddings objects, one for each input.
    """
    self._ensure_model_loaded()
    return self._encode_batch(text_batch, **kwargs)


class TextEncoderAsMultiModalEncoder(MultiModalEncoder):
  """A wrapper for TextEncoder to implement MultiModalEncoder."""

  def __init__(self, text_encoder: TextEncoder):
    super().__init__()
    self._text_encoder = text_encoder

  def _setup(self):
    self._text_encoder.setup()

  def _check_input_types(self, batch: Sequence[types.MultiModalInput]) -> None:
    if not all(isinstance(x, types.Text) for x in batch):
      raise ValueError(
          "TextEncoderAsMultiModalEncoder only supports a batch of all Text"
          " inputs."
      )

  def _encode(
      self, batch: Sequence[types.MultiModalInput]
  ) -> Sequence[types.SoundEmbedding | types.TextEmbeddings]:
    text_batch: list[types.Text] = []
    for example in batch:
      assert isinstance(example, types.Text)
      text_batch.append(example)
    return self._text_encoder.encode_batch(text_batch)


class SoundOrTextEncoder(MultiModalEncoder):
  """Pair Sound and Text encoder as for sound to text retrieval."""

  def __init__(self, sound_encoder: SoundEncoder, text_encoder: TextEncoder):
    super().__init__()
    self._sound_encoder = sound_encoder
    self._text_encoder = text_encoder

  def _setup(self):
    self._sound_encoder.setup()
    self._text_encoder.setup()

  def _check_input_types(self, batch: Sequence[types.MultiModalInput]) -> None:
    if not (
        all(isinstance(x, types.Sound) for x in batch)
        or all(isinstance(x, types.Text) for x in batch)
    ):
      raise ValueError(
          "SoundOrTextEncoder only supports a batch of all Sound or all Text"
          " inputs."
      )

  def _encode(
      self, batch: Sequence[types.MultiModalInput]
  ) -> Sequence[types.SoundEmbedding | types.TextEmbeddings]:
    if isinstance(batch[0], types.Sound):
      return self._sound_encoder.encode_batch(batch)
    else:
      return self._text_encoder.encode_batch(batch)
