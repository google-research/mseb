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
from typing import Any, final, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

from mseb import types
import numpy as np


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
      waveform_batch: Sequence[Sequence[float]],
      params_batch: Sequence[types.SoundContextParams],
      **kwargs: Any,
  ) -> Sequence[tuple[np.ndarray, np.ndarray]]:
    """Encodes a batch of sound sources.

    This is a default, non-performant implementation that processes items
    serially. For optimal performance, subclasses SHOULD override this method
    with a truly batched implementation that processes the entire batch in a
    single model call.

    Args:
      waveform_batch: A sequence of sound sources to encode.
      params_batch: A sequence of `SoundContextParams` objects, each
        corresponding to an item in `sound_batch`.
      **kwargs: Any additional parameters required for encoding.

    Returns:
      A list of tuples, one for each input, each tuple containing:
        - waveform_embeddings (np.ndarray): A 2D array of shape
          (n, embedding_dim).
        - embedding_timestamps (np.ndarray): A 2D array of shape (m, 2),
          where each row is an [start, end] pair indicating a segment by
          sound waveform index.
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
      waveform: Sequence[float],
      params: types.SoundContextParams,
      **kwargs: Any,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Ensures the model is loaded, then encodes a single audio source.

    This method acts as a template, handling the model loading check before
    delegating the core encoding logic to the `_encode` method.
    Subclasses should not override this method.

    Args:
      waveform: The sound source to encode. It is pre-loaded  waveform
        (Sequence[float]).
      params: A `SoundContextParams` object containing metadata and context
        about the sound, such as its sample rate.
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
    return self.encode_batch([waveform], [params], **kwargs)[0]

  @final
  def encode_batch(
      self,
      waveform_batch: Sequence[Sequence[float]],
      params_batch: Sequence[types.SoundContextParams],
      **kwargs: Any,
  ) -> Sequence[tuple[np.ndarray, np.ndarray]]:
    """Encodes a batch of sound sources.

    This is a default, non-performant implementation that processes items
    serially. For optimal performance, subclasses SHOULD override this method
    with a truly batched implementation that processes the entire batch in a
    single model call.

    Args:
      waveform_batch: A sequence of sound sources to encode.
      params_batch: A sequence of `SoundContextParams` objects, each
        corresponding to an item in `sound_batch`.
      **kwargs: Any additional parameters required for encoding.

    Returns:
      A list of (waveform_embeddings, embedding_timestamps) tuples, one
      for each input.
    """
    self._ensure_model_loaded()
    return self._encode_batch(waveform_batch, params_batch, **kwargs)


@dataclasses.dataclass
class ContextParams:
  """A dataclass to hold configuration parameters for a model."""
  frame_length: Optional[int] = None
  frame_step: Optional[int] = None
  language: Optional[str] = None
  speaker_id: Optional[str] = None
  speaker_age: Optional[int] = None
  speaker_gender: Optional[int] = None
  sample_rate: Optional[int] = None
  text: Optional[str] = None
  audio_start_seconds: float = 0.0
  audio_end_seconds: float = np.finfo(np.float32).max
  prompt: Optional[str] = None


@runtime_checkable
class Encoder(Protocol):
  """The MSEB encoder's base class.

  The base class to encode waveform sequences along with optional context to
  sequence of embeddings.
  """

  def encode(self,
             sequence: Union[str, Sequence[float]],
             context: ContextParams,
             **kwargs: Any,
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Encodes the given sentences using the encoder.

    Args:
      sequence: Input sound sequence to encode. String-type sequences are
        interpreted as sound file paths.
      context: Encoder input context parameters.
      **kwargs: Additional arguments to pass to the encoder.

    Returns:
      The encoded sentence.
    """
    ...

  def encode_batch(
      self,
      sequences: Sequence[Union[str, Sequence[float]]],
      contexts: Sequence[ContextParams],
      **kwargs: Any,
  ) -> Sequence[Tuple[np.ndarray, np.ndarray]]:
    """Encodes the given sentences in batch using the encoder.

    Args:
      sequences: Input sound sequences to encode. String-type sequences are
        interpreted as sound file paths.
      contexts: Encoder input context parameters, one per sequence.
      **kwargs: Additional arguments to pass to the encoder.

    Returns:
      The encoded sentences.
    """
    return [
        self.encode(sequence, context, **kwargs)
        for sequence, context in zip(sequences, contexts)
    ]
