# Copyright 2024 The MSEB Authors.
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

"""MSEB Decoder base class."""

from __future__ import annotations

from typing import Any, Optional, Protocol, Tuple, runtime_checkable

from mseb import encoder
import numpy as np


@runtime_checkable
class Decoder(Protocol):
  """A protocol defining the interface for MSEB decoders.

  Implementations of this protocol are responsible for reconstructing
  the original audio waveform sequence and/or its associated contextual
  information (encapsulated in a `ContextParams` object) from
  timestamps and embeddings.

  These decoders typically work in conjunction with an MSEB encoder
  (e.g., an implementation from `mseb.encoder`) that produces the
  input timestamps and embeddings.
  """

  def decode(
      self,
      timestamps: np.ndarray,
      embeddings: np.ndarray,
      **kwargs: Any,
  ) -> Tuple[Optional[np.ndarray], Optional[encoder.ContextParams]]:
    """Decodes timestamps and embeddings to the waveform and/or its context.

    Args:
      timestamps: A 2D NumPy array of shape `(n_segments, 2)`, where each row
                  contains the start and end time [start_time, end_time] of a
                  segment.
      embeddings: A NumPy array of shape `(n_segments, *embedding_dims)`.
                  The first dimension (`n_segments`) must match the first
                  dimension of `timestamps`. The remaining dimensions
                  (`*embedding_dims`) represent the embedding vector for each
                  segment.
      **kwargs: Additional keyword arguments for decoder-specific behavior or
                parameters.

    Returns:
      A tuple `(waveform, context)`:
        - `waveform`: A NumPy array representing the reconstructed audio
          waveform, or `None` if the waveform is not decoded by this call
          or by this decoder.
        - `context`: An instance of `encoder.ContextParams` containing the
          recovered contextual parameters, or `None` if contextual parameters
          are not decoded by this call or by this decoder.

      Depending on the decoder's capabilities and the specific decoding request,
      one or both elements of the tuple can be non-`None`.
    """
    ...
