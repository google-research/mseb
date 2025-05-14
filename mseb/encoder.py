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

"""MSEB Encoder base class."""

from __future__ import annotations

import dataclasses
from typing import Any, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

import numpy as np


@dataclasses.dataclass
class ContextParams:
  """A dataclass to hold configuration parameters for a model."""
  language: Optional[str] = None
  speaker_id: Optional[str] = None
  speaker_age: Optional[int] = None
  speaker_gender: Optional[int] = None
  sample_rate: Optional[int] = None


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
      The encoded sentences.
    """
    ...
