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

"""MSEB Evaluator base class."""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Sequence, Union

from mseb import encoder


class Evaluator(abc.ABC):
  """The MSEB evaluator's base class.

  Each task should implement an instance of this class.
  """

  def __init__(self,
               sound_encoder: encoder.Encoder,
               encode_kwargs: dict[str, Any]):
    self.sound_encoder = sound_encoder
    self.encode_kwargs = encode_kwargs

  @abc.abstractmethod
  def __call__(self,
               sequence: Union[str, Sequence[float]],
               context: encoder.ContextParams,
               **kwargs: Any,
               ) -> Dict[str, float]:
    """Evaluates quality of the encoder for input sequence and return metrics.

    Args:
      sequence: Input sound sequence to encode. String-type sequences are
        interpreted as sound file paths.
      context: Encoder input context parameters.
      **kwargs: Additional arguments to pass to the evaluator.

    Returns:
      Dictionary of metrics.
    """
    ...

  @abc.abstractmethod
  def combine_scores(self, scores: List[Dict[str, float]]) -> Dict[str, float]:
    """Combines individual example scores into a merged score dictionary.

    This abstract method must be implemented by subclasses to define
    how the scores from multiple examples are aggregated (e.g., averaged,
    summed, or based on specific metrics).

    Args:
      scores: A list of dictionaries, where each dictionary represents
              the metrics and their values for a single example (e.g.,
              `[{'accuracy': 0.9, 'f1': 0.8}, {'accuracy': 0.85, 'f1': 0.82}]`).

    Returns:
      A single dictionary containing the combined/aggregated scores.
      The keys represent metric names and values are their combined results.
    """
    ...
