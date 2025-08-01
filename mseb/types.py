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

"""MSEB types."""

import dataclasses
from typing import Optional

import numpy as np


@dataclasses.dataclass
class SoundContextParams:
  """Parameters for a sound example."""
  sample_rate: int
  length: int

  language: Optional[str] = None
  speaker_id: Optional[str] = None
  speaker_age: Optional[int] = None
  speaker_gender: Optional[int] = None
  text: Optional[str] = None
  # The starting second of the relevant waveform segment.
  # Defaults to 0, representing the beginning of the waveform array.
  waveform_start_second: int = 0
  # The exclusive ending second for the segment. The
  # slice of sound segment is `[start:end]`. Defaults to the
  # maximum float32 value to signify "to the end of the waveform."
  waveform_end_second: int = np.finfo(np.float32).max


@dataclasses.dataclass(frozen=True)
class Score:
  """A dataclass for a single evaluation metric."""
  metric: str
  description: str
  value: float
  min: int | float
  max: int | float
  weight: float = 1.0

  def __post_init__(self):
    """Validates the score data."""
    if not isinstance(self.metric, str) or not self.metric:
      raise TypeError("Score 'metric' must be a non-empty string.")
    if not isinstance(self.description, str):
      raise TypeError("Score 'description' must be a string.")
    if not isinstance(self.value, float):
      raise TypeError("Score 'value' must be a float.")
    if (not isinstance(self.min, (int, float)) or
        not isinstance(self.max, (int, float))):
      raise TypeError("Score 'min' and 'max' must be numbers.")
    if self.min > self.max:
      raise ValueError(
          f"Score 'min' ({self.min}) cannot be greater than 'max' ({self.max})."
      )
    if not isinstance(self.weight, float) or self.weight < 0:
      raise ValueError(
          "Score 'weight' must be a non-negative float."
      )


@dataclasses.dataclass(frozen=True)
class Dataset:
  """A dataclass for dataset metadata."""
  path: str
  revision: str

  def __post_init__(self):
    """Validates the dataset data."""
    if not isinstance(self.path, str) or not self.path:
      raise TypeError("Dataset 'path' must be a non-empty string.")
    if not isinstance(self.revision, str) or not self.revision:
      raise TypeError("Dataset 'revision' must be a non-empty string.")


@dataclasses.dataclass(frozen=True)
class TaskMetadata:
  """A dataclass for storing the metadata of a task."""

  # Attributes with straightforward types
  name: str
  description: str
  reference: str
  type: str
  category: str
  main_score: str
  revision: str

  # Nested dataclass attributes
  dataset: Dataset
  scores: list[Score]

  # List attributes
  eval_splits: list[str]
  eval_langs: list[str]
  domains: list[str] = dataclasses.field(default_factory=list)
  task_subtypes: list[str] = dataclasses.field(default_factory=list)

  def __post_init__(self):
    """Validates task meta data."""
    for field in dataclasses.fields(self):
      value = getattr(self, field.name)
      if field.type is str and not value:
        raise TypeError(
            f"Metadata attribute '{field.name}' must be a non-empty string.")

    for attr_name in ["eval_splits", "eval_langs", "scores"]:
      if not getattr(self, attr_name):
        raise TypeError(
            f"Metadata attribute '{attr_name}' must be a non-empty list."
        )

    for attr_name in ["eval_splits", "eval_langs", "domains", "task_subtypes"]:
      attr_value = getattr(self, attr_name)
      if not isinstance(attr_value, list):
        raise TypeError(f"Metadata attribute '{attr_name}' must be a list.")
      if not all(isinstance(item, str) for item in attr_value):
        raise TypeError(
            f"All items in metadata attribute '{attr_name}' must be strings."
        )

    defined_metrics = {s.metric for s in self.scores}
    if self.main_score not in defined_metrics:
      raise ValueError(
          f"main_score '{self.main_score}' is not defined in the 'scores' list."
      )
