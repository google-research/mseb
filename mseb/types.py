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
from typing import Mapping, Optional

import jaxtyping
import numpy as np


@dataclasses.dataclass(frozen=True)
class DatasetMetadata:
  """Structured high-level information about a dataset."""
  name: str
  description: str
  homepage: str
  version: str
  license: str
  mseb_tasks: list[str]
  citation: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class EncodingStats:
  """Information about how an embedding was encoded."""
  input_size_bytes: int  # Size of the input in bytes.
  embedding_size_bytes: int  # Size of the embedding in bytes.
  flops: Optional[int] = None  # Number of floating point operations.

  @property
  def compression_ratio(self) -> float:
    """Ratio of output embedding size to features input size."""
    return self.embedding_size_bytes / self.input_size_bytes


@dataclasses.dataclass
class TextContextParams:
  """Parameters for a text example."""
  id: str  # Identifier for the text example unique within the dataset.
  title: Optional[str] = None
  context: Optional[str] = None


@dataclasses.dataclass
class Text:
  """A dataclass for a text."""
  text: str
  context: TextContextParams

  @property
  def size_bytes(self) -> int:
    """Returns the size of the text in bytes."""
    return len(self.text.encode("utf-8"))


@dataclasses.dataclass
class SoundContextParams:
  """Parameters for a sound example."""
  id: str  # Identifier for the sound example unique within the dataset.
  sample_rate: int
  length: int

  language: Optional[str] = None
  speaker_id: Optional[str] = None
  speaker_age: Optional[int] = None
  speaker_gender: Optional[int] = None
  text: Optional[str] = None
  # The starting second of the relevant waveform segment.
  # Defaults to 0, representing the beginning of the waveform array.
  waveform_start_second: float = 0.0
  # The exclusive ending second for the segment. The
  # slice of sound segment is `[start:end]`. Defaults to the
  # maximum float32 value to signify "to the end of the waveform."
  waveform_end_second: float = np.finfo(np.float32).max


@dataclasses.dataclass
class Sound:
  """A sound with context."""

  waveform: (
      jaxtyping.Float[jaxtyping.Array, "T"]
      | jaxtyping.Int[jaxtyping.Array, "T"]
  )
  context: SoundContextParams

  @property
  def size_bytes(self) -> int:
    """Returns the size of the waveform in bytes."""
    return self.waveform.size * self.waveform.dtype.itemsize


@dataclasses.dataclass
class SoundEmbedding:
  """A sound embedding with context."""
  # N embeddings, where the embeddings are either all float vectors or strings.
  embedding: (
      jaxtyping.Float[jaxtyping.Array, "N D"]
      | jaxtyping.Shaped[np.ndarray, "N"]
  )
  # Each row is a timestamp, i.e., a [start, end] pair indicating a segment
  # where start and end are in seconds.
  # There are two common cases for the relation between embeddings (n) and
  # timestamps (m):
  #    - Frame-Aligned (m == n): The i-th timestamp corresponds directly to the
  #      i-th embedding vector.
  #    - Utterance-Level (m == 1): A single timestamp pair represents the start
  #      and end of the entire audio segment from which the embeddings were
  #      extracted.
  timestamps: jaxtyping.Float[jaxtyping.Array, "M 2"]
  context: SoundContextParams
  encoding_stats: Optional[EncodingStats] = None

  @property
  def size_bytes(self) -> int:
    """Returns the size of the embedding in bytes."""
    return self.embedding.size * self.embedding.dtype.itemsize


@dataclasses.dataclass
class TextEmbedding:
  """A dataclass for text embeddings."""
  # N embeddings, where the embeddings are either all float vectors or strings.
  embedding: (
      jaxtyping.Float[jaxtyping.Array, "N D"]
      | jaxtyping.Shaped[np.ndarray, "N"]
  )
  # Each row is a span, i.e., a tuple of the start (inclusive) and end
  # (exclusive) index of the span in the text (in characters).
  # There are two common cases for the relation between embeddings (n) and
  # spans (m):
  #    - Character-Aligned (m == n): The i-th span corresponds directly to the
  #      i-th embedding vector.
  #    - Text-Level (m == 1): A single span pair represents the start and end of
  #      the entire text segment from which the embeddings were extracted.
  spans: jaxtyping.Int[jaxtyping.Array, "M 2"]
  context: TextContextParams
  encoding_stats: Optional[EncodingStats] = None

  @property
  def size_bytes(self) -> int:
    """Returns the size of the embedding in bytes."""
    return self.embedding.size * self.embedding.dtype.itemsize


@dataclasses.dataclass
class TextWithTitleAndContext(Text):
  """A text with title and context."""
  title_text: str | None = None
  context_text: str | None = None


@dataclasses.dataclass
class SoundWithTitleAndContext(Sound):
  """A sound with title and context."""
  title_text: str | None = None
  context_text: str | None = None

  @property
  def size_bytes(self) -> int:
    """Returns the size of the waveform in bytes."""
    return self.waveform.size * self.waveform.dtype.itemsize


@dataclasses.dataclass
class SoundEmbeddingWithTitleAndContext(SoundEmbedding):
  """A sound embedding with title and context."""
  title_text: str | None = None
  context_text: str | None = None


@dataclasses.dataclass(frozen=True)
class Score:
  """A dataclass for a single evaluation metric."""

  metric: str
  description: str
  value: float
  min: int | float
  max: int | float
  std: float | None = None

  def __post_init__(self):
    """Validates the score data."""
    if not isinstance(self.metric, str) or not self.metric:
      raise TypeError("Score 'metric' must be a non-empty string. {self}")
    if not isinstance(self.description, str):
      raise TypeError("Score 'description' must be a string. {self}")
    if not isinstance(self.value, float):
      raise TypeError(f"Score 'value' must be a float. {self}")
    if not isinstance(self.min, (int, float)) or not isinstance(
        self.max, (int, float)
    ):
      raise TypeError("Score 'min' and 'max' must be numbers.")
    if self.min > self.max:
      raise ValueError(
          f"Score 'min' ({self.min}) cannot be greater than 'max' ({self.max})."
      )
    if self.std is not None and (
        not isinstance(self.std, float) or self.std < 0
    ):
      raise ValueError(
          f"Score 'std' ({self.std}) must be a non-negative float."
      )


@dataclasses.dataclass(frozen=True)
class WeightedValue:
  """A dataclass for a single weighted value."""
  value: float
  weight: float = 1.0


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


MultiModalEmbedding = SoundEmbedding | TextEmbedding
MultiModalObject = Sound | Text | MultiModalEmbedding
MultiModalEmbeddingCache = Mapping[str, MultiModalEmbedding]
