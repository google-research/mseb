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

"""MSEB dataset base class."""

import abc
from typing import Iterator, Optional, Any

from absl import flags
from mseb import types
import numpy as np
import pandas as pd


_DATASET_BASEPATH = flags.DEFINE_string(
    "dataset_basepath",
    None,
    "Path to the MSEB dataset cache.",
)


def get_base_path(basepath: str | None = None) -> str:
  """Return basepath from argument or flag."""
  if basepath is not None:
    return basepath
  if _DATASET_BASEPATH.value is not None:
    return _DATASET_BASEPATH.value
  raise ValueError(
      "basepath must be provided either as an argument or through the"
      " --dataset_basepath flag."
  )


class BatchIterator:
  """A simple batch iterator for Dataset."""

  def __init__(self,
               dataset: "Dataset",
               batch_size: int,
               shuffle: bool = False):
    self.dataset = dataset
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.indices = list(range(len(dataset)))

  def __iter__(self) -> Iterator[list[types.Sound]]:
    if self.shuffle:
      np.random.shuffle(self.indices)
    for i in range(0, len(self.indices), self.batch_size):
      batch_indices = self.indices[i : i + self.batch_size]
      yield [self.dataset[idx] for idx in batch_indices]

  def __len__(self) -> int:
    return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class Dataset(abc.ABC):
  """Abstract Base Class for all benchmark datasets."""

  def __init__(self,
               split: str,
               base_path: Optional[str] = None):
    self._base_path = base_path
    self.split = split
    self._metadata = self._load_metadata()

  @property
  def base_path(self) -> str:
    """Returns the base path for caching this dataset."""
    return get_base_path(self._base_path)

  # --- Methods for child classes to implement ---
  @property
  @abc.abstractmethod
  def metadata(self) -> types.DatasetMetadata:
    """Returns the structured metadata for the dataset."""
    ...

  @abc.abstractmethod
  def _load_metadata(self) -> pd.DataFrame:
    """Loads the low-level metadata for the specified split into a DataFrame."""
    ...

  @abc.abstractmethod
  def _get_sound(self, record: dict[str, Any]) -> types.Sound:
    """Takes a single metadata record and produces a `Sound` object."""
    ...

  # --- Shared functionality for all datasets ---
  def __len__(self) -> int:
    return len(self._metadata)

  def __getitem__(self, index: int) -> types.Sound:
    """Gets a single item by its integer index."""
    if index >= len(self):
      raise IndexError("Index out of range")
    record = self._metadata.iloc[index].to_dict()
    return self._get_sound(record)

  def __iter__(self) -> Iterator[types.Sound]:
    """Iterates through the dataset, yielding Sound objects one by one."""
    for i in range(len(self)):
      yield self[i]

  @property
  def available_labels(self) -> list[str]:
    """Returns a list of all available label/metadata column names."""
    return list(self._metadata.columns)

  def get_labels(self, name: str) -> list[Any]:
    """Retrieves all labels for a given column name."""
    if name not in self.available_labels:
      raise ValueError(
          f"Label '{name}' not found. "
          f"Available labels are: {self.available_labels}"
      )
    return self._metadata[name].tolist()

  def as_batch_iterator(
      self, batch_size: int,
      shuffle: bool = False
  ) -> BatchIterator:
    """Returns a convenient BatchIterator instance for this dataset."""
    return BatchIterator(self, batch_size=batch_size, shuffle=shuffle)
