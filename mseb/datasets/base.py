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

"""Abstract base class for MSEB datasets."""

import abc
from typing import Any, Mapping
from mseb import types
import pandas as pd


class MsebDataset(abc.ABC):
  """Abstract base class for MSEB datasets."""

  @abc.abstractmethod
  def __init__(self, base_path: str | None, split: str):
    self.base_path = base_path
    self.split = split

  @property
  @abc.abstractmethod
  def metadata(self) -> types.DatasetMetadata:
    """Returns the structured metadata for the dataset."""
    pass

  @abc.abstractmethod
  def __len__(self) -> int:
    pass

  @abc.abstractmethod
  def get_task_data(
      self, task_name: str | None = None, dtype: Mapping[str, Any] | None = None
  ) -> pd.DataFrame:
    """Loads the task data for a given task name.

    Args:
      task_name: The name of the task. If None, returns the entire dataset.
      dtype: The data types of the columns.
    """
    pass

  @abc.abstractmethod
  def get_sound(self, record: dict[str, Any]) -> types.Sound:
    """Retrieves a Sound object from a record (row) of the dataset."""
    pass
