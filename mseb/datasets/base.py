# Copyright 2026 The MSEB Authors.
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
from collections.abc import Iterable, Mapping, Sequence
from concurrent import futures
from typing import Any

import apache_beam as beam
from mseb import types
import pandas as pd


class ThreadedLoadSoundsDoFn(beam.DoFn):
  """Loads sounds concurrently across batches via a persistent thread pool."""

  def __init__(
      self, dataset: 'MsebDataset', max_workers: int = 16
  ) -> None:
    """Initializes the DoFn with a sound dataset and worker pool size."""
    super().__init__()
    self._dataset = dataset
    self._max_workers = max_workers
    self._executor: futures.ThreadPoolExecutor | None = None

  def __getstate__(self) -> dict[str, Any]:
    """Drops the live thread pool before Beam DoFn serialization."""
    state = self.__dict__.copy()
    state['_executor'] = None
    return state

  def setup(self) -> None:
    """Creates a persistent thread pool on the Beam worker."""
    if self._executor is None:
      self._executor = futures.ThreadPoolExecutor(
          max_workers=self._max_workers
      )

  def teardown(self) -> None:
    """Shuts down the worker thread pool when the DoFn tears down."""
    if self._executor is not None:
      self._executor.shutdown(wait=True)
      self._executor = None

  def _load_sound(self, record: Mapping[str, Any]) -> types.Sound:
    return self._dataset.get_sound(dict(record))

  def process(
      self, batch: Sequence[Mapping[str, Any]]
  ) -> Iterable[types.Sound]:
    """Loads sounds for a batch of dataset records concurrently."""
    if self._executor is not None:
      sounds = list(self._executor.map(self._load_sound, batch))
    else:
      with futures.ThreadPoolExecutor(
          max_workers=self._max_workers
      ) as executor:
        sounds = list(executor.map(self._load_sound, batch))

    yield from sounds


class MsebDataset(abc.ABC):
  """Abstract base class for MSEB datasets."""

  # Subclasses setting `beam_load_threads > 1` must implement a thread-safe
  # `get_sound()` and a pickle-safe `__getstate__()`.
  beam_load_threads: int = 1

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

  def get_task_sounds_beam(
      self,
      task_name: str | None = None,
  ) -> beam.PTransform:
    """Returns a Beam PTransform that loads `types.Sound` records for a task."""
    records = self.get_task_data(task_name).to_dict('records')
    if self.beam_load_threads > 1:
      return (
          beam.Create(records)
          | 'BatchRecords'
          >> beam.BatchElements(min_batch_size=10, max_batch_size=50)
          | 'ThreadedLoadSounds'
          >> beam.ParDo(
              ThreadedLoadSoundsDoFn(self, max_workers=self.beam_load_threads)
          )
      )
    return beam.Create(records) | 'LoadSound' >> beam.Map(self.get_sound)
