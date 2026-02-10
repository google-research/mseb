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

"""A generic MsebDataset implementation for reading data from Parquet files."""

from typing import Any, Mapping

from etils import epath
from mseb import dataset
from mseb import types
from mseb.datasets import base
import pandas as pd


class ParquetDataset(base.MsebDataset):
  """A dataset that reads task data from a Parquet file."""

  def __init__(
      self,
      dataset_name: str,
      task_name: str,
      filename: str,
      split: str = 'test',
      base_path: str | None = None,
      sample_n: int | None = None,
      id_key: str = 'id',
  ):
    """Initializes the ParquetDataset.

    Args:
      dataset_name: The name of the original dataset.
      task_name: The name of the task.
      filename: The name of the parquet file.
      split: The dataset split (e.g., 'test').
      base_path: The base path where the dataset is stored.
      sample_n: The number of examples to sample from the dataset.
      id_key: The key to use for the Sound ID.
    """
    base_path = dataset.get_base_path(base_path)
    super().__init__(base_path=base_path, split=split)
    self.id_key = id_key
    self.dataset_name = dataset_name
    self.task_name = task_name

    assert self.base_path is not None
    if self.base_path.startswith('https://'):
      # For https://, we can't use epath.Path, so we just concatenate.
      self.parquet_path = self.base_path.rstrip('/') + '/' + filename
    else:
      self.parquet_path = epath.Path(self.base_path) / filename
    self._data = self._load_data()
    if sample_n:
      self._data = self._data.sample(n=sample_n, random_state=42)

  @property
  def metadata(self) -> types.DatasetMetadata:
    """Returns the structured metadata for the dataset."""
    # TODO(tombagby): Store metadata in the parquet file.
    return types.DatasetMetadata(
        name=f'Parquet_{self.dataset_name}_{self.task_name}',
        description=(
            'A generic dataset loaded from a Parquet file produced by the'
            ' downsample_dataset.py script.'
        ),
        homepage='',
        version='1.0.0',
        license='',
        mseb_tasks=[self.task_name],
    )

  def __len__(self) -> int:
    return len(self._data)

  def _load_data(self) -> pd.DataFrame:
    """Loads the data from the Parquet file."""
    return pd.read_parquet(self.parquet_path)

  def get_task_data(
      self, task_name: str | None = None, dtype: Mapping[str, Any] | None = None
  ) -> pd.DataFrame:
    """Returns the task data.

    Args:
      task_name: The name of the task. Must match the task_name this dataset was
        initialized with.
      dtype: The data types of the columns (not used).
    """
    if task_name and task_name != self.task_name:
      raise ValueError(
          f'This dataset only supports task "{self.task_name}", '
          f'but you requested "{task_name}".'
      )
    return self._data

  def get_sound(self, record: dict[str, Any]) -> types.Sound:
    """Retrieves a Sound object from a record (row) of the dataset."""
    if 'audio' not in record:
      record = (
          self._data.loc[self._data['utt_id'] == record['utt_id']]
          .iloc[0]
          .to_dict()
      )
    audio_data = record['audio']
    waveform = audio_data['waveform']
    sample_rate = audio_data['sample_rate']

    context = types.SoundContextParams(
        id=str(record.get(self.id_key, 'id_not_found')),
        sample_rate=sample_rate,
        length=len(waveform),
    )
    return types.Sound(waveform=waveform, context=context)
