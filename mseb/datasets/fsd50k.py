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

"""FSD50K dataset."""

import os
from typing import Any

from absl import logging
from mseb import dataset
from mseb import types
from mseb import utils
import pandas as pd


class FSD50KDataset:
  """FSD50K dataset loader that works with the Hugging Face repository."""

  def __init__(
      self,
      split: str,
      base_path: str | None = None,
      repo_id: str = 'Fhrozen/FSD50k',
  ):
    if split not in ['validation', 'test']:
      raise ValueError(f'Split must be validation or test, but got {split}.')

    self.base_path = dataset.get_base_path(base_path)
    self.split = split
    self._clip_dir = 'eval' if split == 'test' else 'dev'
    self.repo_id = repo_id

    self._data = self._load_metadata()
    self._load_vocabulary()

  def __len__(self) -> int:
    return len(self._data)

  @property
  def metadata(self) -> types.DatasetMetadata:
    return types.DatasetMetadata(
        name='FSD50K',
        description=(
            'FSD50K is an open dataset of 51,197 human-labeled sound events '
            'from Freesound, annotated with 200 classes from the AudioSet '
            'Ontology.'
        ),
        homepage='https://huggingface.co/datasets/Fhrozen/FSD50k',
        version='1.0',
        license='Creative Commons Attribution-NonCommercial 4.0',
        mseb_tasks=['classification', 'clustering', 'retrieval'],
        citation="""
@article{fonseca2022fsd50k,
  author    = {Eduardo Fonseca and Xavier Favory and Jordi Pons and Frederic Font and Xavier Serra},
  title     = {{FSD50K}: an Open Dataset of Human-Labeled Sound Events},
  journal   = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume    = {30},
  pages     = {829--852},
  year      = {2021},
  doi       = {10.1109/TASLP.2022.3149014}
}
""",
    )

  @property
  def class_labels(self) -> list[str]:
    return self._class_labels

  def _path(self, *args):
    return os.path.join(self.base_path, *args)

  def get_task_data(self) -> pd.DataFrame:
    return self._data

  def _load_vocabulary(self) -> None:
    vocab_path = self._path('labels', 'vocabulary.csv')
    if not os.path.exists(vocab_path):
      raise FileNotFoundError(f'Vocabulary file not found at {vocab_path}.')
    vocab_df = pd.read_csv(
        vocab_path, header=None, names=['index', 'display_name', 'mid']
    )
    self._class_labels = vocab_df['display_name'].tolist()
    self.label_to_id = {label: i for i, label in enumerate(self._class_labels)}

  def _load_csv(self):
    """Loads the appropriate FSD50K metadata CSV for the configured split.

    This method handles the logic for selecting and processing the correct
    metadata file based on the dataset's split.

    - For the 'test' split, it loads `eval.csv`, which contains only test data.
    - For the 'validation' split, it loads `dev.csv` and then filters it to
      only include rows where the 'split' column is marked as 'val'.

    Returns:
      A pandas DataFrame containing the metadata for the requested split.
    """
    if self.split == 'test':
      # eval.csv is for the 'test' split and has no 'split' column.
      # We load it directly without filtering.
      csv_path = self._path('labels', 'eval.csv')
      df = pd.read_csv(csv_path)
    elif self.split == 'validation':
      # dev.csv contains multiple splits, so we must load and then filter it.
      csv_path = self._path('labels', 'dev.csv')
      df = pd.read_csv(csv_path)
      df = df[df['split'] == 'val']
    else:
      # This case should not be reached due to the check in __init__
      raise ValueError(
          f'Unsupported split {self.split} for _load_csv'
      )
    return df

  def _load_metadata(self) -> pd.DataFrame:
    """Loads the dataset metadata, utilizing a parquet cache for speed.

    This method first checks for a pre-processed `.parquet` cache file for
    the specified split. If the cache is found, it's loaded directly for a
    fast startup.

    If the cache is not found, it triggers a one-time download of the raw
    dataset from Hugging Face, processes the data via `_load_csv`, and saves
    the result to a new `.parquet` file to accelerate all future loads.

    Returns:
      A pandas DataFrame containing the dataset's metadata.
    """
    cache_path = self._path(f'{self.split}.parquet')
    if os.path.exists(cache_path):
      logging.info(
          'Loading FSD50K %s split from cache...', self.split
      )
      return pd.read_parquet(cache_path)
    logging.info(
        'Cache not found. Processing FSD50K %s split from source...'
    )
    utils.download_from_hf(self.repo_id, self.base_path)
    df = self._load_csv()
    logging.info(
        'Saving FSD50K %s split to cache at %s', self.split, cache_path
    )
    df.to_parquet(cache_path)
    return df

  def _load_wav_for_row(self, row):
    fname = row['fname']
    clip_path = self._path('clips', self._clip_dir, f'{fname}.wav')
    clip_bytes = open(clip_path, 'rb').read()
    waveform, sr = utils.wav_bytes_to_waveform(clip_bytes)
    return waveform, sr

  def get_sound(self, record: dict[str, Any]) -> types.Sound:
    if 'waveform' in record and 'sample_rate' in record:
      waveform, sr = record['waveform'], record['sample_rate']
    else:
      waveform, sr = self._load_wav_for_row(record)
    context = types.SoundContextParams(
        id=str(record['fname']),
        sample_rate=sr,
        length=len(waveform),
        waveform_end_second=len(waveform) / sr if sr > 0 else 0.0,
    )
    return types.Sound(waveform=waveform, context=context)
