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

from mseb import dataset
from mseb import types
from mseb import utils
import numpy as np
import pandas as pd


class FSD50KDataset(dataset.Dataset):
  """FSD50K dataset loader that works with the Hugging Face repository.

  This class loads data directly from a Hugging Face repository (e.g.,
  'speechcolab/fsd50k') and requires the official 'vocabulary.csv' file
  to be present locally for label mapping.
  """

  def __init__(
      self,
      base_path: str,
      split: str,
      repo_id: str = 'Fhrozen/FSD50k',
  ):
    """Initializes the dataset for a specific split from Hugging Face.

    Args:
      base_path: The root directory to use as a cache for Hugging Face
        downloads. This directory should also contain 'labels/vocabulary.cs´v'.
      split: The dataset split to load. Must be 'validation' or 'test'.
      repo_id: The Hugging Face repository ID to download from.
    """
    if split not in ['validation', 'test']:
      raise ValueError(
          f'Split must be validation or test, but got {split}.'
      )
    self.repo_id = repo_id
    super().__init__(base_path=base_path, split=split, target_sr=None)
    self._load_vocabulary()

  @property
  def metadata(self) -> types.DatasetMetadata:
    """Returns the structured metadata for the FSD50K dataset."""
    return types.DatasetMetadata(
        name='FSD50K',
        description=(
            'FSD50K is an open dataset of 51,197 human-labeled sound events '
            'from Freesound, annotated with 200 classes from the AudioSet '
            'Ontology. All clips are weakly labeled and can have multiple '
            'labels.'
        ),
        homepage='https://huggingface.co/datasets/Fhrozen/FSD50k',
        version='1.0',
        license='Creative Commons Attribution-NonCommercial 4.0',
        mseb_tasks=['classification', 'clustering', 'retrieval'],
        citation="""
@article{fonseca2022fsd50k,
  author    = {Eduardo Fonseca and
               Xavier Favory and
               Jordi Pons and
               Frederic Font and
               Xavier Serra},
  title     = {{FSD50K}: an Open Dataset of Human-Labeled Sound Events},
  journal   = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume    = {30},
  pages     = {829--852},
  year      = {2022},
  doi       = {10.1109/TASLP.2022.3149014}
}
""",
    )

  @property
  def class_labels(self) -> list[str]:
    """Returns the ordered list of all 200 class labels."""
    return self._class_labels

  def _path(self, *args):
    return os.path.join(self.base_path, *args)

  def _load_vocabulary(self) -> None:
    """Loads the vocabulary file and creates the label-to-ID mapping."""
    vocab_path = self._path('labels', 'vocabulary.csv')
    if not os.path.exists(vocab_path):
      raise FileNotFoundError(
          f'Vocabulary file not found at {vocab_path}. Please download the '
          'official FSD50K files and place vocabulary.csv in the '
          'labels subdirectory of your base_path.'
      )
    vocab_df = pd.read_csv(
        vocab_path,
        header=None,
        names=['index', 'mid', 'display_name']
    )
    self._class_labels = vocab_df['display_name'].tolist()
    self.label_to_id = {label: i for i, label in enumerate(self._class_labels)}

  def _load_eval(self):
    return pd.read_csv(self._path('labels', 'eval.csv'))

  def _load_metadata(self) -> pd.DataFrame:
    """Loads the dataset for the split using the HF datasets library."""
    cache_path = self._path(f'{self.split}.parquet')
    if os.path.exists(cache_path):
      return pd.read_parquet(cache_path)
    utils.download_from_hf(self.repo_id, self.base_path)
    return self._load_eval()

  def get_string_labels(self, index: int) -> list[str]:
    """Returns the raw string labels for a given index."""
    record = self._metadata.iloc[index]
    return record['labels'].split(',')

  def get_multi_hot_labels(self, index: int) -> np.ndarray:
    """Returns the multi-hot encoded label vector for a given index."""
    string_labels = self.get_string_labels(index)
    num_classes = len(self.class_labels)
    multi_hot_vector = np.zeros(num_classes, dtype=np.int64)
    for label in string_labels:
      if label in self.label_to_id:
        class_id = self.label_to_id[label]
        multi_hot_vector[class_id] = 1
    return multi_hot_vector

  def _load_wav_for_row(self, row):
    fname = row['fname']
    clip_path = self._path('clips', f'{fname}.wav')
    clip_bytes = open(clip_path, 'rb').read()
    waveform, sr = utils.wav_bytes_to_waveform(clip_bytes)
    return waveform, sr

  def load_sounds(self):
    """Loads all sounds from disk and adds them to the metadata."""
    self._metadata[['waveform', 'sample_rate']] = self._metadata.apply(
        self._load_wav_for_row, axis=1, result_type='expand')
    cache_path = self._path(f'{self.split}.parquet')
    self._metadata.to_parquet(cache_path)

  def _get_sound(self, record: dict[str, Any]) -> types.Sound:
    """Extracts a Sound object from a metadata record loaded by HF."""
    if 'waveform' in record and 'sample_rate' in record:
      waveform = record['waveform']
      sr = record['sample_rate']
    else:
      waveform, sr = self._load_wav_for_row(record)
    context = types.SoundContextParams(
        id=str(record['fname']),
        sample_rate=sr,
        length=len(waveform),
        language=None,
        speaker_id=None,
        speaker_age=None,
        speaker_gender=None,
        text=None,
        waveform_start_second=0.0,
        waveform_end_second=len(waveform) / sr if sr > 0 else 0.0,
    )
    return types.Sound(waveform=waveform, context=context)
