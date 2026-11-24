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

"""FSD50K dataset."""

from collections.abc import Callable, Mapping
import json
import os
from typing import Any
import urllib.request

from absl import logging
from etils import epath
from mseb import dataset
from mseb import types
from mseb import utils
from mseb.datasets import base
import pandas as pd
from pyarrow import parquet as pq

_PARQUET_BATCH_SIZE = 1024
# Read Parquet column chunks in blocks of this size instead of all at once.
# Each read from remote storage has a high fixed cost, so smaller blocks are
# slower.
# In measurements, 1 GiB blocks were slower, too.
_PARQUET_BUFFER_SIZE = 256 * 1024 * 1024

# Single source of truth for the FSD50K dataset versions/revisions (referenced
# by task metadata `types.Dataset.revision`).
FULL_VERSION = '1.0.0'
COMMERCIAL_VERSION = '2.0.0'

# Creative Commons licenses permitting commercial use (excluding
# `CC BY-NC 3.0` and `CC Sampling+ 1.0`).
COMMERCIAL_LICENSE_URLS: frozenset[str] = frozenset({
    'http://creativecommons.org/publicdomain/zero/1.0/',
    'https://creativecommons.org/publicdomain/zero/1.0/',
    'http://creativecommons.org/licenses/by/3.0/',
    'https://creativecommons.org/licenses/by/3.0/',
    'http://creativecommons.org/licenses/by/4.0/',
    'https://creativecommons.org/licenses/by/4.0/',
})

# Parquet schema metadata marker written onto pre-filtered commercial-only
# Parquet caches so `commercial_only=False` refuses to read them silently.
COMMERCIAL_ONLY_PARQUET_METADATA_KEY = b'mseb_commercial_only'
COMMERCIAL_ONLY_PARQUET_METADATA_VALUE = b'true'

DEFAULT_BASE_PATH = None


def is_commercial_license(license_url: str | None) -> bool:
  """Returns whether a Creative Commons license URL permits commercial use.

  Retains `CC0 1.0`, `CC BY 3.0`, and `CC BY 4.0`, and excludes non-commercial
  licenses such as `CC BY-NC 3.0` and `CC Sampling+ 1.0`.

  Args:
    license_url: License URL string from `FSD50K.metadata` or a Parquet
      `license` column.
  """
  if not isinstance(license_url, str):
    return False
  return license_url.strip() in COMMERCIAL_LICENSE_URLS


def load_clip_licenses(metadata_path: str) -> dict[str, str]:
  """Loads clip ID to license URL mappings from an FSD50K metadata JSON file.

  Args:
    metadata_path: Path to `eval_clips_info_FSD50K.json` or
      `dev_clips_info_FSD50K.json`.

  Returns:
    Mapping from string clip ID (`fname`) to license URL string.
  """
  with epath.Path(metadata_path).open('r') as f:
    clips_info = json.load(f)
  return {
      str(clip_id): str(info.get('license', '')).strip()
      for clip_id, info in clips_info.items()
  }


class FSD50KDataset(base.MsebDataset):
  """FSD50K dataset loader that works with the Hugging Face repository."""

  def __init__(
      self,
      split: str,
      base_path: str | None = None,
      repo_id: str = 'Fhrozen/FSD50k',
      streaming: bool = False,
      filter_fn: Callable[[Any], bool] | None = None,
      commercial_only: bool = True,
  ):
    """Initializes the FSD50K dataset.

    Args:
      split: Dataset split ('validation' or 'test').
      base_path: Root directory containing the dataset files or cache. If None,
        falls back to `--dataset_basepath`, then `DEFAULT_BASE_PATH` when
        `commercial_only=True` and a built-in default is configured.
      repo_id: Hugging Face repository ID.
      streaming: Whether to stream data from Hugging Face.
      filter_fn: Optional predicate over `fname` to filter loaded examples.
      commercial_only: If True, drops clips carrying non-commercial licenses
        (`CC BY-NC 3.0` and `CC Sampling+ 1.0`) and retains only `CC0 1.0` and
        `CC BY 3.0` clips.
    """
    if split not in ['validation', 'test']:
      raise ValueError(f'Split must be validation or test, but got {split}.')

    super().__init__(base_path=base_path, split=split)
    default_basepath = DEFAULT_BASE_PATH if commercial_only else None
    self.base_path = dataset.get_base_path(
        self.base_path, default_basepath=default_basepath
    )
    self._clip_dir = 'eval' if split == 'test' else 'dev'
    self.repo_id = repo_id
    self.streaming = streaming
    self._commercial_only = commercial_only
    # If set, only the examples whose `fname` satisfies `filter_fn` are loaded.
    # `fname` is passed as stored, which is an `int` for the FSD50K metadata.
    self._filter_fn = filter_fn

    self._data = self._load_metadata()
    if (self._commercial_only or self._filter_fn is not None) and (
        self._data.empty
    ):
      logging.warning(
          'No FSD50K %s examples remain after filtering (commercial_only=%s).',
          self.split,
          self._commercial_only,
      )
    self._load_vocabulary()

  def __len__(self) -> int:
    return len(self._data)

  @property
  def metadata(self) -> types.DatasetMetadata:
    clip_license = (
        'CC0 1.0 and CC BY 3.0 (commercial-use clip subset)'
        if self._commercial_only
        else 'per-clip Creative Commons licenses'
    )
    return types.DatasetMetadata(
        name='FSD50K',
        description=(
            'FSD50K is an open dataset of 51,197 human-labeled sound events '
            'from Freesound, annotated with 200 classes from the AudioSet '
            'Ontology.'
        ),
        homepage='https://huggingface.co/datasets/Fhrozen/FSD50k',
        version=(
            COMMERCIAL_VERSION if self._commercial_only else FULL_VERSION
        ),
        license=f'CC BY 4.0 (dataset and annotations); {clip_license}',
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
    return os.path.join(self.base_path, *args)  # pyrefly: ignore[no-matching-overload]

  def _hf_path(self, *parts):
    """Constructs a URL for a file in the Hugging Face repository."""
    return f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/{'/'.join(parts)}"

  def get_task_data(
      self, task_name: str | None = None, dtype: Mapping[str, Any] | None = None
  ) -> pd.DataFrame:
    return self._data

  @classmethod
  def load_class_labels(
      cls,
      base_path: str | None = None,
      repo_id: str = 'Fhrozen/FSD50k',
      streaming: bool = False,
  ) -> list[str]:
    """Loads FSD50K class labels directly from vocabulary.csv."""
    if streaming:
      vocab_path = (
          f'https://huggingface.co/datasets/{repo_id}/resolve/main/'
          'labels/vocabulary.csv'
      )
      vocab_df = pd.read_csv(
          vocab_path, header=None, names=['index', 'display_name', 'mid']
      )
    else:
      resolved_base_path = dataset.get_base_path(
          base_path, default_basepath=DEFAULT_BASE_PATH
      )
      vocab_path = os.path.join(resolved_base_path, 'labels', 'vocabulary.csv')  # pyrefly: ignore[no-matching-overload]
      if not epath.Path(vocab_path).exists():
        raise FileNotFoundError(f'Vocabulary file not found at {vocab_path}.')
      with epath.Path(vocab_path).open('r') as f:
        vocab_df = pd.read_csv(
            f, header=None, names=['index', 'display_name', 'mid']
        )
    return vocab_df['display_name'].tolist()

  def _load_vocabulary(self) -> None:
    self._class_labels = self.load_class_labels(
        base_path=self.base_path,
        repo_id=self.repo_id,
        streaming=self.streaming,
    )
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
    if self.streaming:
      if self.split == 'test':
        csv_path = self._hf_path('labels', 'eval.csv')
        df = pd.read_csv(csv_path)
      elif self.split == 'validation':
        csv_path = self._hf_path('labels', 'dev.csv')
        df = pd.read_csv(csv_path)
        df = df[df['split'] == 'val']
    else:
      if self.split == 'test':
        csv_path = self._path('labels', 'eval.csv')
        with epath.Path(csv_path).open('r') as f:
          df = pd.read_csv(f)
      elif self.split == 'validation':
        csv_path = self._path('labels', 'dev.csv')
        with epath.Path(csv_path).open('r') as f:
          df = pd.read_csv(f)
        df = df[df['split'] == 'val']
      else:
        raise ValueError(f'Unsupported split {self.split} for _load_csv')
    return df  # pyrefly: ignore[unbound-name]

  def _load_clip_licenses(self) -> dict[str, str] | None:
    """Loads per-clip license URLs from `metadata/{eval,dev}_clips_info_FSD50K.json`."""
    metadata_filename = f'{self._clip_dir}_clips_info_FSD50K.json'
    if self.streaming:
      metadata_url = self._hf_path('metadata', metadata_filename)
      with urllib.request.urlopen(metadata_url, timeout=30) as f:
        clips_info = json.loads(f.read().decode('utf-8'))
      return {
          str(clip_id): str(info.get('license', '')).strip()
          for clip_id, info in clips_info.items()
      }
    metadata_path = self._path('metadata', metadata_filename)
    if not epath.Path(metadata_path).exists():
      return None
    return load_clip_licenses(metadata_path)

  def _load_metadata(self) -> pd.DataFrame:
    """Loads the dataset metadata, utilizing a parquet cache for speed.

    This method first checks for a pre-processed `.parquet` cache file for
    the specified split. If the cache is found, it's loaded directly for a
    fast startup.

    If the cache is not found, it triggers a one-time download of the raw
    dataset from Hugging Face, processes the data via `_load_csv`, attaches
    per-clip license metadata when available, and saves the result to a new
    `.parquet` file to accelerate all future loads.

    If `commercial_only` or `filter_fn` is set, only the matching rows are
    returned. When reading the cache, non-matching rows are dropped batch by
    batch before their audio is converted to pandas, and reading stops after
    the last matching row.

    Returns:
      A pandas DataFrame containing the dataset's metadata.

    Raises:
      FileNotFoundError: If the cache file is missing when using
        `DEFAULT_BASE_PATH`.
      ValueError: If `commercial_only=False` is requested against a Parquet
        cache marked as commercial-only, or if `commercial_only=True` is
        requested without a `license` column or clip metadata JSON.
    """
    cache_path = self._path(f'{self.split}.parquet')
    if epath.Path(cache_path).exists():
      logging.info(
          'Loading FSD50K %s split from cache...', self.split
      )
      with epath.Path(cache_path).open('rb') as f:
        parquet_file = pq.ParquetFile(f, buffer_size=_PARQUET_BUFFER_SIZE)
        schema_metadata = parquet_file.schema_arrow.metadata or {}
        if (
            not self._commercial_only
            and schema_metadata.get(COMMERCIAL_ONLY_PARQUET_METADATA_KEY)
            == COMMERCIAL_ONLY_PARQUET_METADATA_VALUE
        ):
          raise ValueError(
              f'FSD50K Parquet cache at {cache_path} contains only '
              'commercial-licensed clips and cannot be loaded with '
              'commercial_only=False. Point --dataset_basepath to an '
              'unfiltered FSD50K directory or load from Hugging Face.'
          )
        has_license_col = 'license' in parquet_file.schema_arrow.names
        clip_licenses: dict[str, str] | None = None
        if self._commercial_only and not has_license_col:
          clip_licenses = self._load_clip_licenses()
          if clip_licenses is None:
            metadata_filename = f'{self._clip_dir}_clips_info_FSD50K.json'
            raise ValueError(
                f'Cannot filter FSD50K {self.split} split to commercial '
                f'licenses: Parquet cache at {cache_path} has no `license` '
                f'column and {self._path("metadata", metadata_filename)} was '
                'not found.'
            )

        needs_filtering = self._commercial_only or self._filter_fn is not None

        def _matches(fname: Any, license_url: str | None) -> bool:
          if self._commercial_only:
            if has_license_col:
              resolved_license = license_url
            elif clip_licenses is not None:
              resolved_license = clip_licenses.get(str(fname))
            else:
              resolved_license = None
            if not is_commercial_license(resolved_license):
              return False
          if self._filter_fn is not None and not self._filter_fn(fname):
            return False
          return True

        num_matches = None
        all_match = True
        if needs_filtering:
          # Count the matching rows, to stop reading after the last one.
          scan_cols = (
              ['fname', 'license']
              if self._commercial_only and has_license_col
              else ['fname']
          )
          scan_table = parquet_file.read(columns=scan_cols)
          fnames = scan_table['fname'].to_pylist()
          licenses = (
              scan_table['license'].to_pylist()
              if self._commercial_only and has_license_col
              else [None] * len(fnames)
          )
          num_matches = sum(
              1 for fname, lic in zip(fnames, licenses) if _matches(fname, lic)
          )
          all_match = bool(fnames) and num_matches == len(fnames)
        dfs = []
        num_rows = 0
        for batch in parquet_file.iter_batches(batch_size=_PARQUET_BATCH_SIZE):
          if needs_filtering and not all_match:
            batch_fnames = batch['fname'].to_pylist()
            batch_licenses = (
                batch['license'].to_pylist()
                if self._commercial_only and has_license_col
                else [None] * len(batch_fnames)
            )
            batch = batch.filter([
                _matches(fname, lic)
                for fname, lic in zip(batch_fnames, batch_licenses)
            ])
          dfs.append(batch.to_pandas())
          num_rows += batch.num_rows
          if num_rows == num_matches:
            break
        if not dfs:
          return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)  # pyrefly: ignore[bad-return]
    if not self.streaming and self.base_path == DEFAULT_BASE_PATH:
      raise FileNotFoundError(
          f'FSD50K cache file not found at {cache_path} in default base path.'
      )
    logging.info(
        'Cache not found. Processing FSD50K %s split from source...',
        self.split,
    )
    if not self.streaming:
      utils.download_from_hf(self.repo_id, self.base_path)  # pyrefly: ignore[bad-argument-type]
    df = self._load_csv().copy()
    if 'license' not in df.columns and (
        self._commercial_only or not self.streaming
    ):
      clip_licenses = self._load_clip_licenses()
      if clip_licenses is not None:
        df['license'] = df['fname'].map(
            lambda fname: clip_licenses.get(str(fname), '')
        )
      elif self._commercial_only:
        metadata_filename = f'{self._clip_dir}_clips_info_FSD50K.json'
        raise ValueError(
            f'Cannot filter FSD50K {self.split} split to commercial licenses: '
            f'{self._path("metadata", metadata_filename)} was not found.'
        )
    if not self.streaming:
      logging.info(
          'Saving FSD50K %s split to cache at %s', self.split, cache_path
      )
      df.to_parquet(cache_path, row_group_size=32)
    if self._commercial_only:
      df = df[df['license'].map(is_commercial_license)]
    if self._filter_fn is not None:
      df = df[df['fname'].apply(self._filter_fn)]
    if self._commercial_only or self._filter_fn is not None:
      df = df.reset_index(drop=True)
    return df

  def _load_wav_for_row(self, row):
    fname = row['fname']
    if self.streaming:
      clip_url = self._hf_path('clips', self._clip_dir, f'{fname}.wav')

      with urllib.request.urlopen(clip_url) as f:
        clip_bytes = f.read()
    else:
      clip_path = self._path('clips', self._clip_dir, f'{fname}.wav')
      with epath.Path(clip_path).open('rb') as f:
        clip_bytes = f.read()
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


def is_member_of_debug(fname: str | int) -> bool:
  """Returns whether the FSD50K file name is in the debug set."""
  debug_ids = frozenset(['37199', '175151', '329838', '1277'])
  return str(fname) in debug_ids
