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

"""Doppelganger synthetic-real sound-pair dataset."""

from collections.abc import Callable, Mapping
import os
from typing import Any

from etils import epath
from huggingface_hub import hf_hub_download
from mseb import dataset
from mseb import types
from mseb import utils
from mseb.datasets import base
import pandas as pd

_REPO_ID = 'elliottash/doppelganger'
_REVISION = 'mseb-v1'
_METADATA_PATH = 'mseb/test.csv'

DEFAULT_BASE_PATH = None


class DoppelgangerDataset(base.MsebDataset):
  """The fixed 3,065-pair MSEB test configuration of Doppelganger."""

  def __init__(
      self,
      split: str = 'test',
      base_path: str | None = None,
      repo_id: str = _REPO_ID,
      revision: str = _REVISION,
      filter_fn: Callable[[Any], bool] | None = None,
  ):
    """Initializes the instance.

    Args:
      split: Dataset split to load (only `'test'` is supported).
      base_path: Root directory containing local Doppelganger metadata and audio
        files. If `None`, resolves via `dataset.get_base_path` using
        `DEFAULT_BASE_PATH` (or falls back to Hugging Face Hub if unconfigured).
      repo_id: Hugging Face Hub dataset repository ID.
      revision: Hugging Face Hub dataset revision.
      filter_fn: Optional predicate applied to each `pair_id` to filter the
        loaded dataset rows. If `None`, all rows are retained.
    """
    if split != 'test':
      raise ValueError(f'Split must be test, but got {split}.')

    super().__init__(base_path=base_path, split=split)
    try:
      self.base_path = dataset.get_base_path(
          self.base_path, default_basepath=DEFAULT_BASE_PATH
      )
    except ValueError:
      self.base_path = None
    self.repo_id = repo_id
    self.revision = revision
    self._filter_fn = filter_fn
    self._data = self._load_metadata()

  @property
  def metadata(self) -> types.DatasetMetadata:
    return types.DatasetMetadata(
        name='Doppelganger',
        description=(
            'Doppelganger pairs real sound effects with audio-conditioned '
            'synthetic twins across 34 UCS event categories.'
        ),
        homepage='https://huggingface.co/datasets/elliottash/doppelganger',
        version='mseb-v1',
        license=(
            'Mixed: clip-level Creative Commons licenses for real audio and '
            'the Stability AI Community License for synthetic audio'
        ),
        mseb_tasks=['retrieval'],
        citation="""
@article{ash2026doppelganger,
  author = {Elliott Ash},
  title = {Doppelganger: Sound Effects and Their Synthetic Twins},
  journal = {arXiv preprint arXiv:2607.04337},
  year = {2026}
}
""",
    )

  def __len__(self) -> int:
    return len(self._data)

  def _load_metadata(self) -> pd.DataFrame:
    local_metadata_path = None
    if self.base_path is not None:
      local_metadata_path = os.path.join(self.base_path, _METADATA_PATH)

    if local_metadata_path and epath.Path(local_metadata_path).exists():
      with epath.Path(local_metadata_path).open('r') as f:
        data = pd.read_csv(
            f,
            dtype={'pair_id': str, 'source_clip_id': str},
        )
    elif (
        DEFAULT_BASE_PATH is not None
        and self.base_path == DEFAULT_BASE_PATH
    ):
      raise FileNotFoundError(
          f'Doppelganger metadata not found at {local_metadata_path} in'
          ' default base path.'
      )
    else:
      metadata_path = (
          f'https://huggingface.co/datasets/{self.repo_id}/resolve/'
          f'{self.revision}/{_METADATA_PATH}'
      )
      data = pd.read_csv(
          metadata_path,
          dtype={'pair_id': str, 'source_clip_id': str},
      )

    if data['pair_id'].duplicated().any():
      raise ValueError('Doppelganger metadata contains duplicate pair IDs.')
    if self._filter_fn is not None:
      data = data[data['pair_id'].apply(self._filter_fn)].reset_index(
          drop=True
      )
    return data

  def get_task_data(
      self, task_name: str | None = None, dtype: Mapping[str, Any] | None = None
  ) -> pd.DataFrame:
    del task_name
    return self._data.astype(dtype) if dtype else self._data

  @staticmethod
  def sound_id(pair_id: str | int, domain: str) -> str:
    if domain not in ('real', 'synthetic'):
      raise ValueError(f'Unknown Doppelganger audio domain: {domain}.')
    return f'{domain}:{pair_id}'

  def _audio_path(self, record: dict[str, Any], domain: str) -> str:
    path_key = f'{domain}_audio_path'
    repo_key = f'{domain}_repo_id'
    revision_key = f'{domain}_revision'
    relative_path = str(record[path_key])

    if self.base_path is not None:
      local_path = os.path.join(self.base_path, relative_path)
      if epath.Path(local_path).exists():
        return local_path
      if (
          DEFAULT_BASE_PATH is not None
          and self.base_path == DEFAULT_BASE_PATH
      ):
        raise FileNotFoundError(
            f'Doppelganger audio file not found at {local_path} in default'
            ' base path.'
        )

    return hf_hub_download(
        repo_id=str(record[repo_key]),
        filename=relative_path,
        revision=str(record[revision_key]),
        repo_type='dataset',
    )

  def _get_domain_sound(
      self, record: dict[str, Any], domain: str
  ) -> types.Sound:
    waveform, sample_rate = utils.read_audio(self._audio_path(record, domain))
    context = types.SoundContextParams(
        id=self.sound_id(record['pair_id'], domain),
        sample_rate=sample_rate,
        length=len(waveform),
        waveform_end_second=(
            len(waveform) / sample_rate if sample_rate > 0 else 0.0
        ),
    )
    return types.Sound(waveform=waveform, context=context)  # pyrefly: ignore[bad-argument-type]

  def get_real_sound(self, record: dict[str, Any]) -> types.Sound:
    return self._get_domain_sound(record, 'real')

  def get_synthetic_sound(self, record: dict[str, Any]) -> types.Sound:
    return self._get_domain_sound(record, 'synthetic')

  def get_sound(self, record: dict[str, Any]) -> types.Sound:
    if 'domain' not in record:
      raise ValueError(
          'Doppelganger records passed to get_sound() must include a domain.'
      )
    return self._get_domain_sound(record, str(record['domain']))


def is_member_of_debug(pair_id: str | int) -> bool:
  """Returns whether the Doppelganger pair ID is in the debug set."""
  debug_ids = frozenset([
      '168765981935',
      '204643997502',
      '218906114721',
      '224092971726',
      '292264958284',
  ])
  return str(pair_id) in debug_ids
