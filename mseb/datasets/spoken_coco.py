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

"""Spoken COCO dataset.

SpokenCOCO is an audio-image retrieval benchmark where each image from MS COCO
has five spoken captions. The task is to retrieve the correct image given a
spoken description.

When no base_path is provided, the dataset is automatically downloaded from:
  - SpokenCOCO audio/JSON:
  https://data.csail.mit.edu/placesaudio/SpokenCOCO.tar.gz
  - MSCOCO val2014 images: http://images.cocodataset.org/zips/val2014.zip

Reference: https://groups.csail.mit.edu/sls/downloads/placesaudio/
"""

import json
import logging
import os
from typing import Any, Mapping

from etils import epath
from mseb import dataset
from mseb import types
from mseb.datasets import audio_image_base
import pandas as pd

logger = logging.getLogger(__name__)


_SPOKEN_COCO_URL = 'https://data.csail.mit.edu/placesaudio/SpokenCOCO.tar.gz'
_MSCOCO_VAL2014_URL = 'http://images.cocodataset.org/zips/val2014.zip'


def maybe_download_spoken_coco(dest_dir: str, split: str = 'val') -> None:
  """Downloads SpokenCOCO audio and MSCOCO images to dest_dir.

  Downloads are skipped if the expected files already exist in dest_dir.

  Args:
    dest_dir: Directory to download to.
    split: The dataset split (used to verify the JSON file exists).
  """
  epath.Path(dest_dir).mkdir(parents=True, exist_ok=True)

  json_path = os.path.join(dest_dir, f'SpokenCOCO_{split}.json')
  wavs_dir = os.path.join(dest_dir, 'wavs')
  images_dir = os.path.join(dest_dir, 'val2014')

  # Download and extract SpokenCOCO (audio + JSON metadata).
  if not epath.Path(json_path).exists() or not epath.Path(wavs_dir).is_dir():
    tar_path = os.path.join(dest_dir, 'SpokenCOCO.tar.gz')
    if not epath.Path(tar_path).exists():
      audio_image_base.download_file(_SPOKEN_COCO_URL, tar_path)
    audio_image_base.extract_tar_gz(tar_path, dest_dir)
    # The tar may extract into a subdirectory; move files up if needed.
    inner_dir = os.path.join(dest_dir, 'SpokenCOCO')
    if epath.Path(inner_dir).is_dir():
      for item in epath.Path(inner_dir).iterdir():
        src = os.path.join(inner_dir, item)
        dst = os.path.join(dest_dir, item)
        if not epath.Path(dst).exists():
          epath.Path(src).rename(dst)
    logger.info('SpokenCOCO audio data ready at %s', dest_dir)
  else:
    logger.info('SpokenCOCO audio data already exists at %s', dest_dir)

  # Download and extract MSCOCO val2014 images.
  if not epath.Path(images_dir).is_dir():
    zip_path = os.path.join(dest_dir, 'val2014.zip')
    if not epath.Path(zip_path).exists():
      audio_image_base.download_file(_MSCOCO_VAL2014_URL, zip_path)
    audio_image_base.extract_zip(zip_path, dest_dir)
    logger.info('MSCOCO val2014 images ready at %s', images_dir)
  else:
    logger.info('MSCOCO val2014 images already exist at %s', images_dir)


class SpokenCocoDataset(audio_image_base.AudioImageDataset):
  """Spoken COCO dataset for audio-image retrieval.

  The dataset consists of images from MS COCO paired with multiple spoken
  captions. The metadata JSON has structure:

    {"data": [
      {
        "image": "val2014/COCO_val2014_000000325114.jpg",
        "captions": [
          {
            "text": "A URINAL IN A PUBLIC RESTROOM ...",
            "speaker": "m07150...",
            "uttid": "m07150..._325114_629297",
            "wav": "wavs/val/0/m07150..._325114_629297.wav"
          },
          ...
        ]
      },
      ...
    ]}

  If base_path is not provided, the dataset is downloaded automatically to
  a temporary directory. To persist downloads across runs, set base_path or
  the download_dir argument.
  """

  def __init__(
      self,
      base_path: str | None = None,
      split: str = 'val',
  ):
    """Initializes the SpokenCOCO dataset.

    Args:
      base_path: Path to the dataset root. If provided, data is loaded directly
        from this path (no download). If None and download=True, the data is
        downloaded automatically.
      split: Dataset split ('val' or 'train').
    """
    super().__init__(base_path=base_path, split=split)
    self.base_path = dataset.get_base_path(self.base_path)
    maybe_download_spoken_coco(dest_dir=self.base_path, split=split)
    self._data = None

  @property
  def metadata(self) -> types.DatasetMetadata:
    return types.DatasetMetadata(
        name='SpokenCOCO',
        description=(
            'Spoken COCO: audio-image retrieval with spoken captions on'
            ' MS COCO images.'
        ),
        homepage='https://groups.csail.mit.edu/sls/downloads/placesaudio/',
        version='1.0.0',
        license='Creative Commons Attribution-ShareAlike (CC BY-SA)',
        mseb_tasks=['retrieval'],
    )

  def _load_data(self) -> list[Mapping[str, Any]]:
    """Loads and caches the SpokenCOCO JSON metadata."""
    if self._data is None:
      json_path = os.path.join(self.base_path, f'SpokenCOCO_{self.split}.json')
      logger.info('Loading SpokenCOCO metadata from %s', json_path)
      with epath.Path(json_path).open('r') as f:
        self._data = json.load(f)['data']
      logger.info('Loaded %d images from SpokenCOCO', len(self._data))
    return self._data

  def __len__(self) -> int:
    """Returns the number of images in the dataset."""
    return len(self._load_data())

  def num_captions(self) -> int:
    """Returns the total number of spoken captions across all images."""
    return sum(len(item['captions']) for item in self._load_data())

  def get_task_data(
      self,
      task_name: str | None = None,
      dtype: Mapping[str, Any] | None = None,
  ) -> pd.DataFrame:
    """Returns a DataFrame with one row per (caption, image) pair.

    Args:
      task_name: The task name (unused).
      dtype: Optional dictionary of column names to data types.

    Returns:
      A pandas DataFrame with the columns uttid, image, wav, text, speaker.
    """
    del task_name  # Unused.
    data = self._load_data()
    records = []
    for item in data:
      image_path = item['image']
      for caption in item['captions']:
        records.append({
            'uttid': caption['uttid'],
            'image': image_path,
            'wav': caption['wav'],
            'text': caption['text'],
            'speaker': caption['speaker'],
        })
    df = pd.DataFrame(records)
    if dtype:
      df = df.astype(dtype)
    return df
