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

import io
import json
import logging
import os
import tarfile
import tempfile
from typing import Any, Mapping, Sequence
import urllib.request
import zipfile

from etils import epath
from mseb import types
from mseb import utils
from mseb.datasets import base
import numpy as np
import pandas as pd
from PIL import Image as PILImage

logger = logging.getLogger(__name__)


_SPOKEN_COCO_URL = 'https://data.csail.mit.edu/placesaudio/SpokenCOCO.tar.gz'
_MSCOCO_VAL2014_URL = 'http://images.cocodataset.org/zips/val2014.zip'


def _download_file(url: str, dest_path: str) -> None:
  """Downloads a file from url to dest_path with progress logging."""
  logger.info('Downloading %s -> %s', url, dest_path)
  response = urllib.request.urlopen(url)  # pylint: disable=missing-timeout
  total_size = response.headers.get('Content-Length')
  total_size = int(total_size) if total_size else None

  block_size = 8 * 1024 * 1024  # 8 MB
  downloaded = 0
  log_interval = 500 * 1024 * 1024  # Log every 500 MB
  next_log_at = log_interval

  with epath.Path(dest_path).open('wb') as f:
    while True:
      chunk = response.read(block_size)
      if not chunk:
        break
      f.write(chunk)
      downloaded += len(chunk)
      if downloaded >= next_log_at:
        if total_size:
          pct = 100.0 * downloaded / total_size
          logger.info(
              '  Downloaded %.1f GB / %.1f GB (%.1f%%)',
              downloaded / 1e9,
              total_size / 1e9,
              pct,
          )
        else:
          logger.info('  Downloaded %.1f GB', downloaded / 1e9)
        next_log_at += log_interval

  logger.info('Download complete: %s (%.1f GB)', dest_path, downloaded / 1e9)


def _copy_tree(src_dir: str | epath.Path, dst_dir: str | epath.Path) -> None:
  """Copies the contents of src_dir to dst_dir."""
  src_dir = epath.Path(src_dir)
  dst_dir = epath.Path(dst_dir)
  dst_dir.mkdir(parents=True, exist_ok=True)
  for src_path in src_dir.iterdir():
    dst_path = epath.Path(os.path.join(dst_dir, src_path.name))
    if src_path.is_dir():
      _copy_tree(src_path, dst_path)
    else:
      src_path.copy(dst_path, overwrite=True)
      src_path.unlink()


def _extract_tar_gz(tar_path: str, dest_dir: str) -> None:
  """Extracts a .tar.gz file to dest_dir."""
  logger.info('Extracting %s -> %s', tar_path, dest_dir)
  with epath.Path(tar_path).open('rb') as tar_file:
    with tarfile.open(fileobj=tar_file, mode='r:gz') as tar:
      local_dir = tempfile.mkdtemp()
      tar.extractall(local_dir)
      _copy_tree(local_dir, dest_dir)
  logger.info('Extraction complete.')


def _extract_zip(zip_path: str, dest_dir: str) -> None:
  """Extracts a .zip file to dest_dir."""
  logger.info('Extracting %s -> %s', zip_path, dest_dir)
  with epath.Path(zip_path).open('rb') as zip_file:
    with zipfile.ZipFile(zip_file, 'r') as zf:
      local_dir = tempfile.mkdtemp()
      zf.extractall(local_dir)
      _copy_tree(local_dir, dest_dir)
  logger.info('Extraction complete.')


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
      _download_file(_SPOKEN_COCO_URL, tar_path)
    _extract_tar_gz(tar_path, dest_dir)
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
      _download_file(_MSCOCO_VAL2014_URL, zip_path)
    _extract_zip(zip_path, dest_dir)
    logger.info('MSCOCO val2014 images ready at %s', images_dir)
  else:
    logger.info('MSCOCO val2014 images already exist at %s', images_dir)


class SpokenCocoDataset(base.MsebDataset):
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

  def get_sound(self, record: Mapping[str, Any]) -> types.Sound:
    """Loads a Sound object from a dataset record."""
    wav_path = os.path.join(self.base_path, record['wav'])
    waveform, sample_rate = utils.read_audio(wav_path)
    context = types.SoundContextParams(
        id=record['uttid'],
        sample_rate=sample_rate,
        length=len(waveform),
        language='en',
        speaker_id=record.get('speaker'),
        text=record.get('text'),
    )
    return types.Sound(waveform=waveform, context=context)

  def get_image(self, record: Mapping[str, Any]) -> types.Image:
    """Loads an Image object from a dataset record."""
    image_path = os.path.join(self.base_path, record['image'])
    with epath.Path(image_path).open('rb') as f:
      pil_image = PILImage.open(io.BytesIO(f.read()))
      pil_image = pil_image.convert('RGB')
    image_array = np.array(pil_image, dtype=np.uint8)
    context = types.ImageContextParams(
        id=record['image'],
        height=image_array.shape[0],
        width=image_array.shape[1],
        channels=image_array.shape[2],
    )
    return types.Image(image=image_array, context=context)

  def get_unique_images(self) -> Sequence[Mapping[str, str]]:
    """Returns a list of unique image records (one per image)."""
    data = self._load_data()
    unique_images = {item['image']: {'image': item['image']} for item in data}
    return list(unique_images.values())
