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

"""Shared utilities for audio-image retrieval datasets.

Provides the AudioImageDataset base class with common get_sound() and
get_image() implementations used by SpokenCOCO, Flickr8k, and similar datasets
where audio captions are paired with images.
"""

import io
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
from PIL import Image as PILImage

logger = logging.getLogger(__name__)


class AudioImageDataset(base.MsebDataset):
  """Base class for audio-image retrieval datasets.

  Subclasses must implement:
    - _load_data(): Load and cache the dataset metadata.
    - metadata: Return the DatasetMetadata.
    - get_task_data(): Return a DataFrame with columns:
        uttid, image, wav, text, speaker.
    - get_unique_images(): Return a list of unique image records.

  This base class provides shared implementations of:
    - get_sound(): Load a WAV file as a types.Sound.
    - get_image(): Load an image file as a types.Image.
    - __len__(): Number of images (based on _load_data).
    - num_captions(): Total number of spoken captions.
  """

  def get_sound(self, record: Mapping[str, Any]) -> types.Sound:
    """Loads a Sound object from a dataset record.

    Args:
      record: A dict with keys 'uttid', 'wav', and optionally 'text', 'speaker'.

    Returns:
      A types.Sound object with the loaded waveform.
    """
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
    """Loads an Image object from a dataset record.

    Args:
      record: A dict with key 'image' containing the relative path to the image
        file.

    Returns:
      A types.Image object with the loaded pixel data.
    """
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
    """Returns a list of unique image records (one per image).

    Returns:
      A list of dicts with key 'image' containing the relative image path.
    """
    df = self.get_task_data()
    unique_images = df['image'].unique()
    return [{'image': img} for img in unique_images]


def download_file(url: str, dest_path: str) -> None:
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


def extract_tar_gz(tar_path: str, dest_dir: str) -> None:
  """Extracts a .tar.gz file to dest_dir."""
  logger.info('Extracting %s -> %s', tar_path, dest_dir)
  with epath.Path(tar_path).open('rb') as tar_file:
    with tarfile.open(fileobj=tar_file, mode='r:gz') as tar:
      local_dir = tempfile.mkdtemp()
      tar.extractall(local_dir)
      _copy_tree(local_dir, dest_dir)
  logger.info('Extraction complete.')


def extract_zip(zip_path: str, dest_dir: str) -> None:
  """Extracts a .zip file to dest_dir."""
  logger.info('Extracting %s -> %s', zip_path, dest_dir)
  with epath.Path(zip_path).open('rb') as zip_file:
    with zipfile.ZipFile(zip_file, 'r') as zf:
      local_dir = tempfile.mkdtemp()
      zf.extractall(local_dir)
      _copy_tree(local_dir, dest_dir)
  logger.info('Extraction complete.')
