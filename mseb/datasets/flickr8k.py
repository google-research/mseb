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

"""Flickr8k Audio dataset.

Flickr8k Audio is an audio-image retrieval benchmark where each image from
Flickr8k has five spoken captions. The task is to retrieve the correct image
given a spoken description.

When no base_path is provided, the dataset is automatically downloaded from:
  - Flickr8k audio:
  https://groups.csail.mit.edu/sls/downloads/flickraudio/downloads/flickr_audio.tar.gz
  - Flickr8k images:
  https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
  - Flickr8k text (captions + splits):
  https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip

Reference: https://groups.csail.mit.edu/sls/downloads/flickraudio/
"""

import csv
import logging
import os
from typing import Any, Mapping

from etils import epath
from mseb import dataset
from mseb import types
from mseb.datasets import audio_image_base
import pandas as pd

logger = logging.getLogger(__name__)

_FLICKR8K_AUDIO_URL = (
    'https://groups.csail.mit.edu/sls/downloads/flickraudio/'
    'downloads/flickr_audio.tar.gz'
)
_FLICKR8K_IMAGES_URL = (
    'https://github.com/jbrownlee/Datasets/releases/download/'
    'Flickr8k/Flickr8k_Dataset.zip'
)
_FLICKR8K_TEXT_URL = (
    'https://github.com/jbrownlee/Datasets/releases/download/'
    'Flickr8k/Flickr8k_text.zip'
)

_SPLIT_TO_FILENAME = {
    'test': 'Flickr_8k.testImages.txt',
    'dev': 'Flickr_8k.devImages.txt',
    'train': 'Flickr_8k.trainImages.txt',
}


def maybe_download_flickr8k(dest_dir: str) -> None:
  """Downloads Flickr8k audio, images, and text to dest_dir.

  Downloads are skipped if the expected files already exist in dest_dir.

  Args:
    dest_dir: Directory to download to.
  """
  audio_dir = os.path.join(dest_dir, 'flickr_audio')
  images_dir = os.path.join(dest_dir, 'Images')
  captions_path = os.path.join(dest_dir, 'captions.txt')

  # Download and extract Flickr8k audio (wav files + metadata).
  if not epath.Path(audio_dir).is_dir():
    epath.Path(dest_dir).mkdir(parents=True, exist_ok=True)
    tar_path = os.path.join(dest_dir, 'flickr_audio.tar.gz')
    if not epath.Path(tar_path).exists():
      audio_image_base.download_file(_FLICKR8K_AUDIO_URL, tar_path)
    audio_image_base.extract_tar_gz(tar_path, dest_dir)
    logger.info('Flickr8k audio data ready at %s', audio_dir)
  else:
    logger.info('Flickr8k audio data already exists at %s', audio_dir)

  # Download and extract Flickr8k images.
  if not epath.Path(images_dir).is_dir():
    zip_path = os.path.join(dest_dir, 'Flickr8k_Dataset.zip')
    if not epath.Path(zip_path).exists():
      audio_image_base.download_file(_FLICKR8K_IMAGES_URL, zip_path)
    audio_image_base.extract_zip(zip_path, dest_dir)
    # The zip extracts to Flicker8k_Dataset/ (note the typo in the original).
    # Rename to Images/ to match our expected layout.
    for candidate in ('Flicker8k_Dataset', 'Flickr8k_Dataset'):
      extracted = os.path.join(dest_dir, candidate)
      if epath.Path(extracted).is_dir():
        epath.Path(extracted).rename(images_dir)
        break
    logger.info('Flickr8k images ready at %s', images_dir)
  else:
    logger.info('Flickr8k images already exist at %s', images_dir)

  # Download and extract Flickr8k text (captions + split files).
  if not epath.Path(captions_path).exists():
    zip_path = os.path.join(dest_dir, 'Flickr8k_text.zip')
    if not epath.Path(zip_path).exists():
      audio_image_base.download_file(_FLICKR8K_TEXT_URL, zip_path)
    audio_image_base.extract_zip(zip_path, dest_dir)
    # Convert Flickr8k.token.txt to captions.txt CSV format.
    token_path = os.path.join(dest_dir, 'Flickr8k.token.txt')
    if (
        epath.Path(token_path).exists()
        and not epath.Path(captions_path).exists()
    ):
      _convert_token_to_captions_csv(token_path, captions_path)
    logger.info('Flickr8k text data ready at %s', dest_dir)
  else:
    logger.info('Flickr8k text data already exists at %s', dest_dir)


def _convert_token_to_captions_csv(token_path: str, captions_path: str) -> None:
  """Converts Flickr8k.token.txt to captions.txt CSV format."""
  with epath.Path(token_path).open('r') as fin, epath.Path(captions_path).open(
      'w'
  ) as fout:
    fout.write('image,caption\n')
    for line in fin:
      line = line.strip()
      if not line:
        continue
      # Format: "image_file#idx\tcaption"
      parts = line.split('\t', 1)
      if len(parts) != 2:
        continue
      image_caption_id, caption = parts
      image_file = image_caption_id.rsplit('#', 1)[0]
      # Escape commas in caption for CSV.
      caption = caption.replace('"', '""')
      fout.write(f'{image_file},"{caption}"\n')


class Flickr8kDataset(audio_image_base.AudioImageDataset):
  """Flickr8k Audio dataset for audio-image retrieval.

  The dataset structure is:
    - Images/         : JPEG images (e.g., 1000268201_693b08cb0e.jpg)
    - flickr_audio/wavs/  : WAV files (e.g., 1000268201_693b08cb0e_0.wav)
    - flickr_audio/wav2capt.txt : wav -> image + caption index mapping
    - flickr_audio/wav2spk.txt  : wav -> speaker_id mapping
    - captions.txt    : CSV with columns (image, caption)
    - Flickr_8k.{test,dev,train}Images.txt : split image lists

  If base_path is not provided, the dataset is downloaded automatically to
  a temporary directory. Set the TMPDIR environment variable to control
  where the temp directory is created, e.g.:
    TMPDIR=/my/data/dir python my_script.py
  """

  def __init__(
      self,
      base_path: str | None = None,
      split: str = 'test',
  ):
    """Initializes the Flickr8k dataset.

    Args:
      base_path: Path to the dataset root. If provided, data is loaded directly
        from this path (no download). If None, a temporary directory is created
        and data is downloaded automatically. Set the TMPDIR environment
        variable to control where the temp directory is created.
      split: Dataset split ('test', 'dev', or 'train').
    """
    super().__init__(base_path=base_path, split=split)
    self.base_path = dataset.get_base_path(self.base_path)
    maybe_download_flickr8k(dest_dir=self.base_path)
    self._task_data: pd.DataFrame | None = None

  @property
  def metadata(self) -> types.DatasetMetadata:
    return types.DatasetMetadata(
        name='Flickr8kAudio',
        description=(
            'Flickr8k Audio: audio-image retrieval with spoken captions on'
            ' Flickr8k images.'
        ),
        homepage='https://groups.csail.mit.edu/sls/downloads/flickraudio/',
        version='1.0.0',
        license='Creative Commons Attribution-ShareAlike (CC BY-SA)',
        mseb_tasks=['retrieval'],
    )

  def _load_split_images(self) -> set[str]:
    """Loads the set of image filenames for the current split."""
    split_file = _SPLIT_TO_FILENAME.get(self.split)
    if split_file is None:
      raise ValueError(
          f"Unknown split '{self.split}'. "
          f'Must be one of {list(_SPLIT_TO_FILENAME.keys())}.'
      )
    split_path = os.path.join(self.base_path, split_file)  # pyrefly: ignore[no-matching-overload]
    with epath.Path(split_path).open('r') as f:
      return {line.strip() for line in f if line.strip()}

  def _load_captions(self) -> dict[str, list[str]]:
    """Loads captions.txt and returns {image_filename: [caption1, ...]}."""
    captions_path = os.path.join(self.base_path, 'captions.txt')  # pyrefly: ignore[no-matching-overload]
    captions: dict[str, list[str]] = {}
    with epath.Path(captions_path).open('r') as f:
      reader = csv.DictReader(f)
      for row in reader:
        img = row['image']
        captions.setdefault(img, []).append(row['caption'])
    return captions

  def _load_wav2capt(self) -> dict[str, tuple[str, str]]:
    """Loads wav2capt.txt and returns {wav_file: (image_file, caption_idx)}."""
    wav2capt_path = os.path.join(self.base_path, 'flickr_audio', 'wav2capt.txt')  # pyrefly: ignore[no-matching-overload]
    mapping: dict[str, tuple[str, str]] = {}
    with epath.Path(wav2capt_path).open('r') as f:
      for line in f:
        parts = line.strip().split()
        if len(parts) >= 3:
          wav_file, image_file, caption_idx = parts[0], parts[1], parts[2]
          mapping[wav_file] = (image_file, caption_idx)
    return mapping

  def _load_wav2spk(self) -> dict[str, str]:
    """Loads wav2spk.txt and returns {wav_file: speaker_id}."""
    wav2spk_path = os.path.join(self.base_path, 'flickr_audio', 'wav2spk.txt')  # pyrefly: ignore[no-matching-overload]
    mapping: dict[str, str] = {}
    with epath.Path(wav2spk_path).open('r') as f:
      for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
          mapping[parts[0]] = parts[1]
    return mapping

  def _build_task_data(self) -> pd.DataFrame:
    """Builds the task DataFrame by joining wav2capt, wav2spk, and captions."""
    split_images = self._load_split_images()
    wav2capt = self._load_wav2capt()
    wav2spk = self._load_wav2spk()
    captions = self._load_captions()

    logger.info(
        'Building Flickr8k task data for split=%s (%d images)',
        self.split,
        len(split_images),
    )

    records = []
    for wav_file, (image_file, caption_idx_str) in wav2capt.items():
      if image_file not in split_images:
        continue

      caption_idx = int(caption_idx_str.lstrip('#'))
      image_captions = captions.get(image_file, [])
      text = (
          image_captions[caption_idx]
          if caption_idx < len(image_captions)
          else ''
      )

      uttid = os.path.splitext(wav_file)[0]  # e.g., 1000268201_693b08cb0e_0
      speaker = wav2spk.get(wav_file, '')

      records.append({
          'uttid': uttid,
          'image': os.path.join('Images', image_file),
          'wav': os.path.join('flickr_audio', 'wavs', wav_file),
          'text': text,
          'speaker': speaker,
      })

    df = pd.DataFrame(records)
    # Sort for deterministic ordering.
    df = df.sort_values('uttid').reset_index(drop=True)
    logger.info('Built %d caption records for Flickr8k', len(df))
    return df

  def __len__(self) -> int:
    """Returns the number of unique images in the split."""
    df = self.get_task_data()
    return df['image'].nunique()

  def num_captions(self) -> int:
    """Returns the total number of spoken captions in the split."""
    return len(self.get_task_data())

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
    if self._task_data is None:
      self._task_data = self._build_task_data()
    df = self._task_data
    if dtype:
      df = df.astype(dtype)
    return df
