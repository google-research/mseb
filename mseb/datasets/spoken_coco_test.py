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

"""Tests for the SpokenCOCO dataset."""

import json
import os
import tarfile
from unittest import mock
import zipfile

from absl.testing import absltest
from mseb.datasets import spoken_coco
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from scipy.io import wavfile


def _create_mock_dataset(base_dir: str) -> None:
  """Creates a minimal SpokenCOCO file structure in base_dir."""
  # Create image directory.
  image_dir = os.path.join(base_dir, 'val2014')
  os.makedirs(image_dir)

  # Create two small RGB images.
  for filename in (
      'COCO_val2014_000000000001.jpg',
      'COCO_val2014_000000000002.jpg',
  ):
    img = PILImage.fromarray(
        np.random.randint(0, 255, (32, 48, 3), dtype=np.uint8)
    )
    img.save(os.path.join(image_dir, filename))

  # Create WAV directory.
  wav_dir = os.path.join(base_dir, 'wavs', 'val', '0')
  os.makedirs(wav_dir)

  # Create small WAV files (16kHz, 0.5s).
  sample_rate = 16000
  duration_samples = sample_rate // 2
  for filename in (
      'speaker1-utt1_1_100.wav',
      'speaker2-utt2_1_200.wav',
      'speaker3-utt3_2_300.wav',
  ):
    waveform = np.random.randn(duration_samples).astype(np.float32) * 0.1
    wavfile.write(os.path.join(wav_dir, filename), sample_rate, waveform)

  # Create the metadata JSON.
  metadata = {
      'data': [
          {
              'image': 'val2014/COCO_val2014_000000000001.jpg',
              'captions': [
                  {
                      'text': 'A DOG SITTING ON A COUCH',
                      'speaker': 'speaker1',
                      'uttid': 'speaker1-utt1_1_100',
                      'wav': 'wavs/val/0/speaker1-utt1_1_100.wav',
                  },
                  {
                      'text': 'A BROWN DOG ON A RED SOFA',
                      'speaker': 'speaker2',
                      'uttid': 'speaker2-utt2_1_200',
                      'wav': 'wavs/val/0/speaker2-utt2_1_200.wav',
                  },
              ],
          },
          {
              'image': 'val2014/COCO_val2014_000000000002.jpg',
              'captions': [
                  {
                      'text': 'A CAT SLEEPING ON A BED',
                      'speaker': 'speaker3',
                      'uttid': 'speaker3-utt3_2_300',
                      'wav': 'wavs/val/0/speaker3-utt3_2_300.wav',
                  },
              ],
          },
      ]
  }
  with open(os.path.join(base_dir, 'SpokenCOCO_val.json'), 'w') as f:
    json.dump(metadata, f)


class SpokenCocoDatasetTest(absltest.TestCase):
  """Tests for the SpokenCocoDataset class."""

  def setUp(self):
    super().setUp()
    self.testdata_dir = self.create_tempdir().full_path
    _create_mock_dataset(self.testdata_dir)
    self.dataset = spoken_coco.SpokenCocoDataset(
        base_path=self.testdata_dir, split='val'
    )

  def test_metadata(self):
    meta = self.dataset.metadata
    self.assertEqual(meta.name, 'SpokenCOCO')
    self.assertIn('retrieval', meta.mseb_tasks)

  def test_len(self):
    # 2 images in the mock dataset.
    self.assertLen(self.dataset, 2)

  def test_num_captions(self):
    # Image 1 has 2 captions, image 2 has 1 caption.
    self.assertEqual(self.dataset.num_captions(), 3)

  def test_get_task_data_returns_dataframe(self):
    df = self.dataset.get_task_data()
    self.assertIsInstance(df, pd.DataFrame)
    self.assertLen(df, 3)

  def test_get_task_data_columns(self):
    df = self.dataset.get_task_data()
    expected_columns = {'uttid', 'image', 'wav', 'text', 'speaker'}
    self.assertEqual(set(df.columns), expected_columns)

  def test_get_task_data_values(self):
    df = self.dataset.get_task_data()
    first_row = df.iloc[0]
    self.assertEqual(first_row['uttid'], 'speaker1-utt1_1_100')
    self.assertEqual(
        first_row['image'], 'val2014/COCO_val2014_000000000001.jpg'
    )
    self.assertEqual(first_row['text'], 'A DOG SITTING ON A COUCH')
    self.assertEqual(first_row['speaker'], 'speaker1')

  def test_get_task_data_with_dtype(self):
    df = self.dataset.get_task_data(dtype={'uttid': str, 'speaker': str})
    self.assertEqual(df['uttid'].dtype, pd.StringDtype)

  def test_data_is_cached(self):
    """Verify that _load_data caches the parsed JSON."""
    _ = self.dataset.get_task_data()
    data1 = self.dataset._data  # pylint: disable=protected-access
    _ = self.dataset.get_task_data()
    data2 = self.dataset._data  # pylint: disable=protected-access
    self.assertIs(data1, data2)

  def test_base_path_with_explicit_path(self):
    """When base_path is provided, it should be used directly."""
    ds = spoken_coco.SpokenCocoDataset(base_path=self.testdata_dir, split='val')
    self.assertEqual(ds.base_path, self.testdata_dir)


class DownloadSpokenCocoTest(absltest.TestCase):
  """Tests for the download_spoken_coco function."""

  def _create_mock_tar(self, tar_path: str, base_dir: str) -> None:
    """Creates a small tar.gz mimicking the SpokenCOCO archive."""
    # Create files to put in the tar.
    json_data = {'data': [{'image': 'val2014/img.jpg', 'captions': []}]}
    os.makedirs(os.path.join(base_dir, 'SpokenCOCO', 'wavs', 'val', '0'))
    json_path = os.path.join(base_dir, 'SpokenCOCO', 'SpokenCOCO_val.json')
    with open(json_path, 'w') as f:
      json.dump(json_data, f)

    with tarfile.open(tar_path, 'w:gz') as tar:
      tar.add(
          os.path.join(base_dir, 'SpokenCOCO'),
          arcname='SpokenCOCO',
      )

  def _create_mock_zip(self, zip_path: str, base_dir: str) -> None:
    """Creates a small zip mimicking the MSCOCO val2014 archive."""
    img = PILImage.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    img_path = os.path.join(base_dir, 'val2014', 'img.jpg')
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    img.save(img_path)

    with zipfile.ZipFile(zip_path, 'w') as zf:
      zf.write(img_path, 'val2014/img.jpg')

  @mock.patch.object(spoken_coco.audio_image_base, 'download_file')
  def test_download_creates_expected_structure(self, mock_dl):
    """Test that download_spoken_coco produces the expected directory layout."""
    dest_dir = self.create_tempdir().full_path
    staging = self.create_tempdir().full_path

    # Create mock archives.
    tar_path = os.path.join(dest_dir, 'SpokenCOCO.tar.gz')
    zip_path = os.path.join(dest_dir, 'val2014.zip')
    self._create_mock_tar(tar_path, staging)
    self._create_mock_zip(zip_path, staging)

    # Mock _download_file to be a no-op (archives already placed above).
    mock_dl.side_effect = lambda url, path: None

    spoken_coco.maybe_download_spoken_coco(dest_dir=dest_dir, split='val')

    self.assertTrue(
        os.path.exists(
            os.path.join(dest_dir, 'SpokenCOCO', 'SpokenCOCO_val.json')
        )
    )
    self.assertTrue(os.path.isdir(os.path.join(dest_dir, 'val2014')))

  def test_download_skips_if_data_exists(self):
    """Test that existing data is not re-downloaded."""
    dest_dir = self.create_tempdir().full_path

    # Pre-populate the expected files.
    json_path = os.path.join(dest_dir, 'SpokenCOCO_val.json')
    with open(json_path, 'w') as f:
      json.dump({'data': []}, f)
    os.makedirs(os.path.join(dest_dir, 'wavs'))
    os.makedirs(os.path.join(dest_dir, 'val2014'))

    with mock.patch.object(
        spoken_coco.audio_image_base, 'download_file'
    ) as mock_dl:
      spoken_coco.maybe_download_spoken_coco(dest_dir=dest_dir, split='val')
      mock_dl.assert_not_called()


if __name__ == '__main__':
  absltest.main()
