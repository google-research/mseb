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

"""Tests for the Flickr8k dataset."""

import csv
import os
from unittest import mock

from absl.testing import absltest
from mseb.datasets import flickr8k
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from scipy.io import wavfile


def _create_mock_dataset(base_dir: str) -> None:
  """Creates a minimal Flickr8k file structure in base_dir."""
  # Create image directory with two images.
  image_dir = os.path.join(base_dir, 'Images')
  os.makedirs(image_dir)
  for filename in ('img_001.jpg', 'img_002.jpg'):
    img = PILImage.fromarray(
        np.random.randint(0, 255, (32, 48, 3), dtype=np.uint8)
    )
    img.save(os.path.join(image_dir, filename))

  # Create WAV directory with 3 WAV files.
  wav_dir = os.path.join(base_dir, 'flickr_audio', 'wavs')
  os.makedirs(wav_dir)
  sample_rate = 16000
  duration_samples = sample_rate // 2
  for filename in ('img_001_0.wav', 'img_001_1.wav', 'img_002_0.wav'):
    waveform = np.random.randn(duration_samples).astype(np.float32) * 0.1
    wavfile.write(os.path.join(wav_dir, filename), sample_rate, waveform)

  # Create wav2capt.txt
  wav2capt_path = os.path.join(base_dir, 'flickr_audio', 'wav2capt.txt')
  with open(wav2capt_path, 'w') as f:
    f.write('img_001_0.wav img_001.jpg #0\n')
    f.write('img_001_1.wav img_001.jpg #1\n')
    f.write('img_002_0.wav img_002.jpg #0\n')

  # Create wav2spk.txt
  wav2spk_path = os.path.join(base_dir, 'flickr_audio', 'wav2spk.txt')
  with open(wav2spk_path, 'w') as f:
    f.write('img_001_0.wav speaker_A\n')
    f.write('img_001_1.wav speaker_B\n')
    f.write('img_002_0.wav speaker_A\n')

  # Create captions.txt (CSV format with header)
  captions_path = os.path.join(base_dir, 'captions.txt')
  with open(captions_path, 'w') as f:
    f.write('image,caption\n')
    f.write('img_001.jpg,A dog sitting on a couch\n')
    f.write('img_001.jpg,A brown dog on a red sofa\n')
    f.write('img_002.jpg,A cat sleeping on a bed\n')

  # Create split file — both images in test split.
  split_path = os.path.join(base_dir, 'Flickr_8k.testImages.txt')
  with open(split_path, 'w') as f:
    f.write('img_001.jpg\n')
    f.write('img_002.jpg\n')


class Flickr8kDatasetTest(absltest.TestCase):
  """Tests for the Flickr8kDataset class."""

  def setUp(self):
    super().setUp()
    self.testdata_dir = self.create_tempdir().full_path
    _create_mock_dataset(self.testdata_dir)
    self.dataset = flickr8k.Flickr8kDataset(
        base_path=self.testdata_dir, split='test'
    )

  def test_metadata(self):
    meta = self.dataset.metadata
    self.assertEqual(meta.name, 'Flickr8kAudio')
    self.assertIn('retrieval', meta.mseb_tasks)

  def test_len(self):
    # 2 unique images in the test split.
    self.assertLen(self.dataset, 2)

  def test_num_captions(self):
    # img_001 has 2 captions, img_002 has 1 caption.
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
    # Sorted by uttid, so img_001_0 comes first.
    first_row = df.iloc[0]
    self.assertEqual(first_row['uttid'], 'img_001_0')
    self.assertEqual(first_row['image'], 'Images/img_001.jpg')
    self.assertEqual(first_row['wav'], 'flickr_audio/wavs/img_001_0.wav')
    self.assertEqual(first_row['text'], 'A dog sitting on a couch')
    self.assertEqual(first_row['speaker'], 'speaker_A')

  def test_get_task_data_caption_indices(self):
    """Verify caption index mapping: caption #1 maps to second caption."""
    df = self.dataset.get_task_data()
    row_1 = df[df['uttid'] == 'img_001_1'].iloc[0]
    self.assertEqual(row_1['text'], 'A brown dog on a red sofa')

  def test_data_is_cached(self):
    """Verify that get_task_data caches the DataFrame."""
    _ = self.dataset.get_task_data()
    data1 = self.dataset._task_data  # pylint: disable=protected-access
    _ = self.dataset.get_task_data()
    data2 = self.dataset._task_data  # pylint: disable=protected-access
    self.assertIs(data1, data2)

  def test_split_filtering(self):
    """Only images in the split file should appear in task data."""
    # Create a dev split with only img_001.
    dev_path = os.path.join(self.testdata_dir, 'Flickr_8k.devImages.txt')
    with open(dev_path, 'w') as f:
      f.write('img_001.jpg\n')

    dev_dataset = flickr8k.Flickr8kDataset(
        base_path=self.testdata_dir, split='dev'
    )
    df = dev_dataset.get_task_data()
    self.assertLen(df, 2)  # Only 2 captions for img_001.
    self.assertTrue((df['image'] == 'Images/img_001.jpg').all())

  def test_unknown_split_raises(self):
    with self.assertRaises(ValueError):
      flickr8k.Flickr8kDataset(
          base_path=self.testdata_dir, split='unknown'
      ).get_task_data()


class DownloadFlickr8kTest(absltest.TestCase):
  """Tests for the maybe_download_flickr8k function."""

  def test_download_skips_if_data_exists(self):
    """Test that existing data is not re-downloaded."""
    dest_dir = self.create_tempdir().full_path

    # Pre-populate the expected files/dirs.
    os.makedirs(os.path.join(dest_dir, 'flickr_audio'))
    os.makedirs(os.path.join(dest_dir, 'Images'))
    with open(os.path.join(dest_dir, 'captions.txt'), 'w') as f:
      f.write('image,caption\n')

    with mock.patch.object(
        flickr8k.audio_image_base, 'download_file'
    ) as mock_dl:
      flickr8k.maybe_download_flickr8k(dest_dir=dest_dir)
      mock_dl.assert_not_called()

  def test_convert_token_to_captions_csv(self):
    """Test the Flickr8k.token.txt -> captions.txt conversion."""
    dest_dir = self.create_tempdir().full_path
    token_path = os.path.join(dest_dir, 'Flickr8k.token.txt')
    captions_path = os.path.join(dest_dir, 'captions.txt')

    with open(token_path, 'w') as f:
      f.write('img_001.jpg#0\tA dog on a couch\n')
      f.write('img_001.jpg#1\tA brown dog, sitting\n')
      f.write('img_002.jpg#0\tA cat on a bed\n')

    flickr8k._convert_token_to_captions_csv(token_path, captions_path)

    with open(captions_path, 'r') as f:
      reader = csv.DictReader(f)
      rows = list(reader)

    self.assertLen(rows, 3)
    self.assertEqual(rows[0]['image'], 'img_001.jpg')
    self.assertEqual(rows[0]['caption'], 'A dog on a couch')
    # Verify comma in caption is handled.
    self.assertEqual(rows[1]['caption'], 'A brown dog, sitting')


if __name__ == '__main__':
  absltest.main()
