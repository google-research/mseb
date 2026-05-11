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

"""Tests for audio_image_base.py shared utilities and AudioImageDataset."""

import os
import tarfile
from typing import Any, Mapping
import zipfile

from absl.testing import absltest
from mseb import types
from mseb.datasets import audio_image_base
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from scipy.io import wavfile


def _create_mock_audio_image_data(base_dir: str) -> pd.DataFrame:
  """Creates mock WAV and image files, returns a task DataFrame.

  Creates:
    - base_dir/images/img_001.jpg (32x48 RGB)
    - base_dir/images/img_002.jpg (24x36 RGB)
    - base_dir/wavs/utt_001.wav  (16kHz, 0.5s)
    - base_dir/wavs/utt_002.wav  (16kHz, 0.5s)
    - base_dir/wavs/utt_003.wav  (16kHz, 0.5s)

  Args:
    base_dir: Directory to create the mock data in.

  Returns:
    DataFrame with columns: uttid, image, wav, text, speaker
  """
  # Images.
  image_dir = os.path.join(base_dir, 'images')
  os.makedirs(image_dir)
  PILImage.fromarray(
      np.random.randint(0, 255, (32, 48, 3), dtype=np.uint8)
  ).save(os.path.join(image_dir, 'img_001.jpg'))
  PILImage.fromarray(
      np.random.randint(0, 255, (24, 36, 3), dtype=np.uint8)
  ).save(os.path.join(image_dir, 'img_002.jpg'))

  # WAVs.
  wav_dir = os.path.join(base_dir, 'wavs')
  os.makedirs(wav_dir)
  sample_rate = 16000
  for name in ('utt_001.wav', 'utt_002.wav', 'utt_003.wav'):
    waveform = np.random.randn(sample_rate // 2).astype(np.float32) * 0.1
    wavfile.write(os.path.join(wav_dir, name), sample_rate, waveform)

  return pd.DataFrame([
      {
          'uttid': 'utt_001',
          'image': 'images/img_001.jpg',
          'wav': 'wavs/utt_001.wav',
          'text': 'A dog on a couch',
          'speaker': 'spk_A',
      },
      {
          'uttid': 'utt_002',
          'image': 'images/img_001.jpg',
          'wav': 'wavs/utt_002.wav',
          'text': 'A brown dog on a sofa',
          'speaker': 'spk_B',
      },
      {
          'uttid': 'utt_003',
          'image': 'images/img_002.jpg',
          'wav': 'wavs/utt_003.wav',
          'text': 'A cat sleeping on a bed',
          'speaker': 'spk_A',
      },
  ])


class _StubDataset(audio_image_base.AudioImageDataset):
  """Minimal concrete subclass for testing base class methods."""

  def __init__(self, base_path: str, task_data: pd.DataFrame):
    super().__init__(base_path=base_path, split='test')
    self._task_data = task_data

  @property
  def metadata(self) -> types.DatasetMetadata:
    return types.DatasetMetadata(
        name='StubDataset',
        description='Stub for testing.',
        homepage='',
        version='0.0.0',
        license='test',
        mseb_tasks=['retrieval'],
    )

  def __len__(self) -> int:
    return self._task_data['image'].nunique()

  def get_task_data(
      self,
      task_name: str | None = None,
      dtype: Mapping[str, Any] | None = None,
  ) -> pd.DataFrame:
    del task_name
    df = self._task_data
    if dtype:
      df = df.astype(dtype)
    return df


class AudioImageDatasetGetSoundTest(absltest.TestCase):
  """Tests for AudioImageDataset.get_sound()."""

  def setUp(self):
    super().setUp()
    self.base_dir = self.create_tempdir().full_path
    self.task_data = _create_mock_audio_image_data(self.base_dir)
    self.dataset = _StubDataset(self.base_dir, self.task_data)

  def test_returns_sound_type(self):
    record = self.task_data.iloc[0].to_dict()
    sound = self.dataset.get_sound(record)
    self.assertIsInstance(sound, types.Sound)

  def test_sound_context_fields(self):
    record = self.task_data.iloc[0].to_dict()
    sound = self.dataset.get_sound(record)
    self.assertEqual(sound.context.id, 'utt_001')
    self.assertEqual(sound.context.sample_rate, 16000)
    self.assertEqual(sound.context.language, 'en')
    self.assertEqual(sound.context.speaker_id, 'spk_A')
    self.assertEqual(sound.context.text, 'A dog on a couch')

  def test_sound_waveform_shape_and_dtype(self):
    record = self.task_data.iloc[0].to_dict()
    sound = self.dataset.get_sound(record)
    self.assertEqual(sound.context.length, 8000)  # 0.5s at 16kHz
    self.assertLen(sound.waveform, 8000)
    self.assertEqual(sound.waveform.dtype, np.float32)

  def test_sound_without_optional_fields(self):
    """get_sound works even if 'speaker' and 'text' are missing."""
    record = {'uttid': 'utt_001', 'wav': 'wavs/utt_001.wav'}
    sound = self.dataset.get_sound(record)
    self.assertIsInstance(sound, types.Sound)
    self.assertIsNone(sound.context.speaker_id)
    self.assertIsNone(sound.context.text)

  def test_multiple_sounds_load_independently(self):
    """Each record loads its own waveform."""
    s1 = self.dataset.get_sound(self.task_data.iloc[0].to_dict())
    s2 = self.dataset.get_sound(self.task_data.iloc[1].to_dict())
    self.assertEqual(s1.context.id, 'utt_001')
    self.assertEqual(s2.context.id, 'utt_002')
    # Waveforms are random, so they should differ.
    self.assertFalse(np.array_equal(s1.waveform, s2.waveform))


class AudioImageDatasetGetImageTest(absltest.TestCase):
  """Tests for AudioImageDataset.get_image()."""

  def setUp(self):
    super().setUp()
    self.base_dir = self.create_tempdir().full_path
    self.task_data = _create_mock_audio_image_data(self.base_dir)
    self.dataset = _StubDataset(self.base_dir, self.task_data)

  def test_returns_image_type(self):
    record = {'image': 'images/img_001.jpg'}
    image = self.dataset.get_image(record)
    self.assertIsInstance(image, types.Image)

  def test_image_context_fields(self):
    record = {'image': 'images/img_001.jpg'}
    image = self.dataset.get_image(record)
    self.assertEqual(image.context.id, 'images/img_001.jpg')
    self.assertEqual(image.context.height, 32)
    self.assertEqual(image.context.width, 48)
    self.assertEqual(image.context.channels, 3)

  def test_image_pixel_data(self):
    record = {'image': 'images/img_001.jpg'}
    image = self.dataset.get_image(record)
    self.assertEqual(image.image.shape, (32, 48, 3))
    self.assertEqual(image.image.dtype, np.uint8)

  def test_different_image_dimensions(self):
    """Second image has different dimensions (24x36)."""
    record = {'image': 'images/img_002.jpg'}
    image = self.dataset.get_image(record)
    self.assertEqual(image.context.height, 24)
    self.assertEqual(image.context.width, 36)
    self.assertEqual(image.image.shape, (24, 36, 3))

  def test_grayscale_image_converted_to_rgb(self):
    """A grayscale JPEG should be converted to 3-channel RGB."""
    gray_path = os.path.join(self.base_dir, 'images', 'gray.jpg')
    PILImage.fromarray(
        np.random.randint(0, 255, (16, 16), dtype=np.uint8), mode='L'
    ).save(gray_path)

    record = {'image': 'images/gray.jpg'}
    image = self.dataset.get_image(record)
    self.assertEqual(image.context.channels, 3)
    self.assertEqual(image.image.shape, (16, 16, 3))


class AudioImageDatasetGetUniqueImagesTest(absltest.TestCase):
  """Tests for AudioImageDataset.get_unique_images()."""

  def setUp(self):
    super().setUp()
    self.base_dir = self.create_tempdir().full_path
    self.task_data = _create_mock_audio_image_data(self.base_dir)
    self.dataset = _StubDataset(self.base_dir, self.task_data)

  def test_returns_unique_images(self):
    unique_images = self.dataset.get_unique_images()
    self.assertLen(unique_images, 2)

  def test_unique_image_ids(self):
    unique_images = self.dataset.get_unique_images()
    ids = {r['image'] for r in unique_images}
    self.assertEqual(ids, {'images/img_001.jpg', 'images/img_002.jpg'})

  def test_each_record_is_dict_with_image_key(self):
    for record in self.dataset.get_unique_images():
      self.assertIn('image', record)
      self.assertIsInstance(record['image'], str)

  def test_preserves_order_from_dataframe(self):
    """Unique images should appear in order of first occurrence."""
    unique_images = self.dataset.get_unique_images()
    self.assertEqual(unique_images[0]['image'], 'images/img_001.jpg')
    self.assertEqual(unique_images[1]['image'], 'images/img_002.jpg')


class ExtractTarGzTest(absltest.TestCase):
  """Tests for extract_tar_gz()."""

  def test_extract_tar_gz(self):
    src_dir = self.create_tempdir().full_path
    dest_dir = self.create_tempdir().full_path

    # Create a file and tar it.
    os.makedirs(os.path.join(src_dir, 'subdir'))
    with open(os.path.join(src_dir, 'subdir', 'hello.txt'), 'w') as f:
      f.write('world')

    tar_path = os.path.join(src_dir, 'test.tar.gz')
    with tarfile.open(tar_path, 'w:gz') as tar:
      tar.add(os.path.join(src_dir, 'subdir'), arcname='subdir')

    audio_image_base.extract_tar_gz(tar_path, dest_dir)

    extracted_file = os.path.join(dest_dir, 'subdir', 'hello.txt')
    self.assertTrue(os.path.exists(extracted_file))
    with open(extracted_file) as f:
      self.assertEqual(f.read(), 'world')


class ExtractZipTest(absltest.TestCase):
  """Tests for extract_zip()."""

  def test_extract_zip(self):
    src_dir = self.create_tempdir().full_path
    dest_dir = self.create_tempdir().full_path

    # Create a file and zip it.
    file_path = os.path.join(src_dir, 'data.txt')
    with open(file_path, 'w') as f:
      f.write('hello zip')

    zip_path = os.path.join(src_dir, 'test.zip')
    with zipfile.ZipFile(zip_path, 'w') as zf:
      zf.write(file_path, 'data.txt')

    audio_image_base.extract_zip(zip_path, dest_dir)

    extracted_file = os.path.join(dest_dir, 'data.txt')
    self.assertTrue(os.path.exists(extracted_file))
    with open(extracted_file) as f:
      self.assertEqual(f.read(), 'hello zip')


if __name__ == '__main__':
  absltest.main()
