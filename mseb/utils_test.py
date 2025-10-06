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

import io
import os

from absl.testing import absltest
from mseb import types
from mseb import utils
import numpy as np
from scipy.io import wavfile
import soundfile


class UtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = self.create_tempdir().full_path
    # Create a mock 16-bit PCM WAV file
    self.sample_rate = 16000
    self.duration_s = 2
    self.num_samples = self.sample_rate * self.duration_s
    self.mock_audio_path = os.path.join(self.test_dir, 'mock_audio.wav')
    t = np.linspace(0., self.duration_s, self.num_samples, endpoint=False)
    amplitude = np.iinfo(np.int16).max * 0.5
    self.data = (amplitude * np.sin(2. * np.pi * 440. * t)).astype(np.int16)
    soundfile.write(self.mock_audio_path, self.data, self.sample_rate)

  def test_read_audio_loads_correctly(self):
    waveform, sr = utils.read_audio(self.mock_audio_path)
    self.assertEqual(sr, self.sample_rate)
    self.assertEqual(waveform.shape[0], self.num_samples)
    self.assertIsInstance(waveform, np.ndarray)
    self.assertEqual(waveform.dtype, np.float32)
    # Check that normalization happened (values are between -1.0 and 1.0)
    self.assertLessEqual(np.max(np.abs(waveform)), 1.0)

  def test_read_audio_resamples_correctly(self):
    target_sr = 8000
    waveform, sr = utils.read_audio(self.mock_audio_path, target_sr=target_sr)
    self.assertEqual(sr, target_sr)
    # The number of samples should be proportionally smaller
    expected_samples = self.num_samples * (target_sr / self.sample_rate)
    self.assertEqual(waveform.shape[0], expected_samples)

  def test_read_audio_handles_no_resampling_if_sr_matches(self):
    waveform, sr = utils.read_audio(
        self.mock_audio_path, target_sr=self.sample_rate
    )
    self.assertEqual(sr, self.sample_rate)
    self.assertEqual(waveform.shape[0], self.num_samples)

  def test_wav_bytes_to_waveform_loads_correctly(self):
    with open(self.mock_audio_path, 'rb') as f:
      wav_bytes = f.read()
    waveform, sr = utils.wav_bytes_to_waveform(wav_bytes)
    self.assertEqual(sr, self.sample_rate)
    self.assertEqual(waveform.shape[0], self.num_samples)
    self.assertIsInstance(waveform, np.ndarray)
    self.assertEqual(waveform.dtype, np.float32)
    # Check that normalization happened (values are between -1.0 and 1.0)
    self.assertLessEqual(np.max(np.abs(waveform)), 1.0)


class TestSoundUtils(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.sample_rate = 16000
    self.num_samples = self.sample_rate * 1
    t = np.linspace(0., 1., self.num_samples, endpoint=False)
    amplitude = np.iinfo(np.int16).max * 0.5
    self.reference_data_int16 = (
        (amplitude * np.sin(2. * np.pi * 440. * t)).astype(np.int16)
    )
    self.mock_context = types.SoundContextParams(
        id='sound',
        sample_rate=self.sample_rate,
        length=self.num_samples
    )

  def test_sound_to_wav_bytes_from_float32(self):
    float_waveform = (
        self.reference_data_int16.astype(np.float32) / np.iinfo(np.int16).max
    )
    sound_object = types.Sound(
        waveform=float_waveform,
        context=self.mock_context
    )
    wav_bytes = utils.sound_to_wav_bytes(sound_object)
    reread_rate, reread_data = wavfile.read(io.BytesIO(wav_bytes))
    self.assertEqual(reread_rate, self.sample_rate)
    self.assertLen(reread_data, self.num_samples)
    self.assertEqual(reread_data.dtype, np.int16)
    np.testing.assert_allclose(
        reread_data,
        self.reference_data_int16,
        atol=1
    )

  def test_sound_to_wav_bytes_from_int16(self):
    int16_waveform = self.reference_data_int16
    sound_object = types.Sound(
        waveform=int16_waveform,
        context=self.mock_context
    )
    wav_bytes = utils.sound_to_wav_bytes(sound_object)
    reread_rate, reread_data = wavfile.read(io.BytesIO(wav_bytes))
    self.assertEqual(reread_rate, self.sample_rate)
    self.assertLen(reread_data, self.num_samples)
    np.testing.assert_allclose(
        reread_data,
        int16_waveform,
        atol=1
    )

if __name__ == '__main__':
  absltest.main()
