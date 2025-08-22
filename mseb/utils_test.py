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

import os

from absl.testing import absltest
from mseb import utils
import numpy as np
import soundfile


class UtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = self.create_tempdir().full_path
    # Create a mock 16-bit PCM WAV file
    self.sample_rate = 16000
    self.duration_s = 2
    self.num_samples = self.sample_rate * self.duration_s
    self.mock_audio_path = os.path.join(self.test_dir, "mock_audio.wav")
    t = np.linspace(0., self.duration_s, self.num_samples, endpoint=False)
    amplitude = np.iinfo(np.int16).max * 0.5
    data = (amplitude * np.sin(2. * np.pi * 440. * t)).astype(np.int16)
    soundfile.write(self.mock_audio_path, data, self.sample_rate)

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


if __name__ == "__main__":
  absltest.main()
