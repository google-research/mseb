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
import pathlib

from absl.testing import absltest
from mseb import types
from mseb.encoders import soundstream_encoder
import numpy as np
import numpy.testing as npt
import pyarrow.parquet as pq


class SounStreamEncoderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, 'testdata')
    svq_samples = pq.ParquetFile(
        os.path.join(testdata_path, 'en_us.parquet'))
    svq_example = svq_samples.read_row_group(0)
    waveform = svq_example['waveform'].to_numpy()[0]
    self.waveform = waveform.astype(np.float32) / 32767.0
    self.context = types.SoundContextParams(
        id='0',
        length=waveform.shape[0],
        language='en',
        sample_rate=48000,
        text=svq_example['text'].to_numpy()[0],
    )

  def test_soundstream_encoder_without_quantization(self):
    enc = soundstream_encoder.SoundStreamEncoder()
    sound_embedding = enc.encode(
        types.Sound(waveform=self.waveform, context=self.context)
    )
    npt.assert_equal(sound_embedding.timestamps.shape, [375, 2])
    npt.assert_equal(sound_embedding.embedding.shape, [375, 64])

  def test_soundstream_encoder_quantization_9200bps(self):
    enc = soundstream_encoder.SoundStreamEncoder(
        bits_per_second=9200, quantize=True)
    sound_embedding = enc.encode(
        types.Sound(waveform=self.waveform, context=self.context)
    )
    npt.assert_equal(sound_embedding.timestamps.shape, [375, 2])
    npt.assert_equal(sound_embedding.embedding.shape, [375, 46])

  def test_soundstream_encoder_quantization_4600bps(self):
    enc = soundstream_encoder.SoundStreamEncoder(
        bits_per_second=4600, quantize=True)
    sound_embedding = enc.encode(
        types.Sound(waveform=self.waveform, context=self.context)
    )
    npt.assert_equal(sound_embedding.timestamps.shape, [375, 2])
    npt.assert_equal(sound_embedding.embedding.shape, [375, 23])


if __name__ == '__main__':
  absltest.main()
