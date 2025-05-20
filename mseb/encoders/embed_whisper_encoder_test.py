# Copyright 2024 The MSEB Authors.
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
from mseb import encoder
from mseb.encoders import embed_whisper_encoder
import numpy as np
import numpy.testing as npt
import pyarrow.parquet as pq
import tensorflow_hub as tf_hub
import whisper


class EmbedWhisperEncoderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, 'testdata'
    )
    self.svq_samples = pq.ParquetFile(
        os.path.join(testdata_path, 'en_us.parquet')
    )

  def test_gecko_whisper_encoder(self):
    svq_example = self.svq_samples.read_row_group(0)
    waveform = svq_example['waveform'].to_numpy()[0]
    waveform = waveform.astype(np.float32) / 32767.0
    sample_rate = 48000

    gecko_whisper_encoder = embed_whisper_encoder.GeckoWhisperEncoder(
        whisper_model=whisper.load_model('base', device='cpu'),
        gecko_model=tf_hub.load(
            'http://tfhub/model/@gecko/gecko-110m-i18n-cpu/1'
        ),
    )

    context = encoder.ContextParams(language='en', sample_rate=sample_rate)
    timestamps, embeddings = gecko_whisper_encoder.encode(waveform, context)
    npt.assert_equal(timestamps.shape, [1, 2])
    npt.assert_equal(timestamps[0, 0] >= 0.0, True)
    npt.assert_equal(timestamps[0, 1] <= waveform.shape[0] / sample_rate, True)
    npt.assert_equal(embeddings.shape, (1, 768))


if __name__ == '__main__':
  absltest.main()
