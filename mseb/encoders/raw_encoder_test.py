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
from mseb import encoder
from mseb.encoders import raw_encoder
import numpy as np
import numpy.testing as npt
import pyarrow.parquet as pq


class RawEncoderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, 'testdata')
    svq_samples = pq.ParquetFile(
        os.path.join(testdata_path, 'en_us.parquet'))
    svq_example = svq_samples.read_row_group(0)
    waveform = svq_example['waveform'].to_numpy()[0]
    self.waveform = waveform.astype(np.float32) / 32767.0
    self.context = encoder.ContextParams(
        frame_length=16000,
        frame_step=8000,
        sample_rate=48000,
    )
    self.fft_length = int(2 ** np.ceil(np.log2(self.context.frame_length)))
    self.mel_matrix = raw_encoder.linear_to_mel_weight_matrix(
        num_mel_bins=128,
        num_spectrogram_bins=1 + int(self.fft_length / 2),
        sample_rate=48000,
        lower_edge_hertz=125,
        upper_edge_hertz=7500,
    )

  def test_raw_encoder(self):
    transform_fn = lambda x: x
    enc = raw_encoder.RawEncoder(transform_fn)
    timestamps, embeddings = enc.encode(self.waveform, self.context)
    npt.assert_equal(timestamps.shape, [44, 2])
    for i in range(44):
      npt.assert_equal(timestamps[i], [i / 6, (i + 2) / 6])
    npt.assert_equal(embeddings.shape, [44, self.context.frame_length])

  def test_raw_encoder_last_pooling(self):
    transform_fn = lambda x: x
    enc = raw_encoder.RawEncoder(transform_fn, pooling='last')
    timestamps, embeddings = enc.encode(self.waveform, self.context)
    npt.assert_equal(timestamps, [[0, 7.5]])
    npt.assert_equal(embeddings.shape, [1, self.context.frame_length])

  def test_raw_encoder_mean_pooling(self):
    transform_fn = lambda x: x
    enc = raw_encoder.RawEncoder(transform_fn, pooling='mean')
    timestamps, embeddings = enc.encode(self.waveform, self.context)
    npt.assert_equal(timestamps, [[0, 7.5]])
    npt.assert_equal(embeddings.shape, [1, self.context.frame_length])

  def test_raw_encoder_max_pooling(self):
    transform_fn = lambda x: x
    enc = raw_encoder.RawEncoder(transform_fn, pooling='max')
    timestamps, embeddings = enc.encode(self.waveform, self.context)
    npt.assert_equal(timestamps, [[0, 7.5]])
    npt.assert_equal(embeddings.shape, [1, self.context.frame_length])

  def test_raw_encoder_hanning_transform(self):
    transform_fn = raw_encoder.hanning_window_transform
    enc = raw_encoder.RawEncoder(transform_fn)
    timestamps, embeddings = enc.encode(self.waveform, self.context)
    npt.assert_equal(timestamps.shape, [44, 2])
    for i in range(44):
      npt.assert_equal(timestamps[i], [i / 6, (i + 2) / 6])
    npt.assert_equal(embeddings.shape, [44, self.context.frame_length])

  def test_raw_encoder_hanning_transform_last_pooling(self):
    transform_fn = raw_encoder.hanning_window_transform
    enc = raw_encoder.RawEncoder(transform_fn, pooling='last')
    timestamps, embeddings = enc.encode(self.waveform, self.context)
    npt.assert_equal(timestamps, [[0, 7.5]])
    npt.assert_equal(embeddings.shape, [1, self.context.frame_length])

  def test_raw_encoder_hanning_transform_mean_pooling(self):
    transform_fn = raw_encoder.hanning_window_transform
    enc = raw_encoder.RawEncoder(transform_fn, pooling='mean')
    timestamps, embeddings = enc.encode(self.waveform, self.context)
    npt.assert_equal(timestamps, [[0, 7.5]])
    npt.assert_equal(embeddings.shape, [1, self.context.frame_length])

  def test_raw_encoder_hanning_transform_max_pooling(self):
    transform_fn = raw_encoder.hanning_window_transform
    enc = raw_encoder.RawEncoder(transform_fn, pooling='max')
    timestamps, embeddings = enc.encode(self.waveform, self.context)
    npt.assert_equal(timestamps, [[0, 7.5]])
    npt.assert_equal(embeddings.shape, [1, self.context.frame_length])

  def test_raw_encoder_spectrogram_transform(self):
    transform_fn = raw_encoder.spectrogram_transform
    enc = raw_encoder.RawEncoder(transform_fn)
    timestamps, embeddings = enc.encode(self.waveform, self.context)
    npt.assert_equal(timestamps.shape, [44, 2])
    for i in range(44):
      npt.assert_equal(timestamps[i], [i / 6, (i + 2) / 6])
    npt.assert_equal(embeddings.shape, [44, 1 + self.fft_length / 2])

  def test_raw_encoder_spectrogram_transform_last_pooling(self):
    transform_fn = raw_encoder.spectrogram_transform
    enc = raw_encoder.RawEncoder(transform_fn, pooling='last')
    timestamps, embeddings = enc.encode(self.waveform, self.context)
    npt.assert_equal(timestamps, [[0, 7.5]])
    npt.assert_equal(embeddings.shape, [1, 1 + self.fft_length / 2])

  def test_raw_encoder_spectrogram_transform_mean_pooling(self):
    transform_fn = raw_encoder.spectrogram_transform
    enc = raw_encoder.RawEncoder(transform_fn, pooling='mean')
    timestamps, embeddings = enc.encode(self.waveform, self.context)
    npt.assert_equal(timestamps, [[0, 7.5]])
    npt.assert_equal(embeddings.shape, [1, 1 + self.fft_length / 2])

  def test_raw_encoder_spectrogram_transform_max_pooling(self):
    transform_fn = raw_encoder.spectrogram_transform
    enc = raw_encoder.RawEncoder(transform_fn, pooling='max')
    timestamps, embeddings = enc.encode(self.waveform, self.context)
    npt.assert_equal(timestamps, [[0, 7.5]])
    npt.assert_equal(embeddings.shape, [1, 1 + self.fft_length / 2])

  def test_raw_encoder_log_mel_transform(self):
    transform_fn = raw_encoder.log_mel_transform
    enc = raw_encoder.RawEncoder(transform_fn)
    timestamps, embeddings = enc.encode(
        self.waveform, self.context, {'mel_matrix': self.mel_matrix})
    npt.assert_equal(timestamps.shape, [44, 2])
    for i in range(44):
      npt.assert_equal(timestamps[i], [i / 6, (i + 2) / 6])
    npt.assert_equal(embeddings.shape, [44, self.mel_matrix.shape[1]])

  def test_raw_encoder_log_mel_transform_last_pooling(self):
    transform_fn = raw_encoder.log_mel_transform
    enc = raw_encoder.RawEncoder(transform_fn, pooling='last')
    timestamps, embeddings = enc.encode(
        self.waveform, self.context, {'mel_matrix': self.mel_matrix})
    npt.assert_equal(timestamps, [[0, 7.5]])
    npt.assert_equal(embeddings.shape, [1, self.mel_matrix.shape[1]])

  def test_raw_encoder_log_mel_transform_mean_pooling(self):
    transform_fn = raw_encoder.log_mel_transform
    enc = raw_encoder.RawEncoder(transform_fn, pooling='mean')
    timestamps, embeddings = enc.encode(
        self.waveform, self.context, {'mel_matrix': self.mel_matrix})
    npt.assert_equal(timestamps, [[0, 7.5]])
    npt.assert_equal(embeddings.shape, [1, self.mel_matrix.shape[1]])

  def test_raw_encoder_log_mel_transform_max_pooling(self):
    transform_fn = raw_encoder.log_mel_transform
    enc = raw_encoder.RawEncoder(transform_fn, pooling='max')
    timestamps, embeddings = enc.encode(
        self.waveform, self.context, {'mel_matrix': self.mel_matrix})
    npt.assert_equal(timestamps, [[0, 7.5]])
    npt.assert_equal(embeddings.shape, [1, self.mel_matrix.shape[1]])


if __name__ == '__main__':
  absltest.main()
