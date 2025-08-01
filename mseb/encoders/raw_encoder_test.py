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
from mseb.encoders import raw_encoder
import numpy as np
import numpy.testing as npt
import pyarrow.parquet as pq


class RawEncoderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, 'testdata'
    )
    svq_samples = pq.ParquetFile(
        os.path.join(testdata_path, 'en_us.parquet')
    )
    svq_example = svq_samples.read_row_group(0)
    waveform = svq_example['waveform'].to_numpy()[0]
    self.waveform = waveform.astype(np.float32) / 32767.0

    # Encoder configuration parameters
    self.sample_rate = 48000
    self.frame_length = 16000
    self.frame_step = 8000
    self.num_mel_bins = 128

    self.params = types.SoundContextParams(
        sample_rate=self.sample_rate,
        length=len(waveform),
    )

    self.fft_length = int(2 ** np.ceil(np.log2(self.frame_length)))
    self.mel_matrix = raw_encoder.linear_to_mel_weight_matrix(
        num_mel_bins=self.num_mel_bins,
        num_spectrogram_bins=1 + int(self.fft_length / 2),
        sample_rate=self.sample_rate,
        lower_edge_hertz=125,
        upper_edge_hertz=7500,
    )

  def test_raw_encoder(self):
    enc = raw_encoder.RawEncoder(
        transform_fn=lambda x: x,
        frame_length=self.frame_length,
        frame_step=self.frame_step,
    )
    waveform_embeddings, embedding_timestamps = enc.encode(
        self.waveform,
        self.params
    )
    self.assertEqual(waveform_embeddings.shape,
                     (44, self.frame_length))
    self.assertEqual(embedding_timestamps.shape,
                     (44, 2))
    for i in range(44):
      start = i * self.frame_step
      end = start + self.frame_length
      npt.assert_equal(embedding_timestamps[i], [start, end])

  def test_raw_encoder_mean_pooling(self):
    enc = raw_encoder.RawEncoder(
        transform_fn=lambda x: x,
        frame_length=self.frame_length,
        frame_step=self.frame_step,
        pooling='mean',
    )
    waveform_embeddings, embedding_timestamps = enc.encode(
        self.waveform,
        self.params
    )
    self.assertEqual(waveform_embeddings.shape,
                     (1, self.frame_length))
    npt.assert_equal(embedding_timestamps,
                     [[0, len(self.waveform)]])

  def test_raw_encoder_spectrogram_transform(self):
    enc = raw_encoder.RawEncoder(
        transform_fn=raw_encoder.spectrogram_transform,
        frame_length=self.frame_length,
        frame_step=self.frame_step,
    )
    waveform_embeddings, embedding_timestamps = enc.encode(
        self.waveform,
        self.params
    )
    self.assertEqual(waveform_embeddings.shape,
                     (44, 1 + self.fft_length // 2))
    self.assertEqual(embedding_timestamps.shape, (44, 2))

  def test_raw_encoder_spectrogram_transform_max_pooling(self):
    enc = raw_encoder.RawEncoder(
        transform_fn=raw_encoder.spectrogram_transform,
        frame_length=self.frame_length,
        frame_step=self.frame_step,
        pooling='max',
    )
    waveform_embeddings, embedding_timestamps = enc.encode(
        self.waveform,
        self.params
    )
    self.assertEqual(waveform_embeddings.shape,
                     (1, 1 + self.fft_length // 2))
    npt.assert_equal(embedding_timestamps,
                     [[0, len(self.waveform)]])

  def test_raw_encoder_log_mel_transform(self):
    enc = raw_encoder.RawEncoder(
        transform_fn=raw_encoder.log_mel_transform,
        frame_length=self.frame_length,
        frame_step=self.frame_step,
        transform_fn_kwargs={'mel_matrix': self.mel_matrix},
    )
    waveform_embeddings, embedding_timestamps = enc.encode(
        self.waveform,
        self.params
    )
    self.assertEqual(waveform_embeddings.shape,
                     (44, self.num_mel_bins))
    self.assertEqual(embedding_timestamps.shape, (44, 2))

  def test_raw_encoder_log_mel_transform_last_pooling(self):
    enc = raw_encoder.RawEncoder(
        transform_fn=raw_encoder.log_mel_transform,
        frame_length=self.frame_length,
        frame_step=self.frame_step,
        pooling='last',
        transform_fn_kwargs={'mel_matrix': self.mel_matrix},
    )
    waveform_embeddings, embedding_timestamps = enc.encode(
        self.waveform,
        self.params
    )
    self.assertEqual(waveform_embeddings.shape, (1, self.num_mel_bins))
    npt.assert_equal(embedding_timestamps, [[0, len(self.waveform)]])

  def test_initialization_missing_params(self):
    with self.assertRaises(ValueError):
      enc = raw_encoder.RawEncoder(transform_fn=lambda x: x)
      enc.encode(self.waveform, self.params)

  def test_encode_with_runtime_kwargs(self):
    runtime_mel_matrix = self.mel_matrix[:, ::2]
    self.assertEqual(runtime_mel_matrix.shape[1], self.num_mel_bins // 2)

    enc = raw_encoder.RawEncoder(
        transform_fn=raw_encoder.log_mel_transform,
        frame_length=self.frame_length,
        frame_step=self.frame_step,
        transform_fn_kwargs={'mel_matrix': self.mel_matrix},
    )
    waveform_embeddings, _ = enc.encode(
        self.waveform,
        self.params,
        mel_matrix=runtime_mel_matrix
    )
    self.assertEqual(waveform_embeddings.shape, (44, self.num_mel_bins // 2))

  def test_encode_short_waveform(self):
    short_waveform = self.waveform[: self.frame_length - 1]
    enc = raw_encoder.RawEncoder(
        transform_fn=lambda x: x,
        frame_length=self.frame_length,
        frame_step=self.frame_step,
    )
    waveform_embeddings, embedding_timestamps = enc.encode(
        short_waveform,
        types.SoundContextParams(
            sample_rate=self.sample_rate,
            length=len(short_waveform),
        )
    )
    self.assertEqual(waveform_embeddings.shape, (0,))
    self.assertEqual(embedding_timestamps.shape, (0,))

  def test_encode_batch(self):
    enc = raw_encoder.RawEncoder(
        transform_fn=lambda x: x,
        frame_length=self.frame_length,
        frame_step=self.frame_step,
        pooling='mean',
    )
    short_waveform = self.waveform[: len(self.waveform) // 2]
    batch_waveforms = [self.waveform, short_waveform]
    batch_params = [
        self.params,
        types.SoundContextParams(
            sample_rate=self.sample_rate,
            length=len(short_waveform)
        ),
    ]

    results = enc.encode_batch(batch_waveforms, batch_params)

    self.assertLen(results, 2)
    self.assertEqual(results[0][0].shape, (1, self.frame_length))
    npt.assert_equal(results[0][1], [[0, len(batch_waveforms[0])]])
    self.assertEqual(results[1][0].shape, (1, self.frame_length))
    npt.assert_equal(results[1][1], [[0, len(batch_waveforms[1])]])


if __name__ == '__main__':
  absltest.main()
