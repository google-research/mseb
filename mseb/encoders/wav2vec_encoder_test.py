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
from mseb.encoders import wav2vec_encoder
import numpy as np
import numpy.testing as npt
import pyarrow.parquet as pq


class Wav2VecEncoderTest(absltest.TestCase):

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
        sample_rate=48000,
    )
    self.model_name = 'facebook/wav2vec2-base'

  def test_wav2vec_encoder_last_pooling(self):
    transform_fn = lambda x: x
    enc = wav2vec_encoder.Wav2VecEncoder(
        self.model_name, transform_fn, device='cpu', pooling='last')
    timestamp, embedding = enc.encode(self.waveform, self.context)
    npt.assert_equal(timestamp, [[0, 7.5]])
    npt.assert_equal(embedding.shape, [1, 768])

  def test_wav2vec_encoder_mean_pooling(self):
    transform_fn = lambda x: x
    enc = wav2vec_encoder.Wav2VecEncoder(
        self.model_name, transform_fn, device='cpu', pooling='mean')
    timestamp, embedding = enc.encode(self.waveform, self.context)
    npt.assert_equal(timestamp, [[0, 7.5]])
    npt.assert_equal(embedding.shape, [1, 768])

  def test_wav2vec_encoder_max_pooling(self):
    transform_fn = lambda x: x
    enc = wav2vec_encoder.Wav2VecEncoder(
        self.model_name, transform_fn, device='cpu', pooling='max')
    timestamp, embedding = enc.encode(self.waveform, self.context)
    npt.assert_equal(timestamp, [[0, 7.5]])
    npt.assert_equal(embedding.shape, [1, 768])

  def test_wav2vec_encoder_normalized_last_pooling(self):
    transform_fn = wav2vec_encoder.normalize_embeddings
    enc = wav2vec_encoder.Wav2VecEncoder(
        self.model_name, transform_fn, device='cpu', pooling='last')
    timestamp, embedding = enc.encode(self.waveform, self.context)
    npt.assert_equal(timestamp, [[0, 7.5]])
    npt.assert_equal(embedding.shape, [1, 768])
    npt.assert_allclose(
        np.linalg.norm(embedding, axis=1, keepdims=False),
        1.0,
        err_msg='Pooled embedding norm is not 1.0',
        rtol=1e-4,
    )

  def test_wav2vec_encoder_normalized_mean_pooling(self):
    transform_fn = wav2vec_encoder.normalize_embeddings
    enc = wav2vec_encoder.Wav2VecEncoder(
        self.model_name, transform_fn, device='cpu', pooling='mean')
    timestamp, embedding = enc.encode(self.waveform, self.context)
    npt.assert_equal(timestamp, [[0, 7.5]])
    npt.assert_equal(embedding.shape, [1, 768])
    npt.assert_array_less(embedding, 1.0)
    npt.assert_array_less(-1.0, embedding)

  def test_wav2vec_encoder_normalized_max_pooling(self):
    transform_fn = wav2vec_encoder.normalize_embeddings
    enc = wav2vec_encoder.Wav2VecEncoder(
        self.model_name, transform_fn, device='cpu', pooling='max')
    timestamp, embedding = enc.encode(self.waveform, self.context)
    npt.assert_equal(timestamp, [[0, 7.5]])
    npt.assert_equal(embedding.shape, [1, 768])
    npt.assert_array_less(embedding, 1.0)
    npt.assert_array_less(-1.0, embedding)


if __name__ == '__main__':
  absltest.main()
