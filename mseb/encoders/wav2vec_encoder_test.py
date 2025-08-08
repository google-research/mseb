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
from typing import Callable

from absl.testing import absltest
from mseb import types
from mseb.encoders import wav2vec_encoder
import numpy as np
import numpy.testing as npt
import pyarrow.parquet as pq
import transformers


class MockWav2VecEncoder(wav2vec_encoder.Wav2VecEncoder):
  """Mock Wav2VecEncoder for testing."""

  def __init__(
      self,
      transform_fn: Callable[..., np.ndarray],
      device: str | None = None,
      pooling: str = 'mean',
  ):
    super().__init__(
        model_path='dummy_model_path',
        transform_fn=transform_fn,
        device=device,
        pooling=pooling,
    )

  def setup(self):
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, 'testdata'
    )
    self.processor = transformers.Wav2Vec2Processor(
        feature_extractor=transformers.Wav2Vec2FeatureExtractor(),
        tokenizer=transformers.Wav2Vec2CTCTokenizer(
            os.path.join(testdata_path, 'vocab.json')
        ),
    )
    self.model = transformers.Wav2Vec2Model(transformers.Wav2Vec2Config())
    self.model.eval()  # Set model to evaluation mode for inference
    self.model.to(self.device)
    self._model_loaded = True


class Wav2VecEncoderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, 'testdata'
    )
    svq_samples = pq.ParquetFile(os.path.join(testdata_path, 'en_us.parquet'))
    svq_example = svq_samples.read_row_group(0)
    waveform = svq_example['waveform'].to_numpy()[0]
    self.waveform = waveform.astype(np.float32) / 32767.0
    self.params = types.SoundContextParams(
        sample_rate=48000,
        length=len(self.waveform),
    )

  def test_wav2vec_encoder_last_pooling(self):
    transform_fn = lambda x: x
    enc = MockWav2VecEncoder(transform_fn, device='cpu', pooling='last')
    embedding, timestamp = enc.encode(self.waveform, self.params)
    npt.assert_equal(timestamp, [[0, 7.5]])
    npt.assert_equal(embedding.shape, [1, 768])

  def test_wav2vec_encoder_mean_pooling(self):
    transform_fn = lambda x: x
    enc = MockWav2VecEncoder(transform_fn, pooling='mean')
    embedding, timestamp = enc.encode(self.waveform, self.params)
    npt.assert_equal(timestamp, [[0, 7.5]])
    npt.assert_equal(embedding.shape, [1, 768])

  def test_wav2vec_encoder_max_pooling(self):
    transform_fn = lambda x: x
    enc = MockWav2VecEncoder(transform_fn, device='cpu', pooling='max')
    embedding, timestamp = enc.encode(self.waveform, self.params)
    npt.assert_equal(timestamp, [[0, 7.5]])
    npt.assert_equal(embedding.shape, [1, 768])

  def test_wav2vec_encoder_normalized_last_pooling(self):
    transform_fn = wav2vec_encoder.normalize_embeddings
    enc = MockWav2VecEncoder(transform_fn, pooling='last')
    embedding, timestamp = enc.encode(self.waveform, self.params)
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
    enc = MockWav2VecEncoder(transform_fn, device='cpu', pooling='mean')
    embedding, timestamp = enc.encode(self.waveform, self.params)
    npt.assert_equal(timestamp, [[0, 7.5]])
    npt.assert_equal(embedding.shape, [1, 768])
    npt.assert_array_less(embedding, 1.0)
    npt.assert_array_less(-1.0, embedding)

  def test_wav2vec_encoder_normalized_max_pooling(self):
    transform_fn = wav2vec_encoder.normalize_embeddings
    enc = MockWav2VecEncoder(transform_fn, pooling='max')
    embedding, timestamp = enc.encode(self.waveform, self.params)
    npt.assert_equal(timestamp, [[0, 7.5]])
    npt.assert_equal(embedding.shape, [1, 768])
    npt.assert_array_less(embedding, 1.0)
    npt.assert_array_less(-1.0, embedding)


if __name__ == '__main__':
  absltest.main()
