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
from unittest import mock

from absl.testing import absltest
from mseb import types
from mseb.encoders import whisper_encoder
import numpy as np
import numpy.testing as npt
import pyarrow.parquet as pq


def whisper_cache_context(name: str):
  # Use a unique cache directory for each test to avoid collisions when
  # running tests in parallel via pytest.
  original_xdg_cache_home = os.path.join(os.path.expanduser('~'), '.cache')
  new_xdg_cache_home = os.path.join(original_xdg_cache_home, f'{name}_whisper')
  return mock.patch.dict(os.environ, {'XDG_CACHE_HOME': new_xdg_cache_home})


class SpeechToTextEncoderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(whisper_cache_context(self.__class__.__name__))
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, 'testdata'
    )
    self.svq_samples = pq.ParquetFile(
        os.path.join(testdata_path, 'en_us.parquet')
    )

  def test_preprocess(self):
    whisper_encoder_instance = whisper_encoder.SpeechToTextEncoder(
        model_path='base', device='cpu'
    )
    whisper_encoder_instance.setup()
    svq_example = self.svq_samples.read_row_group(0)
    waveform = svq_example['waveform'].to_numpy()[0]
    waveform = waveform.astype(np.float32) / 32767.0
    sample_rate = 48000
    preprocessed = whisper_encoder_instance.preprocess(waveform, sample_rate)
    npt.assert_allclose(preprocessed.shape[0], waveform.shape[0] / 3)

  def test_encode_sentence_level(self):
    whisper_encoder_instance = whisper_encoder.SpeechToTextEncoder(
        model_path='base', device='cpu'
    )
    whisper_encoder_instance.setup()
    svq_example = self.svq_samples.read_row_group(0)
    waveform = svq_example['waveform'].to_numpy()[0]
    waveform = waveform.astype(np.float32) / 32767.0
    sample_rate = 48000
    params = types.SoundContextParams(
        sample_rate=sample_rate,
        length=waveform.shape[0],
        language='en',
        id='test',
    )
    sound = types.Sound(waveform=waveform, context=params)
    results = whisper_encoder_instance.encode([sound])
    self.assertLen(results, 1)
    result = results[0]
    self.assertIsInstance(result, types.SoundEmbedding)
    npt.assert_equal(result.timestamps.shape, [1, 2])
    npt.assert_equal(result.timestamps[0, 0] >= 0.0, True)
    npt.assert_equal(
        result.timestamps[0, 1] <= waveform.shape[0] / sample_rate, True
    )
    npt.assert_equal(
        result.embedding,
        [' How many members does the National Labor Relations Board have?'],
    )

  def test_encode_word_level(self):
    whisper_encoder_instance = whisper_encoder.SpeechToTextEncoder(
        model_path='base', device='cpu', word_timestamps=True
    )
    whisper_encoder_instance.setup()
    svq_example = self.svq_samples.read_row_group(0)
    waveform = svq_example['waveform'].to_numpy()[0]
    waveform = waveform.astype(np.float32) / 32767.0
    params = types.SoundContextParams(
        sample_rate=48000,
        length=waveform.shape[0],
        language='en',
        id='test',
    )
    sound = types.Sound(waveform=waveform, context=params)
    results = whisper_encoder_instance.encode([sound])
    self.assertLen(results, 1)
    result = results[0]
    self.assertIsInstance(result, types.SoundEmbedding)
    npt.assert_equal(result.timestamps.shape[0], result.embedding.shape[0])


class ForcedAlignmentEncoder2Test(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(whisper_cache_context(self.__class__.__name__))
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, 'testdata'
    )
    self.svq_samples = pq.ParquetFile(
        os.path.join(testdata_path, 'en_us.parquet')
    )

  def test_encode_speech_transcript_truth(self):
    whisper_encoder_instance = whisper_encoder.ForcedAlignmentEncoder(
        model_path='base', device='cpu', language='en'
    )
    whisper_encoder_instance.setup()
    svq_example = self.svq_samples.read_row_group(0)
    waveform = svq_example['waveform'].to_numpy()[0]
    waveform = waveform.astype(np.float32) / 32767.0
    params = types.SoundContextParams(
        sample_rate=48000,
        length=waveform.shape[0],
        language='en',
        text=svq_example['text'].to_numpy()[0],
        id='test',
    )
    sound = types.Sound(waveform=waveform, context=params)
    results = whisper_encoder_instance.encode([sound])
    self.assertLen(results, 1)
    result = results[0]
    self.assertIsInstance(result, types.SoundEmbedding)
    npt.assert_equal(result.timestamps.shape[0], result.embedding.shape[0])


class PooledAudioEncoderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(whisper_cache_context(self.__class__.__name__))
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, 'testdata'
    )
    svq_samples = pq.ParquetFile(os.path.join(testdata_path, 'en_us.parquet'))
    svq_example = svq_samples.read_row_group(0)
    waveform = svq_example['waveform'].to_numpy()[0]
    self.waveform = waveform.astype(np.float32) / 32767.0
    self.params = types.SoundContextParams(
        sample_rate=48000,
        length=waveform.shape[0],
        id='test',
    )
    self.sound = types.Sound(waveform=self.waveform, context=self.params)
    self.model_path = 'base'

  def test_pool_fn(self):
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    enc = whisper_encoder.PooledAudioEncoder(self.model_path)
    npt.assert_equal(enc.pool_fn(x), [[1.0, 2.0], [3.0, 4.0]])
    enc = whisper_encoder.PooledAudioEncoder(self.model_path, pooling='last')
    npt.assert_equal(enc.pool_fn(x), [[3.0, 4.0]])
    enc = whisper_encoder.PooledAudioEncoder(self.model_path, pooling='mean')
    npt.assert_equal(enc.pool_fn(x), [[2.0, 3.0]])
    enc = whisper_encoder.PooledAudioEncoder(self.model_path, pooling='max')
    npt.assert_equal(enc.pool_fn(x), [[3.0, 4.0]])

  def test_encode_no_pooling(self):
    enc = whisper_encoder.PooledAudioEncoder(self.model_path)
    enc.setup()
    results = enc.encode([self.sound])
    self.assertLen(results, 1)
    result = results[0]
    self.assertIsInstance(result, types.SoundEmbedding)
    npt.assert_equal(result.timestamps, [[0, 7.5]])
    npt.assert_equal(result.embedding.shape, [375, 512])

  def test_encode_last_pooling(self):
    enc = whisper_encoder.PooledAudioEncoder(self.model_path, pooling='last')
    enc.setup()
    results = enc.encode([self.sound])
    self.assertLen(results, 1)
    result = results[0]
    self.assertIsInstance(result, types.SoundEmbedding)
    npt.assert_equal(result.timestamps, [[0, 7.5]])
    npt.assert_equal(result.embedding.shape, [1, 512])

  def test_encode_mean_pooling(self):
    enc = whisper_encoder.PooledAudioEncoder(self.model_path, pooling='mean')
    enc.setup()
    results = enc.encode([self.sound])
    self.assertLen(results, 1)
    result = results[0]
    self.assertIsInstance(result, types.SoundEmbedding)
    npt.assert_equal(result.timestamps, [[0, 7.5]])
    npt.assert_equal(result.embedding.shape, [1, 512])

  def test_encode_max_pooling(self):
    enc = whisper_encoder.PooledAudioEncoder(self.model_path, pooling='max')
    enc.setup()
    results = enc.encode([self.sound])
    self.assertLen(results, 1)
    result = results[0]
    self.assertIsInstance(result, types.SoundEmbedding)
    npt.assert_equal(result.timestamps, [[0, 7.5]])
    npt.assert_equal(result.embedding.shape, [1, 512])


if __name__ == '__main__':
  absltest.main()
