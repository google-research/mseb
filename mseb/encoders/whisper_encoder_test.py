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
from mseb.encoders import whisper_encoder
import numpy as np
import numpy.testing as npt
import pyarrow.parquet as pq
import whisper


class SpeechToTextEncoderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, 'testdata')
    self.svq_samples = pq.ParquetFile(
        os.path.join(testdata_path, 'en_us.parquet'))
    model = whisper.load_model('base', device='cpu')
    self.whisper_encoder = whisper_encoder.SpeechToTextEncoder(model)

  def test_preprocess(self):
    svq_example = self.svq_samples.read_row_group(0)
    waveform = svq_example['waveform'].to_numpy()[0]
    waveform = waveform.astype(np.float32) / 32767.0
    sample_rate = 48000
    preprocessed = self.whisper_encoder.preprocess(waveform, sample_rate)
    npt.assert_allclose(preprocessed.shape[0], waveform.shape[0] / 3)

  def test_encode_sentence_level(self):
    svq_example = self.svq_samples.read_row_group(0)
    waveform = svq_example['waveform'].to_numpy()[0]
    waveform = waveform.astype(np.float32) / 32767.0
    sample_rate = 48000
    context = encoder.ContextParams(language='en', sample_rate=sample_rate)
    timestamps, embeddings = self.whisper_encoder.encode(waveform, context)
    npt.assert_equal(timestamps.shape, [1, 2])
    npt.assert_equal(timestamps[0, 0] >= 0.0, True)
    npt.assert_equal(timestamps[0, 1] <= waveform.shape[0] / sample_rate, True)
    npt.assert_equal(
        embeddings,
        [' How many members does the National Labor Relations Board have?']
    )

  def test_encode_word_level(self):
    svq_example = self.svq_samples.read_row_group(0)
    waveform = svq_example['waveform'].to_numpy()[0]
    waveform = waveform.astype(np.float32) / 32767.0
    context = encoder.ContextParams(language='en', sample_rate=48000)
    timestamps, embeddings = self.whisper_encoder.encode(
        waveform, context, word_timestamps=True)
    npt.assert_equal(timestamps.shape[0], embeddings.shape[0])


class ForcedAlignmentEncoderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, 'testdata')
    self.svq_samples = pq.ParquetFile(
        os.path.join(testdata_path, 'en_us.parquet'))
    model = whisper.load_model('base', device='cpu')
    self.whisper_encoder = whisper_encoder.ForcedAlignmentEncoder(model, 'en')

  def test_encode_speech_transcript_truth(self):
    svq_example = self.svq_samples.read_row_group(0)
    waveform = svq_example['waveform'].to_numpy()[0]
    waveform = waveform.astype(np.float32) / 32767.0
    context = encoder.ContextParams(language='en',
                                    text=svq_example['text'].to_numpy()[0],
                                    sample_rate=48000)
    timestamps, embeddings = self.whisper_encoder.encode(waveform, context)
    npt.assert_equal(timestamps.shape[0], embeddings.shape[0])


class PooledAudioEncoderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, 'testdata')
    svq_samples = pq.ParquetFile(
        os.path.join(testdata_path, 'en_us.parquet'))
    svq_example = svq_samples.read_row_group(0)
    waveform = svq_example['waveform'].to_numpy()[0]
    self.waveform = waveform.astype(np.float32) / 32767.0
    self.context = encoder.ContextParams(sample_rate=48000)
    self.model = whisper.load_model('base', device='cpu')

  def test_pool_fn(self):
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    enc = whisper_encoder.PooledAudioEncoder(self.model, 'last')
    npt.assert_equal(enc.pool_fn(x), [[3.0, 4.0]])
    enc = whisper_encoder.PooledAudioEncoder(self.model, 'mean')
    npt.assert_equal(enc.pool_fn(x), [[2.0, 3.0]])
    enc = whisper_encoder.PooledAudioEncoder(self.model, 'max')
    npt.assert_equal(enc.pool_fn(x), [[3.0, 4.0]])

  def test_encode_last_pooling(self):
    enc = whisper_encoder.PooledAudioEncoder(self.model, 'last')
    timestamp, embedding = enc.encode(self.waveform, self.context)
    npt.assert_equal(timestamp, [[0, 7.5]])
    npt.assert_equal(embedding.shape, [1, 512])

  def test_encode_mean_pooling(self):
    enc = whisper_encoder.PooledAudioEncoder(self.model, 'mean')
    timestamp, embedding = enc.encode(self.waveform, self.context)
    npt.assert_equal(timestamp, [[0, 7.5]])
    npt.assert_equal(embedding.shape, [1, 512])

  def test_encode_max_pooling(self):
    enc = whisper_encoder.PooledAudioEncoder(self.model, 'max')
    timestamp, embedding = enc.encode(self.waveform, self.context)
    npt.assert_equal(timestamp, [[0, 7.5]])
    npt.assert_equal(embedding.shape, [1, 512])


if __name__ == '__main__':
  absltest.main()
