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
from typing import Callable, Sequence
from unittest import mock

from absl.testing import absltest
from mseb import encoder
from mseb import types
from mseb.encoders import gecko_transcript_truth_encoder
import numpy as np
import numpy.testing as npt
import pyarrow.parquet as pq
import tensorflow as tf


GeckoTranscriptTruthEncoder = (
    gecko_transcript_truth_encoder.GeckoTranscriptTruthEncoder
)
GeckoTranscriptTruthEncoderV2 = (
    gecko_transcript_truth_encoder.GeckoTranscriptTruthEncoderV2
)


def create_gecko_model_mock(
    embedding_by_text: dict[str, Sequence[float]],
) -> mock.Mock:
  """Creates a mock for Gecko model."""
  mock_model = mock.create_autospec(tf.keras.Model, instance=True)
  mock_model.signatures = {
      'serving_default': lambda x: {
          'encodings': tf.constant([embedding_by_text[x[0].numpy().decode()]])
      }
  }
  return mock_model


class GeckoTranscriptTruthEncoderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, 'testdata'
    )
    self.svq_samples = pq.ParquetFile(
        os.path.join(testdata_path, 'en_us.parquet')
    )

  def test_gecko_transcript_truth_encoder(self):
    svq_example = self.svq_samples.read_row_group(0)
    waveform = svq_example['waveform'].to_numpy()[0]
    waveform = waveform.astype(np.float32) / 32767.0
    sample_rate = 48000

    enc = GeckoTranscriptTruthEncoder(
        gecko_model=create_gecko_model_mock(
            embedding_by_text={
                'task: search result | query: This is the transcript truth.': [
                    1.0,
                    2.0,
                ]
            }
        ),
    )

    context = encoder.ContextParams(
        language='en',
        sample_rate=sample_rate,
        text='This is the transcript truth.',
        audio_start_seconds=0.0,
        audio_end_seconds=waveform.shape[0] / sample_rate,
    )
    timestamps, embeddings = enc.encode(waveform, context)
    npt.assert_equal(timestamps.shape, [1, 2])
    npt.assert_equal(timestamps[0, 0] == 0.0, True)
    npt.assert_equal(timestamps[0, 1] == waveform.shape[0] / sample_rate, True)
    npt.assert_equal(embeddings, [[1.0, 2.0]])


class MockGeckoTranscriptTruthEncoderV2(GeckoTranscriptTruthEncoderV2):
  """Mock transcript truth encoder with Gecko model for testing."""

  def __init__(
      self,
      transcript_truths_encode_fn: Callable[[Sequence[str]], np.ndarray],
  ):
    super().__init__('dummy_model_path')
    self.transcript_truths_encode_fn = transcript_truths_encode_fn

  def setup(self):
    """Mock setup method."""
    assert self.transcript_truths_encode_fn is not None
    self._model_loaded = True


class GeckoTranscriptTruthEncoderV2Test(absltest.TestCase):

  def setUp(self):
    super().setUp()
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, 'testdata'
    )
    self.svq_samples = pq.ParquetFile(
        os.path.join(testdata_path, 'en_us.parquet')
    )

    sample_rate = 48000
    svq_example = self.svq_samples.read_row_group(0)
    self.waveform1 = svq_example['waveform'].to_numpy()[0][:100]
    self.waveform1 = self.waveform1.astype(np.float32) / 32767.0
    self.params1 = types.SoundContextParams(
        sample_rate=sample_rate,
        length=self.waveform1.shape[0],
        language='en',
        text='This is the transcript truth.',
        waveform_start_second=0.0,
        waveform_end_second=self.waveform1.shape[0] / sample_rate,
        sound_id='test1',
    )
    self.waveform2 = svq_example['waveform'].to_numpy()[0][100:]
    self.waveform2 = self.waveform2.astype(np.float32) / 32767.0
    self.params2 = types.SoundContextParams(
        sample_rate=sample_rate,
        length=self.waveform2.shape[0],
        language='en',
        text='This is another transcript truth.',
        waveform_start_second=0.0,
        waveform_end_second=self.waveform2.shape[0] / sample_rate,
        sound_id='test2',
    )

    def transcript_truths_encode_fn(prompts: Sequence[str]) -> np.ndarray:
      embedding_by_prompt = {
          'task: search result | query: This is the transcript truth.': [
              1,
              2,
          ],
          'task: search result | query: This is another transcript truth.': [
              2,
              1,
          ],
      }
      return np.array(
          [embedding_by_prompt[prompt] for prompt in prompts], np.float32
      )

    self.transcript_truths_encode_fn = transcript_truths_encode_fn

  def test_gecko_transcript_truth_encoder_with_mock_model_encode(self):
    enc = MockGeckoTranscriptTruthEncoderV2(
        transcript_truths_encode_fn=self.transcript_truths_encode_fn
    )
    result = enc.encode(self.waveform1, self.params1)
    npt.assert_equal(result.timestamps.shape, [1, 2])
    npt.assert_equal(result.timestamps[0, 0] == 0.0, True)
    npt.assert_equal(
        result.timestamps[0, 1]
        == self.waveform1.shape[0] / self.params1.sample_rate,
        True,
    )
    npt.assert_equal(result.embedding, [[1.0, 2.0]])

  def test_gecko_transcript_truth_encoder_with_mock_model_encode_batch(self):
    enc = MockGeckoTranscriptTruthEncoderV2(
        transcript_truths_encode_fn=self.transcript_truths_encode_fn
    )
    result1 = enc.encode(self.waveform1, self.params1)
    result2 = enc.encode(self.waveform2, self.params2)
    results_batch = enc.encode_batch(
        [self.waveform1, self.waveform2], [self.params1, self.params2]
    )
    npt.assert_equal(len(results_batch), 2)
    self.assertEqual(
        results_batch[0].embedding.tolist(), result1.embedding.tolist()
    )
    self.assertEqual(
        results_batch[0].timestamps.tolist(), result1.timestamps.tolist()
    )
    self.assertEqual(
        results_batch[1].embedding.tolist(), result2.embedding.tolist()
    )
    self.assertEqual(
        results_batch[1].timestamps.tolist(), result2.timestamps.tolist()
    )


if __name__ == '__main__':
  absltest.main()
