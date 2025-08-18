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
from typing import Sequence
from unittest import mock

from absl.testing import absltest
from mseb import encoder
from mseb.encoders import gecko_transcript_truth_encoder
import numpy as np
import numpy.testing as npt
import pyarrow.parquet as pq
import tensorflow as tf


GeckoTranscriptTruthEncoder = (
    gecko_transcript_truth_encoder.GeckoTranscriptTruthEncoder
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


if __name__ == '__main__':
  absltest.main()
