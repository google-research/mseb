# Copyright 2026 The MSEB Authors.
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

"""Tests for the LiteLLM API-based encoder."""

import os
import pathlib
from unittest import mock

from absl.testing import absltest
from litellm.types.utils import Embedding
from litellm.types.utils import EmbeddingResponse
from mseb import types
from mseb import utils
import pytest

encoder_lib = pytest.importorskip('mseb.encoders.litellm_embedding_encoder')


class LiteLLMEmbeddingEncoderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, 'testdata'
    )
    with open(os.path.join(testdata_path, 'roses-are.wav'), 'rb') as f:
      wav_bytes = f.read()
      samples, sample_rate = utils.wav_bytes_to_waveform(wav_bytes)
    self.test_sound = types.Sound(
        waveform=samples,
        context=types.SoundContextParams(
            id='test',
            sample_rate=sample_rate,
            length=len(samples),
            waveform_start_second=0.0,
            waveform_end_second=len(samples) / sample_rate,
        ),
    )
    self.mock_response = EmbeddingResponse(
        data=[
            Embedding(
                embedding=[0.1, 0.2, 0.3],
                index=0,
                object='embedding',
            ),
            Embedding(
                embedding=[0.4, 0.5, 0.6],
                index=1,
                object='embedding',
            ),
        ],
        model='test_model',
        object='list',
    )

  @mock.patch('litellm.embedding')
  def test_encode(self, mock_transcription):
    mock_transcription.return_value = self.mock_response
    embedding_encoder = encoder_lib.LiteLLMEmbeddingEncoder(
        model_name='test_model',
        api_key='test_api_key',
        embedding_dim=3,
    )
    embedding_encoder.setup()
    outputs = embedding_encoder.encode([self.test_sound] * 2)
    self.assertLen(outputs, 2)
    output0 = outputs[0]
    output1 = outputs[1]
    assert isinstance(output0, types.SoundEmbedding)
    assert isinstance(output1, types.SoundEmbedding)
    self.assertEqual(output0.embedding.shape, (1, 3))
    self.assertEqual(output1.embedding.shape, (1, 3))
    self.assertEqual(output0.embedding.tolist(), [[0.1, 0.2, 0.3]])
    self.assertEqual(output1.embedding.tolist(), [[0.4, 0.5, 0.6]])
    self.assertEqual(output0.timestamps.shape, (1, 2))
    self.assertEqual(output0.timestamps[0, 0], 0.0)
    self.assertEqual(output0.timestamps[0, 1], 3.0763125)
    self.assertEqual(output1.timestamps.shape, (1, 2))
    self.assertEqual(output1.timestamps[0, 0], 0.0)
    self.assertEqual(output1.timestamps[0, 1], 3.0763125)


if __name__ == '__main__':
  absltest.main()
