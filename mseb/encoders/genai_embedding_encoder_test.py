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

"""Tests for the GenaiEmbeddingEncoder class."""

from unittest import mock
from absl.testing import absltest
from mseb import types
from mseb.encoders import genai_embedding_encoder as genai_embedding_encoder_lib
from mseb.encoders import prompt as prompt_lib
import numpy as np


class GenaiEmbeddingEncoderTest(absltest.TestCase):

  @mock.patch('mseb.encoders.genai_embedding_encoder.genai.Client')
  def test_encode_text_only(self, mock_client):
    mock_client.return_value = mock.Mock()
    mock_client.return_value.models.embed_content.return_value = mock.Mock()
    mock_client.return_value.models.embed_content.return_value.embeddings = [
        mock.Mock(),
        mock.Mock(),
    ]
    mock_client.return_value.models.embed_content.return_value.embeddings[
        0
    ].values = np.random.rand(
        768,
    ).tolist()
    mock_client.return_value.models.embed_content.return_value.embeddings[
        1
    ].values = np.random.rand(
        768,
    ).tolist()

    genai_encoder = genai_embedding_encoder_lib.GenaiEmbeddingEncoder(
        model_path='gemini-embedding-2-preview',
        api_key='mock_api_key',
        prompt=prompt_lib.DefaultPrompt('search result: {text}'),
    )
    genai_encoder.setup()
    outputs = genai_encoder.encode(
        [
            types.TextWithTitleAndContext(
                text='What is the capital of France?',
                title_text='France',
                context_text='Paris is the capital of France.',
                context=types.TextContextParams(id='1'),
            ),
            types.TextWithTitleAndContext(
                text='How tall is Michael Jordan?',
                title_text='Michael Jordan',
                context_text='Michael Jordan is 6 feet 9 inches tall.',
                context=types.TextContextParams(id='2'),
            ),
        ],
    )

    self.assertLen(outputs, 2)
    self.assertEqual(outputs[0].context.id, '1')
    self.assertEqual(outputs[1].context.id, '2')
    for output in outputs:
      self.assertIsInstance(output, types.TextEmbedding)
      self.assertEqual(output.embedding.shape, (1, 768))
      self.assertEqual(output.embedding.dtype, np.float32)

  @mock.patch('mseb.encoders.genai_embedding_encoder.genai.Client')
  def test_encode_text_and_audio(self, mock_client):
    mock_client.return_value = mock.Mock()
    mock_client.return_value.models.embed_content.return_value = mock.Mock()
    mock_client.return_value.models.embed_content.return_value.embeddings = [
        mock.Mock(),
    ]
    mock_client.return_value.models.embed_content.return_value.embeddings[
        0
    ].values = np.random.rand(
        768,
    ).tolist()

    genai_encoder = genai_embedding_encoder_lib.GenaiEmbeddingEncoder(
        model_path='gemini-embedding-2-preview',
        api_key='mock_api_key',
        prompt=prompt_lib.DefaultPrompt('search result: {text}'),
    )
    genai_encoder.setup()
    outputs = genai_encoder.encode(
        [
            types.Sound(
                waveform=np.random.rand(
                    16000,
                ),
                context=types.SoundContextParams(
                    id='2', sample_rate=16000, length=16000
                ),
            ),
        ],
    )

    self.assertLen(outputs, 1)
    self.assertEqual(outputs[0].context.id, '2')
    output = outputs[0]
    self.assertIsInstance(output, types.SoundEmbedding)
    self.assertEqual(output.embedding.shape, (1, 768))
    self.assertEqual(output.embedding.dtype, np.float32)


if __name__ == '__main__':
  absltest.main()
