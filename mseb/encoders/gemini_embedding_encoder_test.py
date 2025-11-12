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
import numpy as np
import numpy.testing as npt
import pyarrow.parquet as pq
import pytest


gemini_embedding_encoder = pytest.importorskip(
    'mseb.encoders.gemini_embedding_encoder'
)


@pytest.mark.whisper
@pytest.mark.optional
class GeminiEmbeddingTextEncoderTest(absltest.TestCase):

  def test_gemini_embedding_text_encoder(self):
    enc = gemini_embedding_encoder.GeminiEmbeddingTextEncoder(
        model_path='dummy_model_path'
    )
    enc.prompt_encode_fn = lambda prompts: np.zeros((len(prompts), 3072))
    enc._is_setup = True

    context = types.TextContextParams(id='id', title='This is the title.')
    outputs = enc.encode(
        [
            types.Text(
                text='This is the text.',
                context=context,
            )
        ]
    )[0]
    self.assertIsInstance(outputs, types.TextEmbedding)
    npt.assert_equal(outputs.embedding.shape, (1, 3072))
    npt.assert_equal(outputs.spans.shape, (1, 2))

  def test_gemini_embedding_text_encoder_batch(self):
    enc = gemini_embedding_encoder.GeminiEmbeddingTextEncoder(
        model_path='dummy_model_path',
        normalizer=lambda x: x.lower(),
        prompt_template='title: {title} | text: {text}',
    )
    enc.prompt_encode_fn = lambda prompts: [
        {
            'title: None | text: this is a text.': np.zeros((3072,)),
            'title: this is another title. | text: this is another text.': (
                np.ones((3072,))
            ),
        }[x[0]]
        for x in prompts
    ]
    enc._is_setup = True

    outputs_batch = enc.encode([
        types.TextWithTitleAndContext(
            text='This is a text.',
            context=types.TextContextParams(id='id1'),
        ),
        types.TextWithTitleAndContext(
            text='This is another text.',
            title_text='This is another title.',
            context=types.TextContextParams(id='id2'),
        ),
    ])
    np.testing.assert_equal(len(outputs_batch), 2)
    outputs1 = outputs_batch[0]
    self.assertIsInstance(outputs1, types.TextEmbedding)
    npt.assert_equal(outputs1.embedding, np.zeros((1, 3072)))
    npt.assert_equal(outputs1.spans, np.array([[0, 15]]))
    outputs2 = outputs_batch[1]
    self.assertIsInstance(outputs2, types.TextEmbedding)
    npt.assert_equal(outputs2.embedding, np.ones((1, 3072)))
    npt.assert_equal(outputs2.spans, np.array([[0, 21]]))

  def test_gemini_embedding_text_encoder_prompt_template(self):
    enc1 = gemini_embedding_encoder.GeminiEmbeddingTextEncoder(
        model_path='dummy_model_path',
        normalizer=lambda x: x,
        prompt_template='title: {title} | text: {text}',
    )
    enc1.prompt_encode_fn = lambda prompts: [
        {
            'title: None | text: This is a text.': np.zeros((3072,)) + 0,
            'title: Abc | text: This is another text.': np.zeros((3072,)) + 1,
        }[x[0]]
        for x in prompts
    ]
    enc1._is_setup = True
    outputs_batch1 = enc1.encode([
        types.TextWithTitleAndContext(
            text='This is a text.',
            context=types.TextContextParams(id='id1'),
        ),
        types.TextWithTitleAndContext(
            text='This is another text.',
            title_text='Abc',
            context=types.TextContextParams(id='id2'),
        ),
    ])

    enc2 = gemini_embedding_encoder.GeminiEmbeddingTextEncoder(
        model_path='dummy_model_path', normalizer=None, prompt_template=None
    )
    enc2.prompt_encode_fn = enc1.prompt_encode_fn
    enc2._is_setup = True
    outputs_batch2 = enc2.encode([
        types.Text(
            text='title: None | text: This is a text.',
            context=types.TextContextParams(id='id1'),
        ),
        types.Text(
            text='title: Abc | text: This is another text.',
            context=types.TextContextParams(id='id2', title='Abc'),
        ),
    ])

    npt.assert_equal(len(outputs_batch1), 2)
    npt.assert_equal(len(outputs_batch2), 2)
    for outputs1, outputs2 in zip(outputs_batch1, outputs_batch2):
      self.assertIsInstance(outputs1, types.TextEmbedding)
      self.assertIsInstance(outputs2, types.TextEmbedding)
      npt.assert_equal(outputs1.embedding, outputs2.embedding)


@pytest.mark.whisper
@pytest.mark.optional
class GeminiEmbeddingTranscriptTruthEncoderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, 'testdata'
    )
    self.svq_samples = pq.ParquetFile(
        os.path.join(testdata_path, 'en_us.parquet')
    )

  def test_gemini_embedding_transcript_truth_encoder(self):
    svq_example = self.svq_samples.read_row_group(0)
    waveform = svq_example['waveform'].to_numpy()[0]
    waveform = waveform.astype(np.float32) / 32767.0
    sample_rate = 48000

    enc = gemini_embedding_encoder.GeminiEmbeddingTranscriptTruthEncoder(
        model_path='dummy_model_path'
    )
    enc._encoders[-1].prompt_encode_fn = lambda prompts: np.zeros(
        (len(prompts), 3072)
    )
    enc._encoders[-1]._is_setup = True

    params = types.SoundContextParams(
        id='test',
        length=len(waveform),
        sample_rate=sample_rate,
        language='en',
        text='This is the transcript truth.',
    )
    sound_embeddings = enc.encode([types.Sound(waveform, params)])[0]
    self.assertIsInstance(sound_embeddings, types.TextEmbedding)
    npt.assert_equal(sound_embeddings.spans.shape, [1, 2])
    npt.assert_equal(sound_embeddings.spans[0, 0] >= 0.0, True)
    npt.assert_equal(sound_embeddings.spans[0, 1] <= np.inf, True)
    npt.assert_equal(sound_embeddings.embedding.shape, (1, 3072))

  def test_gemini_embedding_transcript_truth_encoder_batch(self):
    svq_example = self.svq_samples.read_row_group(0)
    waveform = svq_example['waveform'].to_numpy()[0]
    waveform = waveform.astype(np.float32) / 32767.0
    sample_rate = 48000

    enc = gemini_embedding_encoder.GeminiEmbeddingTranscriptTruthEncoder(
        model_path='dummy_model_path'
    )
    enc._encoders[-1].prompt_encode_fn = lambda prompts: np.zeros(
        (len(prompts), 3072)
    )
    enc._encoders[-1]._is_setup = True

    params = types.SoundContextParams(
        id='test',
        length=len(waveform),
        sample_rate=sample_rate,
        language='en',
        text='This is the transcript truth.',
    )
    batch_size = 2
    sound_embeddings_batch = enc.encode(
        [types.Sound(waveform, params)] * batch_size
    )
    np.testing.assert_equal(len(sound_embeddings_batch), batch_size)
    for sound_embeddings in sound_embeddings_batch:
      self.assertIsInstance(sound_embeddings, types.TextEmbedding)
      npt.assert_equal(sound_embeddings.spans.shape, [1, 2])
      npt.assert_equal(sound_embeddings.spans[0, 0] >= 0.0, True)
      npt.assert_equal(sound_embeddings.spans[0, 1] <= np.inf, True)
      npt.assert_equal(sound_embeddings.embedding.shape, (1, 3072))

  def test_genai_api(self):
    if not os.environ.get('GEMINI_API_KEY'):
      return

    enc = gemini_embedding_encoder.GeminiEmbeddingTextEncoder(
        model_path='gemini-embedding-001'
    )
    enc._setup()
    self.assertIsNotNone(enc.prompt_encode_fn)
    outputs_batch = enc.encode([
        types.TextWithTitleAndContext(
            text='This is a text.',
            context=types.TextContextParams(id='id1'),
        ),
        types.TextWithTitleAndContext(
            text='This is another text.',
            title_text='This is another title.',
            context=types.TextContextParams(id='id2'),
        ),
    ])
    np.testing.assert_equal(len(outputs_batch), 2)
    outputs1 = outputs_batch[0]
    self.assertIsInstance(outputs1, types.TextEmbedding)
    npt.assert_equal(outputs1.embedding.shape, (1, 3072))
    npt.assert_equal(outputs1.spans, np.array([[0, 15]]))
    outputs2 = outputs_batch[1]
    self.assertIsInstance(outputs2, types.TextEmbedding)
    npt.assert_equal(outputs2.embedding.shape, (1, 3072))
    npt.assert_equal(outputs2.spans, np.array([[0, 21]]))


if __name__ == '__main__':
  absltest.main()
