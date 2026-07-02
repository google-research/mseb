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

from absl.testing import absltest
from mseb import types
import numpy as np
import pytest

tf_idf_encoder = pytest.importorskip('mseb.encoders.tf_idf_encoder')


IDF_TABLE = {
    'anatolian': 0.9,
    'shepherds': 0.8,
    'rare': 0.7,
    'national': 3.0,
    'labor': 2.0,
    'relations': 4.0,
    'board': 3.0,
}


@pytest.mark.segmentation
@pytest.mark.optional
class TfIdfEncoderTest(absltest.TestCase):

  def test_encode_from_sound_embedding(self):
    sound_embedding = types.SoundEmbedding(
        embedding=np.array(
            ['How many members does the National Labour Relations Board have?'],
            dtype=object,
        ),
        timestamps=np.array([[0.0, 10.0]]),  # pyrefly: ignore[bad-argument-type]
        context=types.SoundContextParams(
            id='test_utt',
            sample_rate=16000,
            length=32000,
        ),
    )
    encoder: tf_idf_encoder.TermExtractorEncoder = (
        tf_idf_encoder.create_tf_idf_encoder(
            language='en',
            idf_table=IDF_TABLE,
            top_k=100,
            idf_table_path=None,
        )
    )
    encoder.setup()
    tv_embeddings = encoder.encode([sound_embedding])
    self.assertLen(tv_embeddings, 1)
    assert isinstance(tv_embeddings[0], types.SoundEmbedding)
    self.assertEqual(
        dict(tv_embeddings[0].embedding[0]),
        {
            'national': np.float64(3.0),
            'relations': np.float64(4.0),
            'board': np.float64(3.0),
        },
    )

  def test_encode_from_empty_sound_embedding(self):
    sound_embedding = types.SoundEmbedding(
        embedding=np.array(
            list[str](),
            dtype=object,
        ),
        timestamps=np.array([[0.0, 10.0]]),  # pyrefly: ignore[bad-argument-type]
        context=types.SoundContextParams(
            id='test_utt',
            sample_rate=16000,
            length=32000,
        ),
    )
    encoder: tf_idf_encoder.TermExtractorEncoder = (
        tf_idf_encoder.create_tf_idf_encoder(
            language='en',
            idf_table=IDF_TABLE,
            top_k=100,
            idf_table_path=None,
        )
    )
    encoder.setup()
    tv_embeddings = encoder.encode([sound_embedding])
    self.assertLen(tv_embeddings, 1)
    assert isinstance(tv_embeddings[0], types.SoundEmbedding)
    self.assertEmpty(
        dict(tv_embeddings[0].embedding[0]),
    )

  def test_combine_tf_idf_embeddings(self):
    embeddings = [
        types.SoundEmbedding(
            embedding=np.array([
                {'anatolian': 8, 'shepherds': 4}
            ], dtype=object),
            scores=np.array([np.log(0.25)]),  # pyrefly: ignore[bad-argument-type]
            timestamps=np.array([[0.0, 1.0]]),  # pyrefly: ignore[bad-argument-type]
            context=types.SoundContextParams(
                id='test_utt',
                sample_rate=16000,
                length=32000,
            ),
        ),
        types.SoundEmbedding(
            embedding=np.array([
                {'anatolian': 4, 'rare': 4}
            ], dtype=object),
            scores=np.array([np.log(0.75)]),  # pyrefly: ignore[bad-argument-type]
            timestamps=np.array([[0.0, 1.0]]),  # pyrefly: ignore[bad-argument-type]
            context=types.SoundContextParams(
                id='test_utt',
                sample_rate=16000,
                length=32000,
            ),
        ),
    ]
    combined_embedding = tf_idf_encoder.combine_tf_idf_embeddings(
        embeddings,
        types.SoundContextParams(
            id='test_utt',
            sample_rate=16000,
            length=32000,
        ),
    )
    self.assertEqual(
        dict(combined_embedding.embedding[0]),
        {
            'anatolian': np.float64(5.0),
            'shepherds': np.float64(1.0),
            'rare': np.float64(3.0),
        },
    )


if __name__ == '__main__':
  absltest.main()
