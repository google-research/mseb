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


from unittest import mock

from absl.testing import absltest
from mseb import types
from mseb.encoders import mapper_reducer
import numpy as np


class MapperReducerTest(absltest.TestCase):

  def test_mapper_setup(self):
    mock_encoder = mock.MagicMock()
    mapper = mapper_reducer.SoundEmbeddingCollectionMapper(mock_encoder)
    mapper.setup()
    mock_encoder.setup.assert_called_once()

  def test_mapper_encode(self):
    mock_encoder = mock.MagicMock()
    # Mock encode to return a list with one element
    # Mapper calls self._encoder.encode([v])[0]
    mock_encoder.encode.side_effect = lambda x: [
        types.SoundEmbedding(
            embedding=x[0].embedding * 2,
            timestamps=x[0].timestamps,
            context=x[0].context,
        )
    ]

    mapper = mapper_reducer.SoundEmbeddingCollectionMapper(mock_encoder)

    context = types.SoundContextParams(id="test", sample_rate=16000, length=10)
    embedding1 = types.SoundEmbedding(
        embedding=np.array([[1.0]]),
        timestamps=np.array([[0.0, 1.0]]),  # pyrefly: ignore[bad-argument-type]
        context=context,
    )
    embedding2 = types.SoundEmbedding(
        embedding=np.array([[2.0]]),
        timestamps=np.array([[1.0, 2.0]]),  # pyrefly: ignore[bad-argument-type]
        context=context,
    )

    collection = types.SoundEmbeddingCollection(
        embeddings={"head1": embedding1, "head2": embedding2},
        context=context,
    )

    outputs = mapper.encode([collection])

    self.assertLen(outputs, 1)
    output_collection = outputs[0]
    self.assertIsInstance(output_collection, types.SoundEmbeddingCollection)
    self.assertEqual(output_collection.context, context)

    # Check that embeddings were processed
    self.assertIn("head1", output_collection.embeddings)
    self.assertIn("head2", output_collection.embeddings)

    head1_emb = output_collection.embeddings["head1"]
    assert isinstance(head1_emb, types.SoundEmbedding)
    np.testing.assert_array_equal(head1_emb.embedding, np.array([[2.0]]))
    head2_emb = output_collection.embeddings["head2"]
    assert isinstance(head2_emb, types.SoundEmbedding)
    np.testing.assert_array_equal(head2_emb.embedding, np.array([[4.0]]))

  def test_mapper_invalid_input(self):
    mock_encoder = mock.MagicMock()
    mapper = mapper_reducer.SoundEmbeddingCollectionMapper(mock_encoder)

    invalid_input = [
        types.Text(text="hello", context=types.TextContextParams(id="1"))
    ]

    with self.assertRaises(ValueError):
      mapper.encode(invalid_input)

  def test_reducer_encode(self):
    def combine_fn(embeddings, context):
      # Just sum them up for testing
      total_emb = sum(e.embedding for e in embeddings)
      return types.SoundEmbedding(
          embedding=total_emb,  # pyrefly: ignore[bad-argument-type]
          timestamps=embeddings[0].timestamps,
          context=context,
      )

    reducer = mapper_reducer.SoundEmbeddingCollectionReducer(combine_fn)

    context = types.SoundContextParams(id="test", sample_rate=16000, length=10)
    embedding1 = types.SoundEmbedding(
        embedding=np.array([[1.0]]),
        timestamps=np.array([[0.0, 1.0]]),  # pyrefly: ignore[bad-argument-type]
        context=context,
    )
    embedding2 = types.SoundEmbedding(
        embedding=np.array([[2.0]]),
        timestamps=np.array([[0.0, 1.0]]),  # pyrefly: ignore[bad-argument-type]
        context=context,
    )

    collection = types.SoundEmbeddingCollection(
        embeddings={"head1": embedding1, "head2": embedding2},
        context=context,
    )

    outputs = reducer.encode([collection])

    self.assertLen(outputs, 1)
    output_embedding = outputs[0]
    self.assertIsInstance(output_embedding, types.SoundEmbedding)
    self.assertEqual(output_embedding.context, context)

    np.testing.assert_array_equal(output_embedding.embedding, np.array([[3.0]]))

  def test_reducer_invalid_input(self):
    reducer = mapper_reducer.SoundEmbeddingCollectionReducer(
        combine_fn=lambda x, y: None  # pyrefly: ignore[bad-argument-type]
    )

    invalid_input = [
        types.Text(text="hello", context=types.TextContextParams(id="1"))
    ]

    with self.assertRaises(ValueError):
      reducer.encode(invalid_input)


if __name__ == "__main__":
  absltest.main()
