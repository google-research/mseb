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

from absl.testing import absltest
from mseb import types
from mseb.encoders import converter as converter_lib
import numpy as np


class SoundToSoundEmbeddingConverterTest(absltest.TestCase):

  def test_eval_sound(self):
    converter = converter_lib.SoundToSoundEmbeddingConverter()
    converter.setup()
    context = types.SoundContextParams(
        sample_rate=2, length=4, id="test", text="transcript truth"
    )
    embedding = converter.encode(
        [types.Sound(waveform=np.array([1.0, 2.0, 3.0, 4.0]), context=context)]
    )[0]
    self.assertIsInstance(embedding, types.SoundEmbedding)
    self.assertEqual(embedding.context, context)
    self.assertEqual(embedding.embedding.shape, (1,))
    self.assertEqual(str(embedding.embedding[0]), "transcript truth")
    self.assertEqual(embedding.timestamps.shape, (1, 2))

  def test_eval_sound_with_title_and_context(self):
    converter = converter_lib.SoundToSoundEmbeddingConverter()
    converter.setup()
    context = types.SoundContextParams(
        sample_rate=2, length=4, id="test", text="transcript truth"
    )
    embedding = converter.encode([
        types.SoundWithTitleAndContext(
            waveform=np.array([1.0, 2.0, 3.0, 4.0]),
            context=context,
            title_text="title",
            context_text="context",
        )
    ])[0]
    self.assertIsInstance(embedding, types.SoundEmbeddingWithTitleAndContext)
    self.assertEqual(embedding.context, context)
    self.assertEqual(embedding.embedding.shape, (1,))
    self.assertEqual(str(embedding.embedding[0]), "transcript truth")
    self.assertEqual(embedding.timestamps.shape, (1, 2))
    self.assertEqual(embedding.title_text, "title")
    self.assertEqual(embedding.context_text, "context")


class SoundEmbeddingToTextConverterTest(absltest.TestCase):

  def test_eval_sound_embedding(self):
    converter = converter_lib.SoundEmbeddingToTextConverter()
    converter.setup()
    context = types.SoundContextParams(sample_rate=2, length=4, id="test")
    text = converter.encode([
        types.SoundEmbedding(
            embedding=np.array(["transcript truth"]),
            timestamps=np.array([[0.0, 2.0]]),
            context=context,
        )
    ])[0]
    self.assertIsInstance(text, types.Text)
    self.assertEqual(text.context.id, "test")
    self.assertEqual(text.text, "transcript truth")

  def test_eval_sound_embedding_with_title_and_context(self):
    converter = converter_lib.SoundEmbeddingToTextConverter()
    converter.setup()
    context = types.SoundContextParams(sample_rate=2, length=4, id="test")
    text = converter.encode([
        types.SoundEmbeddingWithTitleAndContext(
            embedding=np.array(["transcript truth"]),
            timestamps=np.array([[0.0, 2.0]]),
            context=context,
            title_text="title",
            context_text="context",
        )
    ])[0]
    self.assertIsInstance(text, types.TextWithTitleAndContext)
    self.assertEqual(text.context.id, "test")
    self.assertEqual(text.text, "transcript truth")
    self.assertEqual(text.title_text, "title")
    self.assertEqual(text.context_text, "context")


class TextEmbeddingToTextPredictionConverterTest(absltest.TestCase):

  def test_eval_text_embedding(self):
    converter = converter_lib.TextEmbeddingToTextPredictionConverter()
    converter.setup()
    prediction = converter.encode([
        types.TextEmbedding(
            embedding=np.array(["transcript truth"]),
            spans=np.array([[0, 16]]),
            context=types.TextContextParams(id="test"),
        )
    ])[0]
    self.assertIsInstance(prediction, types.TextPrediction)
    self.assertEqual(prediction.prediction, "transcript truth")
    self.assertEqual(prediction.context.id, "test")

  def test_eval_sound_embedding(self):
    converter = converter_lib.TextEmbeddingToTextPredictionConverter()
    converter.setup()
    prediction = converter.encode([
        types.SoundEmbedding(
            embedding=np.array(["transcript truth"]),
            timestamps=np.array([[0.0, 2.0]]),
            context=types.SoundContextParams(
                id="test", sample_rate=2, length=4
            ),
        )
    ])[0]
    self.assertIsInstance(prediction, types.TextPrediction)
    self.assertEqual(prediction.prediction, "transcript truth")
    self.assertEqual(prediction.context.id, "test")


if __name__ == "__main__":
  absltest.main()
