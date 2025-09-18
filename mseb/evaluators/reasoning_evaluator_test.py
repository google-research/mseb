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

"""Tests for ReasoningEvaluator class."""

from absl.testing import absltest
from mseb import types
from mseb.evaluators import reasoning_evaluator
import numpy as np
import numpy.testing as npt


class ReasoningEvaluatorTest(absltest.TestCase):

  def test_compute_predictions_float_embedding(self):
    evaluator = reasoning_evaluator.ReasoningEvaluator(
        span_embeddings_by_text={
            'b l i': types.TextEmbeddings(
                embeddings=np.array([[3.0, 4.0]], dtype=np.float32),
                spans=np.array([[0, -1]]),
                context=types.TextContextParams(id='b l i'),
            ),
            'b l a': types.TextEmbeddings(
                embeddings=np.array([[5.0, 6.0]], dtype=np.float32),
                spans=np.array([[0, -1]]),
                context=types.TextContextParams(id='b l a'),
            ),
            'x y z': types.TextEmbeddings(
                embeddings=np.array([[1.0, 2.0]], dtype=np.float32),
                spans=np.array([[0, -1]]),
                context=types.TextContextParams(id='x y z'),
            ),
        },
        no_answer_threshold=0.5,
    )
    predictions = evaluator.compute_predictions(
        embeddings={
            'test': types.SoundEmbedding(
                embedding=np.array([[2.5, 3.0]]),
                timestamps=np.array([[0.0, 1.0]]),
                context=types.SoundContextParams(
                    id='test',
                    sample_rate=16000,
                    length=100,
                    language='en',
                ),
            ),
        },
        spans_batch=[
            reasoning_evaluator.ReasoningSpans(
                sound_id='test',
                texts=['b l i', 'b l a', 'x y z'],
                reference_answer='b l i',
            ),
        ],
    )
    self.assertLen(predictions, 1)
    self.assertIn('test', predictions)
    self.assertEqual(predictions['test'], 'b l a')

  def test_compute_predictions_string_embedding(self):
    evaluator = reasoning_evaluator.ReasoningEvaluator(
        span_embeddings_by_text={
            'b l i': types.TextEmbeddings(
                embeddings=np.array([[3.0, 4.0]], dtype=np.float32),
                spans=np.array([[0, -1]]),
                context=types.TextContextParams(id='b l i'),
            ),
            'b l a': types.TextEmbeddings(
                embeddings=np.array([[5.0, 6.0]], dtype=np.float32),
                spans=np.array([[0, -1]]),
                context=types.TextContextParams(id='b l a'),
            ),
            'x y z': types.TextEmbeddings(
                embeddings=np.array([[1.0, 2.0]], dtype=np.float32),
                spans=np.array([[0, -1]]),
                context=types.TextContextParams(id='x y z'),
            ),
        },
        no_answer_threshold=0.5,
    )
    predictions = evaluator.compute_predictions(
        embeddings={
            'test': types.SoundEmbedding(
                embedding=np.array(['b l a']),
                timestamps=np.array([[0.0, 1.0]]),
                context=types.SoundContextParams(
                    id='test',
                    sample_rate=16000,
                    length=100,
                    language='en',
                ),
            ),
        },
        spans_batch=[
            reasoning_evaluator.ReasoningSpans(
                sound_id='test',
                texts=['b l i', 'b l a', 'x y z'],
                reference_answer='b l i',
            ),
        ],
    )
    self.assertLen(predictions, 1)
    self.assertIn('test', predictions)
    self.assertEqual(predictions['test'], 'b l a')

  def test_evaluate_predictions(self):
    evaluator = reasoning_evaluator.ReasoningEvaluator(
        span_embeddings_by_text={
            'b l i': types.TextEmbeddings(
                embeddings=np.array([[3.0, 4.0]], dtype=np.float32),
                spans=np.array([[0, -1]]),
                context=types.TextContextParams(id='b l i'),
            ),
            'b l a': types.TextEmbeddings(
                embeddings=np.array([[5.0, 6.0]], dtype=np.float32),
                spans=np.array([[0, -1]]),
                context=types.TextContextParams(id='b l a'),
            ),
            'x y z': types.TextEmbeddings(
                embeddings=np.array([[1.0, 2.0]], dtype=np.float32),
                spans=np.array([[0, -1]]),
                context=types.TextContextParams(id='x y z'),
            ),
        },
        no_answer_threshold=0.5,
    )
    scores = evaluator.evaluate_predictions(
        predictions={'test': 'b l a'},
        spans_batch=[
            reasoning_evaluator.ReasoningSpans(
                sound_id='test',
                texts=['b l i', 'b l a', 'x y z'],
                reference_answer='b l i',
            ),
        ],
    )
    npt.assert_equal(len(scores), 1)
    self.assertIn('F1', scores[0].metric)
    npt.assert_equal(scores[0].value, 2 / 3)
    npt.assert_equal(scores[0].std, 0)

  def test_call(self):
    evaluator = reasoning_evaluator.ReasoningEvaluator(
        span_embeddings_by_text={
            'b l i': types.TextEmbeddings(
                embeddings=np.array([[3.0, 4.0]], dtype=np.float32),
                spans=np.array([[0, -1]]),
                context=types.TextContextParams(id='b l i'),
            ),
            'b l a': types.TextEmbeddings(
                embeddings=np.array([[5.0, 6.0]], dtype=np.float32),
                spans=np.array([[0, -1]]),
                context=types.TextContextParams(id='b l a'),
            ),
            'x y z': types.TextEmbeddings(
                embeddings=np.array([[1.0, 2.0]], dtype=np.float32),
                spans=np.array([[0, -1]]),
                context=types.TextContextParams(id='x y z'),
            ),
        },
        no_answer_threshold=0.5,
    )
    scores = evaluator(
        embeddings={
            'test': types.SoundEmbedding(
                embedding=np.array([[2.5, 3.0]]),
                timestamps=np.array([[0.0, 1.0]]),
                context=types.SoundContextParams(
                    id='test',
                    sample_rate=16000,
                    length=100,
                    language='en',
                ),
            ),
        },
        spans_batch=[
            reasoning_evaluator.ReasoningSpans(
                sound_id='test',
                texts=['b l i', 'b l a', 'x y z'],
                reference_answer='b l i',
            ),
        ],
    )
    npt.assert_equal(len(scores), 1)
    self.assertIn('F1', scores[0].metric)
    npt.assert_equal(scores[0].value, 2 / 3)
    npt.assert_equal(scores[0].std, 0)


if __name__ == '__main__':
  absltest.main()
