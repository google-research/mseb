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

from typing import Sequence, Tuple, Union

from absl.testing import absltest
from mseb import encoder
from mseb import types
from mseb.evaluators import reranking_evaluator
import numpy as np
import numpy.testing as npt


class IdentityEncoder(encoder.Encoder):

  def encode(
      self,
      sequence: Union[str, Sequence[float]],
      context: encoder.ContextParams,
      index: int = 0,
  ) -> Tuple[np.ndarray, np.ndarray]:
    timestamps = np.array([[0.0, 1.0]])
    embeddings = np.array([sequence], dtype=np.float32)
    return timestamps, embeddings


class RerankingEvaluatorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.identity_encoder = IdentityEncoder()
    self.context = encoder.ContextParams(language='en')

  def test_call(self):
    evaluator = reranking_evaluator.RerankingEvaluator(
        sound_encoder=self.identity_encoder,
        encode_kwargs={},
    )
    scores = evaluator(
        np.array([2.5, 3.0]),
        self.context,
        candidate_texts=('b l i', 'b l a', 'x y z'),
        candidate_embeddings=np.array([
            [3.0, 4.0],
            [5.0, 6.0],
            [1.0, 2.0],
        ]),
        document_top_k=2,
    )
    npt.assert_equal(len(scores), 4)
    npt.assert_equal(scores['reciprocal_rank'], 1 / 2)
    npt.assert_equal(scores['word_errors'], 1)
    npt.assert_equal(scores['word_errors_weight'], 3)
    npt.assert_equal(scores['correct'], 1)

  def test_combine_scores(self):
    evaluator = reranking_evaluator.RerankingEvaluator(
        sound_encoder=self.identity_encoder,
        encode_kwargs={},
    )
    combined_scores = evaluator.combine_scores([
        {
            'reciprocal_rank': 1,
            'correct': 1,
            'word_errors': 2,
            'word_errors_weight': 3,
        },
        {
            'reciprocal_rank': 1 / 2,
            'correct': 0,
            'word_errors': 1,
            'word_errors_weight': 2,
        },
    ])
    npt.assert_equal(len(combined_scores), 6)
    npt.assert_equal(combined_scores['mrr'], 3 / 4)
    npt.assert_equal(combined_scores['mrr_std'], 1 / 4)
    npt.assert_equal(combined_scores['qer'], 1 / 2)
    npt.assert_equal(combined_scores['qer_std'], 1 / 2)
    npt.assert_equal(combined_scores['wer'], 8 / 5)
    npt.assert_equal(combined_scores['wer_std'], (6 / 25) ** 0.5)


class RerankingEvaluatorV2Test(absltest.TestCase):

  def test_evaluate_predictions(self):
    evaluator = reranking_evaluator.RerankingEvaluatorV2(
        candidate_embeddings_by_text={
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
    )
    scores = evaluator.evaluate_predictions(
        predictions={
            'test': (['b l a', 'b l i', 'x y z'], [1.0, 0.5, 0.0])
        },
        candidates_batch=[
            reranking_evaluator.RerankingCandidates(
                sound_id='test',
                texts=['b l i', 'b l a', 'x y z'],
            ),
        ],
        is_english=True,
    )
    npt.assert_equal(len(scores), 4)
    self.assertIn('MAP', scores[0].metric)
    npt.assert_equal(scores[0].value, 1)
    npt.assert_equal(scores[0].std, 0)
    self.assertIn('WER', scores[1].metric)
    npt.assert_equal(scores[1].value, 1 / 3)
    npt.assert_equal(scores[1].std, 0)
    self.assertIn('CER', scores[2].metric)
    npt.assert_equal(scores[2].value, 1)
    npt.assert_equal(scores[2].std, 0)
    self.assertIn('MRR', scores[3].metric)
    npt.assert_equal(scores[3].value, 1 / 2)
    npt.assert_equal(scores[3].std, 0)

  def test_call(self):
    evaluator = reranking_evaluator.RerankingEvaluatorV2(
        candidate_embeddings_by_text={
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
        mrr_at_k=2,
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
        candidates_batch=[
            reranking_evaluator.RerankingCandidates(
                sound_id='test',
                texts=['b l i', 'b l a', 'x y z'],
            ),
        ],
    )
    npt.assert_equal(len(scores), 4)
    self.assertIn('MAP', scores[0].metric)
    npt.assert_equal(scores[0].value, 1)
    npt.assert_equal(scores[0].std, 0)
    self.assertIn('WER', scores[1].metric)
    npt.assert_equal(scores[1].value, 1 / 3)
    npt.assert_equal(scores[1].std, 0)
    self.assertIn('CER', scores[2].metric)
    npt.assert_equal(scores[2].value, 1)
    npt.assert_equal(scores[2].std, 0)
    self.assertIn('MRR', scores[3].metric)
    npt.assert_equal(scores[3].value, 1 / 2)
    npt.assert_equal(scores[3].std, 0)


if __name__ == '__main__':
  absltest.main()
