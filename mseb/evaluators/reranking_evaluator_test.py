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
from mseb.evaluators import reranking_evaluator
import numpy as np
import numpy.testing as npt


class RerankingEvaluatorTest(absltest.TestCase):

  def test_compute_predictions(self):
    evaluator = reranking_evaluator.RerankingEvaluator(
        candidate_embeddings_by_sound_id={
            'test': [
                types.TextEmbedding(
                    embedding=np.array([[3.0, 4.0]], dtype=np.float32),
                    spans=np.array([[0, -1]]),
                    context=types.TextContextParams(id='b l i'),
                ),
                types.TextEmbedding(
                    embedding=np.array([[5.0, 6.0]], dtype=np.float32),
                    spans=np.array([[0, -1]]),
                    context=types.TextContextParams(id='b l a'),
                ),
                types.TextEmbedding(
                    embedding=np.array([[1.0, 2.0]], dtype=np.float32),
                    spans=np.array([[0, -1]]),
                    context=types.TextContextParams(id='x y z'),
                ),
            ]
        },
    )
    predictions = evaluator.compute_predictions(
        embeddings_by_sound_id={
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
    )
    self.assertLen(predictions, 1)
    self.assertIn('test', predictions)
    npt.assert_equal(predictions['test'][0], [30.5, 19.5, 8.5])
    npt.assert_equal(predictions['test'][1], ['b l a', 'b l i', 'x y z'])

  def test_compute_metrics(self):
    evaluator = reranking_evaluator.RerankingEvaluator(
        candidate_embeddings_by_sound_id={}, mrr_at_k=2
    )
    scores = evaluator.compute_metrics(
        predictions={'test': ([1.0, 0.5, 0.0], ['b l a', 'b l i', 'x y z'])},
        candidates_batch=[
            reranking_evaluator.RerankingCandidates(
                sound_id='test',
                texts=['b l i', 'b l a', 'x y z'],
                language='en',
            ),
        ],
    )
    npt.assert_equal(len(scores), 4)
    self.assertIn('MAP', scores[0].metric)
    npt.assert_equal(scores[0].value, 1 / 2)
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
