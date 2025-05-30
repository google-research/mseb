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

from os import path
import pathlib
from typing import Sequence, Tuple, Union

from absl.testing import absltest
from mseb import encoder
from mseb.evaluators import retrieval_evaluator
import numpy as np
import numpy.testing as npt
import tensorflow as tf
import tensorflow_recommenders as tfrs


class IdentityEncoder(encoder.Encoder):

  def encode(
      self,
      sequence: Union[str, Sequence[float]],
      context: encoder.ContextParams,
  ) -> Tuple[np.ndarray, np.ndarray]:
    timestamps = np.array([[0.0, 1.0]])
    embeddings = np.array([sequence], dtype=np.float32)
    return timestamps, embeddings


class RetrievalEvaluatorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.identity_encoder = IdentityEncoder()
    self.context = encoder.ContextParams()
    self.testdata_path = path.join(
        pathlib.Path(path.abspath(__file__)).parent.parent,
        'testdata',
    )

  def test_call(self):
    searcher = tfrs.layers.factorized_top_k.BruteForce(k=2)
    id_by_index_id = ('bli', 'bla', 'blo', 'blu')
    searcher.index(
        candidates=tf.constant(
            [
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [3.0, 4.0, 5.0],
                [4.0, 5.0, 6.0],
            ],
            tf.float32,
        ),
    )
    evaluator = retrieval_evaluator.RetrievalEvaluator(
        sound_encoder=self.identity_encoder,
        encode_kwargs={},
        searcher=searcher,
        id_by_index_id=id_by_index_id,
    )
    scores = evaluator(
        np.array([1.0, 2.0, 3.0]),
        self.context,
        reference_id='blo',
    )
    npt.assert_equal(len(scores), 2)
    npt.assert_equal(scores['reciprocal_rank'], 0.5)
    npt.assert_equal(scores['correct'], 0.0)

  def test_call_with_cache(self):
    id_by_index_id = ('bli', 'bla', 'blo', 'blu')
    cache_path = path.join(self.testdata_path, 'scann_artefacts')
    searcher = tf.saved_model.load(cache_path)
    _ = searcher(tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32))
    evaluator = retrieval_evaluator.RetrievalEvaluator(
        sound_encoder=self.identity_encoder,
        encode_kwargs={},
        searcher=searcher,
        id_by_index_id=id_by_index_id,
    )
    scores = evaluator(
        np.array([1.0, 2.0, 3.0]),
        self.context,
        reference_id='blo',
    )
    npt.assert_equal(len(scores), 2)
    npt.assert_equal(scores['reciprocal_rank'], 0.5)
    npt.assert_equal(scores['correct'], 0.0)

  def test_combine_scores(self):
    evaluator = retrieval_evaluator.RetrievalEvaluator(
        sound_encoder=self.identity_encoder,
        encode_kwargs={},
        searcher=tfrs.layers.factorized_top_k.BruteForce(),  # Not used.
        id_by_index_id=(),
    )
    combined_scores = evaluator.combine_scores([
        {'reciprocal_rank': 1, 'correct': 1},
        {'reciprocal_rank': 1 / 2, 'correct': 0},
    ])
    npt.assert_equal(len(combined_scores), 4)
    npt.assert_equal(combined_scores['mrr'], 3 / 4)
    npt.assert_equal(combined_scores['mrr_std'], 1 / 4)
    npt.assert_equal(combined_scores['em'], 1 / 2)
    npt.assert_equal(combined_scores['em_std'], 1 / 2)


if __name__ == '__main__':
  absltest.main()
