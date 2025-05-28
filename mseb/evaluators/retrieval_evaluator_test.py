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
from mseb.evaluators import retrieval_evaluator
from mseb.evaluators import retriever as retriever_lib
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


class RetrievalEvaluatorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.identity_encoder = IdentityEncoder()
    self.context = encoder.ContextParams()

  def test_call(self):
    retriever = retriever_lib.ExactMatchRetriever()
    retriever.build_index(
        all_doc_embeds=np.array(
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]]
        ),
        ids=['bli', 'bla', 'blo', 'blu'],
    )
    evaluator = retrieval_evaluator.RetrievalEvaluator(
        sound_encoder=self.identity_encoder,
        encode_kwargs={},
        retriever=retriever,
    )
    scores = evaluator(
        np.array([1.0, 2.0, 3.0]),
        self.context,
        reference_id='blo',
        document_top_k=2,
    )
    npt.assert_equal(len(scores), 2)
    npt.assert_equal(scores['reciprocal_rank'], 0.5)
    npt.assert_equal(scores['correct'], 0.0)

  def test_combine_scores(self):
    evaluator = retrieval_evaluator.RetrievalEvaluator(
        sound_encoder=self.identity_encoder,
        encode_kwargs={},
        retriever=retriever_lib.ExactMatchRetriever(),  # Not used.
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
