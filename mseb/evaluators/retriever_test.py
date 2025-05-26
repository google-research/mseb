# Copyright 2024 The MSEB Authors.
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
from mseb.evaluators import retriever as retriever_lib
import numpy as np
import numpy.testing as npt


class RetrieverTest(absltest.TestCase):

  def test_exact_match_searcher(self):
    searcher = retriever_lib.ExactMatchSearcher(
        all_doc_embeds=np.array(
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]]
        )
    )
    ranked_doc_ids, ranked_doc_scores = searcher.search(
        np.array([[1.0, 2.0, 3.0]]), final_num_neighbors=2
    )
    npt.assert_equal(ranked_doc_ids, [[3, 2]])
    npt.assert_equal(ranked_doc_scores, [[32.0, 26.0]])

  def test_exact_match_retriever(self):
    retriever = retriever_lib.ExactMatchRetriever()
    retriever.build_index(
        all_doc_embeds=np.array(
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]]
        ),
        ids=['bli', 'bla', 'blo', 'blu'],
    )
    ranked_doc_ids, ranked_doc_scores = retriever.retrieve_docs(
        np.array([[1.0, 2.0, 3.0]]), document_top_k=2
    )
    npt.assert_equal(ranked_doc_ids, [['blu', 'blo']])
    npt.assert_equal(ranked_doc_scores, [[32.0, 26.0]])


if __name__ == '__main__':
  absltest.main()
