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

"""Base class for retrievers and implementation for exact match retriever."""

from __future__ import annotations

import abc
import logging
from typing import Any, Sequence

import numpy as np


class ExactMatchSearcher(object):
  """Searcher for brute force, exact match retrieval."""

  def __init__(self, all_doc_embeds: np.ndarray):
    self.all_doc_embeds = all_doc_embeds

  def search(
      self, query_embeds: np.ndarray, final_num_neighbors: int
  ) -> tuple[Sequence[Sequence[int]], Sequence[Sequence[float]]]:
    """Runs retrieval for a query.

    Args:
      query_embeds: Query embeddings as a numpy array, (B x D). B is the number
        of queries. D is the embedding dimension.
      final_num_neighbors: Number of neighbors to retrieve.

    Returns:
      A tuple of two sequences. The first sequence contains the raw ids of the
      documents that are closest to the query. The second sequence contains the
      corresponding score for each document.
    """
    # B x Q: (B x D) * (Q x D).T
    scores = query_embeds.dot(self.all_doc_embeds.T)
    top_ids = scores.argsort(axis=1)[:, ::-1][
        :, :final_num_neighbors
    ]  # B x top_ka
    return (
        [[int(x) for x in ids] for ids in top_ids],
        [
            q_score[q_top_ids].tolist()
            for q_score, q_top_ids in zip(scores, top_ids)
        ],
    )  # (B x top_k, B x top_k)


class Retriever(abc.ABC):
  """Base class for retrievers."""

  def __init__(self):
    self.searcher: Any = None
    self.ids: Sequence[str] = None

  @abc.abstractmethod
  def build_index(
      self,
      all_doc_embeds: np.ndarray,
      ids: Sequence[str],
  ) -> None:
    """Builds the index.

    Derived classes should implement this method and set the self.searcher and
    self.ids attributes.

    Args:
      all_doc_embeds: Document embeddings as a numpy array, (Q x D). Q is the
        number of documents. D is the embedding dimension.
      ids: Document ids as a sequence of strings. The order of ids should match
        the order of the documents in all_doc_embeds.
    """
    ...

  def retrieve_docs(
      self,
      query_embeds: np.ndarray,
      document_top_k: int = 100,
  ) -> tuple[Sequence[Sequence[str]], Sequence[Sequence[float]]]:
    """Runs retrieval for a query."""
    index_available = self.searcher is not None and self.ids is not None
    if not index_available:
      raise ValueError('Index is not available.')
    ranked_doc_ids, ranked_doc_scores = self.searcher.search(
        query_embeds, final_num_neighbors=document_top_k
    )
    ranked_doc_scores = [
        [float(x) for x in scores] for scores in ranked_doc_scores
    ]
    return [
        [self.ids[did] for did in dids] for dids in ranked_doc_ids
    ], ranked_doc_scores


class ExactMatchRetriever(Retriever):
  """Retriever for brute force, exact match retrieval."""

  def build_index(
      self,
      all_doc_embeds: np.ndarray,
      ids: Sequence[str],
  ) -> None:
    """Builds the index."""
    self.searcher = ExactMatchSearcher(all_doc_embeds)
    self.ids = ids

    logging.info('\nIndex available!')
