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

"""Evaluator for retrieval tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Union

from mseb import encoder
from mseb import evaluator
import tensorflow_recommenders as tfrs


def compute_reciprocal_rank(
    reference: str, predicted_neighbors: Sequence[str]
) -> float:
  """Computes the reciprocal rank for mean reciprocal rank (MRR)."""
  rank = 0
  for i, neighbor in enumerate(predicted_neighbors):
    if reference == neighbor:
      # Matched the referebce at this position in the ranking.
      rank = i + 1
      break
  mrr = 1 / rank if rank > 0 else 0
  return mrr


class RetrievalEvaluator(evaluator.Evaluator):
  """Evaluator for retrieval tasks."""

  def __init__(
      self,
      sound_encoder: encoder.Encoder,
      encode_kwargs: dict[str, Any],
      searcher: tfrs.layers.factorized_top_k.TopK,
      id_by_index_id: Sequence[str],
  ):
    """Initializes the evaluator with the encoder and retrieval parameters."""
    super().__init__(sound_encoder, encode_kwargs)
    self.searcher = searcher
    self.id_by_index_id = id_by_index_id

  def __call__(
      self,
      sequence: Union[str, Sequence[float]],
      context: encoder.ContextParams,
      reference_id: str = '',
  ) -> dict[str, float]:
    """Evaluates quality of the encoder for input sequence and return metrics.

    Args:
      sequence: Input sound sequence to encode. String-type sequences are
        interpreted as sound file paths.
      context: Encoder input context parameters.
      reference_id: Reference document id.

    Returns:
      Dictionary of metrics, including mean reciprocal rank (MRR) and exact
      match (EM).
    """
    _, query_embeddings = self.sound_encoder.encode(
        sequence=sequence, context=context, **self.encode_kwargs
    )
    _, ranked_index_ids = self.searcher(query_embeddings)
    ranked_doc_ids = [  # pylint: disable=g-complex-comprehension
        [self.id_by_index_id[int(x.numpy())] for x in ids]
        for ids in ranked_index_ids
    ]
    return {
        'reciprocal_rank': compute_reciprocal_rank(
            reference_id, ranked_doc_ids[0]
        ),
        'correct': float(reference_id == ranked_doc_ids[0][0]),
    }

  def combine_scores(self, scores: List[Dict[str, float]]) -> Dict[str, float]:
    """Combines the scores of the examples."""
    return evaluator.compute_weighted_average_and_std(
        scores, (('reciprocal_rank', 'mrr'), ('correct', 'em'))
    )
