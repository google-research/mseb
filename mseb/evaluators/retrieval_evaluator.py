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

from concurrent import futures
import dataclasses
import io
import os
from typing import Any, Dict, Iterable, List, Sequence, Union

from mseb import encoder
from mseb import evaluator
from mseb import types
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs


ThreadPoolExecutor = futures.ThreadPoolExecutor


def mrr(value: float = 0.0, std: float | None = None):
  return types.Score(
      metric='MRR',
      description='Mean Reciprocal Rank',
      value=value,
      min=0,
      max=1,
      std=std,
  )


def em(value: float = 0.0, std: float | None = None):
  return types.Score(
      metric='EM',
      description='Exact Match',
      value=value,
      min=0,
      max=1,
      std=std,
  )


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
  reciprocal_rank = 1 / rank if rank > 0 else 0
  return reciprocal_rank


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
    return self.evaluate_batch(
        sequences=[sequence],
        contexts=[context],
        reference_ids=[reference_id],
    )[0]

  def evaluate_batch(
      self,
      sequences: Sequence[Union[str, Sequence[float]]],
      contexts: Sequence[encoder.ContextParams],
      reference_ids: Sequence[str] = (),
  ) -> Sequence[dict[str, float]]:
    """Evaluates quality of the encoder for input sequences and return metrics.

    Args:
      sequences: Input sound sequences to encode. String-type sequences are
        interpreted as sound file paths.
      contexts: Encoder input context parameters, one per sequence.
      reference_ids: Reference document ids, one per sequence.

    Returns:
      List of dictionaries of metrics, including mean reciprocal rank (MRR) and
      exact
      match (EM).
    """
    timestamps_and_embeddings = self.sound_encoder.encode_batch(
        sequences=sequences, contexts=contexts, **self.encode_kwargs
    )
    metrics_batch = []
    for reference_id, (_, embeddings) in zip(
        reference_ids, timestamps_and_embeddings
    ):
      _, ranked_index_ids = self.searcher(embeddings)
      ranked_doc_ids = [  # pylint: disable=g-complex-comprehension
          [self.id_by_index_id[int(x.numpy())] for x in ids]
          for ids in ranked_index_ids
      ]
      metrics_batch.append({
          'reciprocal_rank': compute_reciprocal_rank(
              reference_id, ranked_doc_ids[0]
          ),
          'correct': float(reference_id == ranked_doc_ids[0][0]),
      })
    return metrics_batch

  def combine_scores(self, scores: List[Dict[str, float]]) -> Dict[str, float]:
    """Combines the scores of the examples."""
    return evaluator.compute_weighted_average_and_std(
        scores, (('reciprocal_rank', 'mrr'), ('correct', 'em'))
    )


@dataclasses.dataclass
class RetrievalReferenceId:
  sound_id: str
  reference_id: str


class RetrievalEvaluatorV2:
  """Evaluator for retrieval tasks."""

  def __init__(
      self,
      searcher: tfrs.layers.factorized_top_k.TopK,
      id_by_index_id: Sequence[str],
  ):
    self.searcher = searcher
    self.id_by_index_id = id_by_index_id

  def __call__(
      self,
      embeddings: types.SoundEmbeddingCache,
      reference_ids: Iterable[RetrievalReferenceId],
  ) -> list[types.Score]:
    """Evaluates quality of the encoder for input sequence and return metrics.

    Args:
      embeddings: The embeddings to evaluate.
      reference_ids: The reference ids used for metric computation.

    Returns:
      A list of Score objects containing the final, aggregated scores. The
      scores include mean reciprocal rank (MRR) and exact match (EM).
    """
    values_by_metric = {'mrr': [], 'em': []}
    for reference_id in reference_ids:
      embedding = embeddings[reference_id.sound_id].embedding
      if embedding.ndim != 2 or embedding.shape[0] != 1:
        raise ValueError(
            'Embedding must be a 2D array of shape (1, embedding_dim),'
            f' but got a {embedding.shape} array.'
        )
      _, ranked_index_ids = self.searcher(embedding.astype(np.float32))
      ranked_doc_ids = [  # pylint: disable=g-complex-comprehension
          [self.id_by_index_id[int(x.numpy())] for x in ids]
          for ids in ranked_index_ids
      ]
      values_by_metric['mrr'].append(
          types.WeightedValue(
              value=compute_reciprocal_rank(
                  reference_id.reference_id, ranked_doc_ids[0]
              )
          )
      )
      values_by_metric['em'].append(
          types.WeightedValue(value=float(reference_id == ranked_doc_ids[0][0]))
      )

    mrr_score = mrr(
        *evaluator.compute_weighted_average_and_std_v2(values_by_metric['mrr'])
    )
    em_score = em(
        *evaluator.compute_weighted_average_and_std_v2(values_by_metric['em'])
    )
    return [mrr_score, em_score]


def build_index(
    embeddings: types.TextEmbeddingCache, k: int = 10
) -> tuple[tfrs.layers.factorized_top_k.TopK, Sequence[str]]:
  """Builds the ScaNN index from the embeddings.

  Args:
    embeddings: The embeddings to build the index from.
    k: The number of neighbors to return.

  Returns:
    A tuple of the searcher of type tfrs.layers.factorized_top_k.TopK and the
    mapping from index id (int) to id (str).
  """
  id_by_index_id: Sequence[str] = sorted(embeddings.keys())

  scann = tfrs.layers.factorized_top_k.ScaNN(
      k=k,
      distance_measure='dot_product',
      num_leaves=min(2000, len(id_by_index_id)),
      num_leaves_to_search=min(100, len(id_by_index_id)),
      training_iterations=1,
      dimensions_per_block=1,
  )
  candidates = tf.constant(
      [embeddings[did].embeddings[0] for did in id_by_index_id], tf.float32
  )
  scann.index(candidates=candidates)
  _ = scann(tf.constant(tf.zeros((1, candidates.shape[1]), dtype=tf.float32)))
  return scann, id_by_index_id


def save_index(
    searcher: tfrs.layers.factorized_top_k.TopK,
    id_by_index_id: Sequence[str],
    scann_base_dir: str,
    id_by_index_id_filepath: str = 'ids.txt',
):
  """Saves the ScaNN index and its metadata to a directory.

  Args:
    searcher: The ScaNN index.
    id_by_index_id: The mapping from index id (int) to id (str).
    scann_base_dir: The base directory for the ScaNN model.
    id_by_index_id_filepath: The filepath for the id by index id mapping
      relative to scann_base_dir.
  """
  os.makedirs(scann_base_dir, exist_ok=True)
  with io.open(os.path.join(scann_base_dir, id_by_index_id_filepath), 'w') as f:
    f.write('\n'.join(id_by_index_id))
  tf.saved_model.save(
      searcher,
      scann_base_dir,
      options=tf.saved_model.SaveOptions(namespace_whitelist=['Scann']),
  )


def load_index(
    scann_base_dir: str,
    id_by_index_id_filepath: str = 'ids.txt',
) -> tuple[tfrs.layers.factorized_top_k.TopK, Sequence[str]]:
  """Loads the ScaNN index and its metadata from a directory.

  Args:
    scann_base_dir: The base directory for the ScaNN model.
    id_by_index_id_filepath: The filepath for the id by index id mapping
      relative to scann_base_dir.

  Returns:
    A tuple of the searcher of type tfrs.layers.factorized_top_k.TopK and the
    mapping from index id (int) to id (str).
  """
  with io.open(os.path.join(scann_base_dir, id_by_index_id_filepath), 'r') as f:
    id_by_index_id: Sequence[str] = f.read().splitlines()
  searcher: tfrs.layers.factorized_top_k.TopK = tf.saved_model.load(
      scann_base_dir
  )
  return searcher, id_by_index_id
