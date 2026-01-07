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

from collections.abc import Sequence
import dataclasses
import itertools
import logging
import os
from typing import Any, Mapping

from etils import epath
import jaxtyping
from mseb import encoder as encoder_lib
from mseb import evaluator as evaluator_lib
from mseb import metrics as metrics_lib
from mseb import types
import numpy as np

from scann.scann_ops.py import scann_ops_pybind
ScannSearcher = scann_ops_pybind.ScannSearcher

logger = logging.getLogger(__name__)

NO_RESPONSE_STR = encoder_lib.NO_RESPONSE_STR
INVALID_ANSWER_STR = encoder_lib.INVALID_ANSWER_STR


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


@dataclasses.dataclass
class RetrievalReferenceId:
  sound_id: str
  reference_id: str


RetrievalPredictionsCache = Mapping[str, Sequence[tuple[float, str]] | str]


def as_retrieval_prediction(prediction: Any) -> Sequence[tuple[float, str]]:
  """Checks and returns a prediction to a sequence of tuples of score and doc id."""
  assert isinstance(prediction, Sequence)
  assert all(isinstance(x, tuple) and len(x) == 2 for x in prediction)
  assert all(isinstance(x[0], float) for x in prediction)
  assert all(isinstance(x[1], str) for x in prediction)
  return prediction


def get_ranked_doc_ids(
    score_and_doc_ids: Sequence[tuple[float, str]], top_k: int
) -> Sequence[str]:
  """Sorts and deduplicates predictions and returns the topk ranked doc ids.

  Args:
    score_and_doc_ids: A sequence of tuples, where each tuple contains a
      predicted document ID and its corresponding score.
    top_k: The number of top doc ids to keep.

  Returns:
    A sequence of predicted document IDs, sorted in descending order of score
    and truncated to `top_k`.
  """
  ranked_score_and_doc_ids = sorted(
      score_and_doc_ids, key=lambda x: x[0], reverse=True
  )
  ranked_doc_ids = [doc_id for _, doc_id in ranked_score_and_doc_ids]
  ranked_doc_ids = [doc_id for doc_id, _ in itertools.groupby(ranked_doc_ids)]
  if len(ranked_doc_ids) != len(set(ranked_doc_ids)):
    logger.warning(
        'Duplicate doc ids found in ranked doc ids: %s', ranked_doc_ids
    )
  return ranked_doc_ids[:top_k]


def _compute_metrics(
    predictions: RetrievalPredictionsCache,
    reference_ids: Sequence[RetrievalReferenceId],
    top_k: int = 10,
) -> list[types.Score]:
  """Computes the quality metrics for the given predictions.

  Args:
    predictions: A cache of predictions for each sound id.
    reference_ids: The reference ids used for metric computation.
    top_k: The number of top predictions to consider.

  Returns:
    A list of Score objects containing the final, aggregated scores, including
    mean reciprocal rank (MRR) and exact match (EM).
  """
  values_by_metric = {'mrr': [], 'em': [], 'invalid': [], 'no_response': []}
  for reference_id in reference_ids:
    prediction = predictions[reference_id.sound_id]
    if prediction not in (NO_RESPONSE_STR, INVALID_ANSWER_STR):
      ranked_doc_ids = get_ranked_doc_ids(
          as_retrieval_prediction(prediction), top_k
      )
      values_by_metric['mrr'].append(
          types.WeightedValue(
              value=metrics_lib.compute_reciprocal_rank(
                  reference_id.reference_id, ranked_doc_ids
              )
          )
      )
      values_by_metric['em'].append(
          types.WeightedValue(
              value=float(reference_id.reference_id == ranked_doc_ids[0])
          )
      )
      values_by_metric['invalid'].append(types.WeightedValue(value=0.0))
      values_by_metric['no_response'].append(types.WeightedValue(value=0.0))
    else:
      values_by_metric['mrr'].append(types.WeightedValue(value=0.0))
      values_by_metric['em'].append(types.WeightedValue(value=0.0))
      values_by_metric['invalid'].append(
          types.WeightedValue(value=float(prediction == INVALID_ANSWER_STR))
      )
      values_by_metric['no_response'].append(
          types.WeightedValue(value=float(prediction == NO_RESPONSE_STR))
      )

  mrr_score = mrr(
      *evaluator_lib.compute_weighted_average_and_std(values_by_metric['mrr'])
  )
  em_score = em(
      *evaluator_lib.compute_weighted_average_and_std(values_by_metric['em'])
  )
  invalid_result_rate = evaluator_lib.compute_weighted_average_and_std(
      values_by_metric['invalid']
  )
  invalid_result_score = types.Score(
      metric='InvalidResultRate',
      description='Invalid result rate',
      value=invalid_result_rate[0],
      min=0,
      max=1,
      std=invalid_result_rate[1],
  )
  no_result_rate = evaluator_lib.compute_weighted_average_and_std(
      values_by_metric['no_response']
  )
  no_result_score = types.Score(
      metric='NoResultRate',
      description='No result rate',
      value=no_result_rate[0],
      min=0,
      max=1,
      std=no_result_rate[1],
  )

  return [mrr_score, em_score, invalid_result_score, no_result_score]


class RetrievalEvaluator:
  """Evaluator for retrieval tasks."""

  def __init__(
      self,
      searcher: ScannSearcher,
      id_by_index_id: Sequence[str],
      top_k: int = 10,
  ):
    self.searcher = searcher
    self.id_by_index_id = id_by_index_id
    self.top_k = top_k

  def compute_predictions(
      self, embeddings_by_sound_id: types.MultiModalEmbeddingCache
  ) -> RetrievalPredictionsCache:
    """Computes the predictions for the given embeddings.

    Args:
      embeddings_by_sound_id: The embeddings to evaluate.

    Returns:
      A mapping from sound_id to a sequence of predicted document IDs, truncated
      to `self.top_k`.
    """
    predictions = {}
    for sound_id, embeddings in embeddings_by_sound_id.items():
      assert hasattr(embeddings, 'embedding')
      embedding: jaxtyping.Float[jaxtyping.Array, 'N D'] = embeddings.embedding
      ranked_index_ids, ranked_doc_scores = self.searcher.search_batched(
          embedding.astype(np.float32)
      )
      ranked_doc_scores = [  # pylint: disable=g-complex-comprehension
          [float(score) for score in scores] for scores in ranked_doc_scores
      ]
      ranked_doc_ids = [  # pylint: disable=g-complex-comprehension
          [self.id_by_index_id[int(x)] for x in ids] for ids in ranked_index_ids
      ]
      predictions[sound_id] = tuple(
          zip(ranked_doc_scores[0], ranked_doc_ids[0])
      )
    return predictions  # pytype: disable=bad-return-type

  def compute_metrics(
      self,
      predictions: RetrievalPredictionsCache,
      reference_ids: Sequence[RetrievalReferenceId],
  ) -> list[types.Score]:
    """Computes the quality metrics for the given predictions.

    Args:
      predictions: A cache of predictions for each sound id.
      reference_ids: The reference ids used for metric computation.

    Returns:
      A list of Score objects containing the final, aggregated scores, including
      mean reciprocal rank (MRR) and exact match (EM).
    """
    return _compute_metrics(predictions, reference_ids, self.top_k)


class RetrievalEvaluatorPartitioned:
  """Evaluator for retrieval tasks with partitioned index."""

  def __init__(
      self,
      index_dir: str,
      top_k: int = 10,
  ):
    self.index_dir = index_dir
    self.top_k = top_k

  def compute_predictions(
      self,
      embeddings_by_sound_id: types.MultiModalEmbeddingCache,
  ) -> RetrievalPredictionsCache:
    """Computes the predictions for the given embeddings and reference ids."""
    predictions = {}
    num_partitions = len(tuple(epath.Path(self.index_dir).glob('[0-9]*')))
    for partition_id in range(num_partitions):
      logger.info('Processing partition %d/%d', partition_id, num_partitions)
      searcher, id_by_index_id = load_index(
          scann_base_dir=os.path.join(self.index_dir, str(partition_id))
      )
      evaluator = RetrievalEvaluator(
          searcher=searcher,
          id_by_index_id=id_by_index_id,
          top_k=self.top_k,
      )
      predictions_for_partition = evaluator.compute_predictions(
          embeddings_by_sound_id
      )
      for (
          sound_id,
          predictions_for_sound_id,
      ) in predictions_for_partition.items():
        if sound_id not in predictions:
          predictions[sound_id] = list(predictions_for_sound_id)
        else:
          predictions[sound_id].extend(predictions_for_sound_id)

    return predictions  # pytype: disable=bad-return-type

  def compute_metrics(
      self,
      predictions: RetrievalPredictionsCache,
      reference_ids: Sequence[RetrievalReferenceId],
  ) -> list[types.Score]:
    """Returns quality metrics of the predictions."""
    return _compute_metrics(predictions, reference_ids, self.top_k)


def build_index(
    embeddings: types.MultiModalEmbeddingCache, k: int = 10
) -> tuple[ScannSearcher, Sequence[str]]:
  """Builds the ScaNN index from the embeddings.

  Args:
    embeddings: The embeddings to build the index from.
    k: The number of neighbors to return.

  Returns:
    A tuple of the searcher of type ScannSearcher and the mapping from index id
    (int) to id (str).
  """
  logger.info('Building ScaNN index...')
  id_by_index_id: Sequence[str] = sorted(embeddings.keys())

  def _get_embedding(embeddings: types.MultiModalEmbedding) -> np.ndarray:
    assert hasattr(embeddings, 'embedding')
    embedding: jaxtyping.Float[jaxtyping.Array, '1 D'] = embeddings.embedding
    return embedding[0]

  candidates = np.array(
      [_get_embedding(embeddings[did]) for did in id_by_index_id], np.float32
  )
  searcher = (
      scann_ops_pybind.builder(
          db=candidates, num_neighbors=k, distance_measure='dot_product'
      )
      .tree(
          num_leaves=min(2000, len(id_by_index_id)),
          num_leaves_to_search=min(100, len(id_by_index_id)),
          training_sample_size=250_000,
      )
      .score_ah(2, anisotropic_quantization_threshold=0.2)
      .reorder(100)
      .build()
  )
  _ = searcher.search(np.zeros((candidates.shape[1],)))
  _ = searcher.search_batched(np.zeros((1, candidates.shape[1])))
  logger.info('Built ScaNN index with %d documents.', len(id_by_index_id))
  return searcher, id_by_index_id


def save_index(
    searcher: ScannSearcher,
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
  logger.info('Saving ScaNN index to %s', scann_base_dir)
  epath.Path(scann_base_dir).mkdir(parents=True, exist_ok=True)
  with epath.Path(os.path.join(scann_base_dir, id_by_index_id_filepath)).open(
      'w'
  ) as f:
    f.write('\n'.join(id_by_index_id))
  searcher.serialize(scann_base_dir, relative_path=False)


def load_index(
    scann_base_dir: str,
    id_by_index_id_filepath: str = 'ids.txt',
) -> tuple[ScannSearcher, Sequence[str]]:
  """Loads the ScaNN index and its metadata from a directory.

  Args:
    scann_base_dir: The base directory for the ScaNN model.
    id_by_index_id_filepath: The filepath for the id by index id mapping
      relative to scann_base_dir.

  Returns:
    A tuple of the searcher of type ScannSearcher and the mapping from index id
    (int) to id (str).

  Raises:
    FileNotFoundError: If the ScaNN base directory or ID by index ID file is
    not found.
  """
  logger.info('Loading ScaNN index from %s', scann_base_dir)
  ids_path = epath.Path(os.path.join(scann_base_dir, id_by_index_id_filepath))
  if not epath.Path(scann_base_dir).exists() or not ids_path.exists():
    raise FileNotFoundError(
        'ScaNN base directory or ID by index ID file not found:'
        f' {scann_base_dir}.'
    )
  with epath.Path(ids_path).open('r') as f:
    id_by_index_id: Sequence[str] = f.read().splitlines()
  searcher = scann_ops_pybind.load_searcher(scann_base_dir)
  logger.info('Loaded ScaNN index with %d documents.', len(id_by_index_id))
  return searcher, id_by_index_id
