# Copyright 2026 The MSEB Authors.
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
import logging
import math
import os
from typing import Mapping, Protocol, runtime_checkable

from etils import epath
import jaxtyping
from mseb import evaluator as evaluator_lib
from mseb import metrics as metrics_lib
from mseb import types
import numpy as np

logger = logging.getLogger(__name__)

try:
  from scann.scann_ops.py import scann_ops_pybind
except ImportError:
  scann_ops_pybind = None


class BruteForceSearcher:
  """Dot-product brute force nearest-neighbor searcher."""

  def __init__(self, candidates: np.ndarray, num_neighbors: int):
    self.candidates = candidates
    self.num_neighbors = num_neighbors

  def search_batched(
      self, embeddings: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    """Returns (ranked_index_ids, ranked_doc_scores) for a batch."""
    dot_products = np.matmul(embeddings, self.candidates.T)
    # Use argpartition to get the top K indices (unsorted)
    top_k_indices = np.argpartition(dot_products, -self.num_neighbors, axis=1)[
        :, -self.num_neighbors :
    ]
    # Sort only the top K elements
    top_k_dots = np.take_along_axis(dot_products, top_k_indices, axis=1)
    sorted_top_k_idx = np.argsort(top_k_dots, axis=1)[:, ::-1]

    ranked_index_ids = np.take_along_axis(
        top_k_indices, sorted_top_k_idx, axis=1
    )
    ranked_doc_scores = np.take_along_axis(top_k_dots, sorted_top_k_idx, axis=1)
    return ranked_index_ids, ranked_doc_scores

  def search(self, embedding: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ranked_index_ids, ranked_doc_scores = self.search_batched(
        embedding[np.newaxis, :]
    )
    return ranked_index_ids[0], ranked_doc_scores[0]

  def serialize(self, path: str, relative_path: bool = False) -> None:
    """Saves the index to disk."""
    assert not relative_path
    p = epath.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    with (p / 'candidates.npy').open('wb') as f:
      np.save(f, self.candidates)
    (p / 'num_neighbors.txt').write_text(str(self.num_neighbors))

  @classmethod
  def load_searcher(cls, path: str) -> BruteForceSearcher:
    """Loads a BruteForceSearcher from disk.

    Raises:
      FileNotFoundError: If the candidates file does not exist.
    """
    p = epath.Path(path)
    try:
      with (p / 'candidates.npy').open('rb') as f:
        candidates = np.load(f)
    except OSError as e:
      raise FileNotFoundError(
          f'Failed to load candidates from {p / "candidates.npy"}'
      ) from e
    num_neighbors = int((p / 'num_neighbors.txt').read_text())
    return cls(candidates, num_neighbors)


@runtime_checkable
class Searcher(Protocol):
  """Protocol for nearest-neighbor searcher (BruteForceSearcher or ScaNN)."""

  def search(self, embedding: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ...

  def search_batched(
      self, embeddings: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    ...

  def serialize(self, path: str, relative_path: bool = False) -> None:
    ...


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


def compute_recall_at_k(
    reference: str | Sequence[str],
    predicted_neighbors: Sequence[str],
    k: int = 10,
) -> float:
  """Computes the recall at k."""
  if isinstance(reference, str):
    reference = [reference]
  for neighbor in predicted_neighbors[:k]:
    if neighbor in reference:
      return 1.0
  return 0.0


@dataclasses.dataclass
class RetrievalReferenceId:
  sound_id: str
  reference_id: str | Sequence[str]


RetrievalPredictionsCache = Mapping[str, types.ListPrediction]


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
  values_by_metric = {
      'mrr': [],
      'em': [],
      'recall_at_k': [],
      'recall_at_inf': [],
      'invalid': [],
      'no_response': [],
      'ndcg': [],
  }
  for reference_id in reference_ids:
    if reference_id.sound_id in predictions:
      prediction = predictions[reference_id.sound_id]
    else:
      prediction = types.NoResponseListPrediction()
    if isinstance(prediction, types.ValidListPrediction):
      prediction.normalize(k=top_k)
      ranked_doc_ids = [x['id'] for x in prediction.items]
      values_by_metric['mrr'].append(
          types.WeightedValue(
              value=metrics_lib.compute_reciprocal_rank(
                  reference_id.reference_id, ranked_doc_ids
              )
          )
      )
      values_by_metric['em'].append(
          types.WeightedValue(
              value=metrics_lib.compute_exact_match(
                  reference_id.reference_id, ranked_doc_ids
              )
          )
      )
      values_by_metric['recall_at_k'].append(
          types.WeightedValue(
              value=compute_recall_at_k(
                  reference_id.reference_id, ranked_doc_ids, k=top_k
              )
          )
      )
      values_by_metric['recall_at_inf'].append(
          types.WeightedValue(
              value=compute_recall_at_k(
                  reference_id.reference_id,
                  ranked_doc_ids,
                  k=len(ranked_doc_ids),
              )
          )
      )
      values_by_metric['ndcg'].append(
          types.WeightedValue(
              value=metrics_lib.compute_ndcg_at_k(
                  reference_id.reference_id, ranked_doc_ids, k=10  # pyrefly: ignore[bad-argument-type]
              )
          )
      )
      values_by_metric['invalid'].append(types.WeightedValue(value=0.0))
      values_by_metric['no_response'].append(types.WeightedValue(value=0.0))
    else:
      values_by_metric['mrr'].append(types.WeightedValue(value=0.0))
      values_by_metric['em'].append(types.WeightedValue(value=0.0))
      values_by_metric['recall_at_k'].append(types.WeightedValue(value=0.0))
      values_by_metric['recall_at_inf'].append(types.WeightedValue(value=0.0))
      values_by_metric['ndcg'].append(types.WeightedValue(value=0.0))
      values_by_metric['invalid'].append(
          types.WeightedValue(
              value=float(
                  isinstance(prediction, types.InvalidAnswerListPrediction)
              )
          )
      )
      values_by_metric['no_response'].append(
          types.WeightedValue(
              value=float(
                  isinstance(prediction, types.NoResponseListPrediction)
              )
          )
      )

  mrr_score = mrr(
      *evaluator_lib.compute_weighted_average_and_std(values_by_metric['mrr'])
  )
  em_score = em(
      *evaluator_lib.compute_weighted_average_and_std(values_by_metric['em'])
  )
  recall_at_k = evaluator_lib.compute_weighted_average_and_std(
      values_by_metric['recall_at_k']
  )
  recall_at_k_score = types.Score(
      metric=f'RecallAt{top_k}',
      description=f'Recall at {top_k}',
      value=recall_at_k[0],
      min=0,
      max=1,
      std=recall_at_k[1],
  )
  recall_at_inf = evaluator_lib.compute_weighted_average_and_std(
      values_by_metric['recall_at_inf']
  )
  recall_at_inf_score = types.Score(
      metric='RecallAtInf',
      description='Recall at Inf',
      value=recall_at_inf[0],
      min=0,
      max=1,
      std=recall_at_inf[1],
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

  ndcg = evaluator_lib.compute_weighted_average_and_std(
      values_by_metric['ndcg']
  )
  ndcg_score = types.Score(
      metric='NDCG@10',
      description='Normalized Discounted Cumulative Gain at 10',
      value=ndcg[0],
      min=0,
      max=1,
      std=ndcg[1],
  )

  return [
      mrr_score,
      em_score,
      recall_at_k_score,
      recall_at_inf_score,
      invalid_result_score,
      no_result_score,
      ndcg_score,
  ]


class RetrievalEvaluator:
  """Evaluator for retrieval tasks."""

  def __init__(
      self, searcher: Searcher, id_by_index_id: Sequence[str], top_k: int = 10
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
      embedding: jaxtyping.Float[jaxtyping.Array, 'N D'] = embeddings.embedding  # pyrefly: ignore[bad-assignment]
      ranked_index_ids, ranked_doc_scores = self.searcher.search_batched(
          embedding.astype(np.float32)
      )
      ranked_doc_scores = [  # pylint: disable=g-complex-comprehension
          [float(score) for score in scores] for scores in ranked_doc_scores
      ]
      ranked_doc_ids = [  # pylint: disable=g-complex-comprehension
          [self.id_by_index_id[int(x)] for x in ids] for ids in ranked_index_ids
      ]
      predictions[sound_id] = types.ValidListPrediction([
          {'id': i, 'score': s}
          for s, i in zip(ranked_doc_scores[0], ranked_doc_ids[0])
      ])
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
          predictions[sound_id] = predictions_for_sound_id
        else:
          assert isinstance(predictions[sound_id], types.ValidListPrediction)
          assert isinstance(predictions_for_sound_id, types.ValidListPrediction)
          predictions[sound_id].merge(predictions_for_sound_id)

    return predictions  # pytype: disable=bad-return-type

  def compute_metrics(
      self,
      predictions: RetrievalPredictionsCache,
      reference_ids: Sequence[RetrievalReferenceId],
  ) -> list[types.Score]:
    """Returns quality metrics of the predictions."""
    return _compute_metrics(predictions, reference_ids, self.top_k)


def build_index(
    embeddings: types.MultiModalEmbeddingCache,
    *,
    k: int = 10,
    allow_scann: bool = False,
) -> tuple[Searcher, Sequence[str]]:
  """Builds the index from the embeddings.

  Uses brute force for small indices (<50k documents) and ScaNN for larger ones
  when ``allow_scann`` is True and the library is available.

  Args:
    embeddings: The embeddings to build the index from.
    k: The number of neighbors to return.
    allow_scann: Whether to allow building a ScaNN index for large datasets.

  Returns:
    A tuple of (searcher, id_by_index_id).

  Raises:
    ImportError: If ScaNN is needed but not available.
  """

  def _get_embedding(emb: types.MultiModalEmbedding) -> np.ndarray:
    assert hasattr(emb, 'embedding')
    embedding: jaxtyping.Float[jaxtyping.Array, '1 D'] = emb.embedding  # pyrefly: ignore[bad-assignment]
    return embedding[0]  # pyrefly: ignore[bad-return]

  id_by_index_id: Sequence[str] = sorted(embeddings.keys())
  candidates = np.array(
      [_get_embedding(embeddings[did]) for did in id_by_index_id], np.float32
  )
  n = len(id_by_index_id)
  if not allow_scann or n < 50_000:
    logger.info('Building brute force index with %d documents...', n)
    searcher = BruteForceSearcher(candidates, num_neighbors=k)
  else:
    if scann_ops_pybind is None:
      raise ImportError(
          'ScaNN library is required for indices with >= 50k documents.'
          ' Install google3.research.scam or set allow_scann=False.'
      )
    num_leaves = math.isqrt(n)
    num_leaves_to_search = max(1, num_leaves // 10)
    logger.info(
        'Building ScaNN index with %d documents'
        ' (num_leaves=%d, num_leaves_to_search=%d)...',
        n,
        num_leaves,
        num_leaves_to_search,
    )
    builder = (
        scann_ops_pybind.builder(
            db=candidates, num_neighbors=k, distance_measure='dot_product'
        )
        .tree(
            num_leaves=num_leaves,
            num_leaves_to_search=num_leaves_to_search,
        )
        .score_ah(2, anisotropic_quantization_threshold=0.2)
        .reorder(200 if n >= 500_000 else 100)
    )
    searcher = builder.build()
  logger.info('Index built successfully.')
  # Warm up the searcher.
  _ = searcher.search(np.zeros((candidates.shape[1],)))
  _ = searcher.search_batched(np.zeros((1, candidates.shape[1])))
  return searcher, id_by_index_id


def save_index(
    searcher: Searcher,
    id_by_index_id: Sequence[str],
    scann_base_dir: str,
    id_by_index_id_filepath: str = 'ids.txt',
) -> None:
  """Saves the index and its metadata to a directory.

  Args:
    searcher: The searcher to save.
    id_by_index_id: The mapping from index id (int) to id (str).
    scann_base_dir: The base directory to save to.
    id_by_index_id_filepath: Filename for the id mapping within scann_base_dir.
  """
  logger.info('Saving index to %s', scann_base_dir)
  base = epath.Path(scann_base_dir)
  base.mkdir(parents=True, exist_ok=True)
  (base / id_by_index_id_filepath).write_text('\n'.join(id_by_index_id))
  searcher.serialize(scann_base_dir, relative_path=False)


def load_index(
    scann_base_dir: str,
    id_by_index_id_filepath: str = 'ids.txt',
) -> tuple[Searcher, Sequence[str]]:
  """Loads a brute force or ScaNN index and its metadata from a directory.

  Tries BruteForceSearcher first; falls back to ScaNN if the brute force
  artifacts are not found.

  Args:
    scann_base_dir: The base directory containing the index.
    id_by_index_id_filepath: Filename for the id mapping within scann_base_dir.

  Returns:
    A tuple of (searcher, id_by_index_id).

  Raises:
    FileNotFoundError: If the base directory or id mapping file is not found.
    ImportError: If ScaNN is needed but not available.
  """
  base = epath.Path(scann_base_dir)
  ids_path = base / id_by_index_id_filepath
  if not base.exists() or not ids_path.exists():
    raise FileNotFoundError(
        f'Index directory or ID file not found: {scann_base_dir}'
    )
  id_by_index_id: Sequence[str] = ids_path.read_text().splitlines()
  try:
    logger.info('Loading brute force index from %s', scann_base_dir)
    searcher = BruteForceSearcher.load_searcher(scann_base_dir)
    logger.info(
        'Loaded brute force index with %d documents.', len(id_by_index_id)
    )
  except FileNotFoundError:
    if scann_ops_pybind is None:
      raise ImportError(
          'ScaNN library is required to load this index.'
          ' Install google3.research.scam.'
      ) from None
    logger.info('Loading ScaNN index from %s', scann_base_dir)
    searcher = scann_ops_pybind.load_searcher(scann_base_dir)
    logger.info('Loaded ScaNN index with %d documents.', len(id_by_index_id))
  return searcher, id_by_index_id
