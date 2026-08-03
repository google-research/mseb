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

import os
import pathlib
import shutil

from absl.testing import absltest
from mseb import types
from mseb.evaluators import retrieval_evaluator
import numpy as np
import numpy.testing as npt
import pytest

BruteForceSearcher = retrieval_evaluator.BruteForceSearcher


@pytest.mark.scann
@pytest.mark.optional
class RetrievalEvaluatorTest(absltest.TestCase):

  def test_compute_recall_at_k(self):
    reference = 'bla'
    predicted_neighbors = ['bli', 'bla', 'blu']
    self.assertEqual(
        retrieval_evaluator.compute_recall_at_k(
            reference, predicted_neighbors, k=1
        ),
        0.0,
    )
    self.assertEqual(
        retrieval_evaluator.compute_recall_at_k(
            reference, predicted_neighbors, k=2
        ),
        1.0,
    )
    self.assertEqual(
        retrieval_evaluator.compute_recall_at_k(
            reference, predicted_neighbors, k=3
        ),
        1.0,
    )

  def test_compute_recall_at_k_multi_reference(self):
    """recall_at_k with multiple valid references."""
    reference = ['bla', 'blo']
    predicted_neighbors = ['bli', 'blo', 'bla', 'blu']
    # k=1: 'bli' not in reference -> 0.0
    self.assertEqual(
        retrieval_evaluator.compute_recall_at_k(
            reference, predicted_neighbors, k=1
        ),
        0.0,
    )
    # k=2: 'blo' in reference -> 1.0
    self.assertEqual(
        retrieval_evaluator.compute_recall_at_k(
            reference, predicted_neighbors, k=2
        ),
        1.0,
    )

  def test_compute_recall_at_k_multi_reference_no_match(self):
    """No match among any of the multi-references."""
    reference = ['x', 'y']
    predicted_neighbors = ['a', 'b', 'c']
    self.assertEqual(
        retrieval_evaluator.compute_recall_at_k(
            reference, predicted_neighbors, k=3
        ),
        0.0,
    )

  def test_compute_predictions(self):
    id_by_index_id = ('bli', 'bla', 'blo', 'blu')
    candidates = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ],
        np.float32,
    )
    searcher = BruteForceSearcher(candidates, num_neighbors=2)
    evaluator = retrieval_evaluator.RetrievalEvaluator(
        searcher=searcher,
        id_by_index_id=id_by_index_id,
    )
    predictions = evaluator.compute_predictions(
        embeddings_by_sound_id={
            '1': types.SoundEmbedding(
                timestamps=np.array(
                    [[0.0, 1.0]]
                ),  # pyrefly: ignore[bad-argument-type]
                embedding=np.array([[1.0, 2.0, 3.0]]),
                context=types.SoundContextParams(
                    id='1', sample_rate=16000, length=16000 * 5
                ),
            ),
            '2': types.SoundEmbedding(
                timestamps=np.array(
                    [[0.0, 1.0]]
                ),  # pyrefly: ignore[bad-argument-type]
                embedding=np.array([[1.0, 2.0, 3.0]]),
                context=types.SoundContextParams(
                    id='2', sample_rate=16000, length=16000 * 5
                ),
            ),
        },
    )
    self.assertLen(predictions, 2)
    predictions_1 = predictions['1']
    self.assertIsInstance(predictions_1, types.ValidListPrediction)
    self.assertSequenceEqual(
        predictions_1.items,
        [{'id': 'blu', 'score': 32.0}, {'id': 'blo', 'score': 26.0}],
    )
    predictions_2 = predictions['2']
    self.assertIsInstance(predictions_2, types.ValidListPrediction)
    self.assertSequenceEqual(
        predictions_2.items,
        [{'id': 'blu', 'score': 32.0}, {'id': 'blo', 'score': 26.0}],
    )

  def test_compute_metrics(self):
    dummy = BruteForceSearcher(np.zeros((1, 1), np.float32), num_neighbors=1)
    evaluator = retrieval_evaluator.RetrievalEvaluator(
        searcher=dummy,
        id_by_index_id=(),  # Not used.
    )
    scores = evaluator.compute_metrics(
        predictions={
            '1': types.ValidListPrediction([
                {'id': 'bli', 'score': 1.0},
                {'id': 'bla', 'score': 0.5},
                {'id': 'blo', 'score': 0.25},
            ]),
            '2': types.ValidListPrediction([
                {'id': 'bli', 'score': 1.0},
                {'id': 'bla', 'score': 0.5},
                {'id': 'blu', 'score': 0.25},
            ]),
        },
        reference_ids=[
            retrieval_evaluator.RetrievalReferenceId(
                sound_id='1', reference_id='bla'
            ),
            retrieval_evaluator.RetrievalReferenceId(
                sound_id='2', reference_id='bli'
            ),
        ],
    )
    self.assertLen(scores, 8)
    for score in scores:
      if score.metric == 'MRR':
        npt.assert_equal(score.value, (0.5 + 1.0) / 2)
        npt.assert_equal(score.std, 1 / 4)
      elif score.metric == 'EM':
        npt.assert_equal(score.value, (0.0 + 1.0) / 2)
        npt.assert_equal(score.std, 1 / 2)
      elif score.metric == 'MAP':
        npt.assert_equal(score.value, (0.5 + 1.0) / 2)
        npt.assert_equal(score.std, 1 / 4)
      elif score.metric == 'RecallAt10':
        npt.assert_equal(score.value, (1.0 + 1.0) / 2)
        npt.assert_equal(score.std, 0.0)
      elif score.metric == 'RecallAtInf':
        npt.assert_equal(score.value, (1.0 + 1.0) / 2)
        npt.assert_equal(score.std, 0.0)
      elif score.metric == 'InvalidResultRate':
        npt.assert_equal(score.value, 0.0)
        npt.assert_equal(score.std, 0.0)
      elif score.metric == 'NoResultRate':
        npt.assert_equal(score.value, 0.0)
        npt.assert_equal(score.std, 0.0)
      elif score.metric == 'NDCG@10':
        # Query '1': ref='bla' at rank 2 → 1/log2(3) ≈ 0.6309
        # Query '2': ref='bli' at rank 1 → 1/log2(2) = 1.0
        expected = (1.0 / np.log2(3) + 1.0) / 2
        npt.assert_almost_equal(score.value, expected, decimal=4)
      else:
        raise ValueError(f'Unexpected metric: {score.metric}')


@pytest.mark.scann
@pytest.mark.optional
class RetrievalEvaluatorPartitionedTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, 'testdata'
    )
    num_partitions = 2
    self.index_dir = self.create_tempdir().full_path

    source_base_dir = os.path.join(
        testdata_path, 'retrievals', 'svq_passage_retrieval_in_lang'
    )
    for partition_id in range(num_partitions):
      target_base_dir = os.path.join(self.index_dir, str(partition_id))
      shutil.copytree(source_base_dir, target_base_dir)
      os.chmod(os.path.join(target_base_dir, 'scann_assets.pbtxt'), 0o755)
      with open(
          os.path.join(target_base_dir, 'scann_assets.pbtxt'),
          'w',
      ) as fout:
        with open(
            os.path.join(source_base_dir, 'scann_assets.pbtxt'),
        ) as fin:
          for line in fin:
            line = line.replace(
                'asset_path: "dataset.npy"',
                f'asset_path: "{target_base_dir}/dataset.npy"',
            )
            fout.write(line)

  def test_compute_metrics(self):
    evaluator = retrieval_evaluator.RetrievalEvaluatorPartitioned(
        index_dir='not_used'
    )
    scores = evaluator.compute_metrics(
        predictions={
            '1': types.ValidListPrediction([
                {'id': 'bli', 'score': 1.0},
                {'id': 'bla', 'score': 0.5},
                {'id': 'blu', 'score': 0.25},
            ]),
            '2': types.ValidListPrediction([
                {'id': 'bli', 'score': 1.0},
                {'id': 'bla', 'score': 0.5},
                {'id': 'blu', 'score': 0.25},
            ]),
        },
        reference_ids=[
            retrieval_evaluator.RetrievalReferenceId(
                sound_id='1', reference_id='bla'
            ),
            retrieval_evaluator.RetrievalReferenceId(
                sound_id='2', reference_id='bli'
            ),
        ],
    )
    self.assertLen(scores, 8)
    for score in scores:
      if score.metric == 'MRR':
        npt.assert_equal(score.value, (0.5 + 1.0) / 2)
        npt.assert_equal(score.std, 1 / 4)
      elif score.metric == 'EM':
        npt.assert_equal(score.value, (0.0 + 1.0) / 2)
        npt.assert_equal(score.std, 1 / 2)
      elif score.metric == 'MAP':
        npt.assert_equal(score.value, (0.5 + 1.0) / 2)
        npt.assert_equal(score.std, 1 / 4)
      elif score.metric == 'RecallAt10':
        npt.assert_equal(score.value, (1.0 + 1.0) / 2)
        npt.assert_equal(score.std, 0.0)
      elif score.metric == 'RecallAtInf':
        npt.assert_equal(score.value, (1.0 + 1.0) / 2)
        npt.assert_equal(score.std, 0.0)
      elif score.metric == 'InvalidResultRate':
        npt.assert_equal(score.value, 0.0)
        npt.assert_equal(score.std, 0.0)
      elif score.metric == 'NoResultRate':
        npt.assert_equal(score.value, 0.0)
        npt.assert_equal(score.std, 0.0)
      elif score.metric == 'NDCG@10':
        expected = (1.0 / np.log2(3) + 1.0) / 2
        npt.assert_almost_equal(score.value, expected, decimal=4)
      else:
        raise ValueError(f'Unexpected metric: {score.metric}')

  def test_compute_predictions(self):
    evaluator = retrieval_evaluator.RetrievalEvaluatorPartitioned(
        index_dir=self.index_dir
    )
    predictions = evaluator.compute_predictions(
        embeddings_by_sound_id={
            '1': types.SoundEmbedding(
                timestamps=np.array(
                    [[0.0, 1.0]]
                ),  # pyrefly: ignore[bad-argument-type]
                embedding=np.array([[1.0, 2.0, 3.0]]),
                context=types.SoundContextParams(
                    id='1', sample_rate=16000, length=16000 * 5
                ),
            ),
            '2': types.SoundEmbedding(
                timestamps=np.array(
                    [[0.0, 1.0]]
                ),  # pyrefly: ignore[bad-argument-type]
                embedding=np.array([[1.0, 2.0, 3.0]]),
                context=types.SoundContextParams(
                    id='2', sample_rate=16000, length=16000 * 5
                ),
            ),
        },
    )
    self.assertLen(predictions, 2)
    predictions_1 = predictions['1']
    self.assertIsInstance(predictions_1, types.ValidListPrediction)
    self.assertSequenceEqual(
        predictions_1.items,
        [
            {'id': 'blu', 'score': 32.0},
            {'id': 'blo', 'score': 26.0},
        ],
    )
    predictions_2 = predictions['2']
    self.assertIsInstance(predictions_2, types.ValidListPrediction)
    self.assertSequenceEqual(
        predictions_2.items,
        [
            {'id': 'blu', 'score': 32.0},
            {'id': 'blo', 'score': 26.0},
        ],
    )


@pytest.mark.scann
@pytest.mark.optional
class RetrievalEvaluatorUtilTest(absltest.TestCase):

  def test_get_ranked_doc_ids(self):
    predictions_1 = types.ValidListPrediction([
        {'id': 'bli', 'score': 1.0},
        {'id': 'bla', 'score': 0.5},
        {'id': 'blo', 'score': 0.25},
    ])
    predictions_1.normalize(k=2)
    self.assertSequenceEqual(
        predictions_1.items,
        [{'id': 'bli', 'score': 1.0}, {'id': 'bla', 'score': 0.5}],
    )

    predictions_2 = types.ValidListPrediction([
        {'id': 'bli', 'score': 0.5},
        {'id': 'bla', 'score': 0.25},
        {'id': 'blu', 'score': 1.0},
    ])
    predictions_2.normalize(k=2)
    self.assertSequenceEqual(
        predictions_2.items,
        [{'id': 'blu', 'score': 1.0}, {'id': 'bli', 'score': 0.5}],
    )

  def test_compute_metrics(self):
    scores = retrieval_evaluator._compute_metrics(
        predictions={
            '1': types.ValidListPrediction([
                {'id': 'bli', 'score': 1.0},
                {'id': 'bla', 'score': 0.5},
                {'id': 'blo', 'score': 0.25},
            ]),
            '2': types.ValidListPrediction([
                {'id': 'bli', 'score': 1.0},
                {'id': 'bla', 'score': 0.5},
                {'id': 'blu', 'score': 0.25},
            ]),
        },
        reference_ids=[
            retrieval_evaluator.RetrievalReferenceId(
                sound_id='1', reference_id='bla'
            ),
            retrieval_evaluator.RetrievalReferenceId(
                sound_id='2', reference_id='bli'
            ),
        ],
    )
    self.assertLen(scores, 8)
    for score in scores:
      if score.metric == 'MRR':
        npt.assert_equal(score.value, (0.5 + 1.0) / 2)
        npt.assert_equal(score.std, 1 / 4)
      elif score.metric == 'EM':
        npt.assert_equal(score.value, (0.0 + 1.0) / 2)
        npt.assert_equal(score.std, 1 / 2)
      elif score.metric == 'MAP':
        npt.assert_equal(score.value, (0.5 + 1.0) / 2)
        npt.assert_equal(score.std, 1 / 4)
      elif score.metric == 'RecallAt10':
        npt.assert_equal(score.value, (1.0 + 1.0) / 2)
        npt.assert_equal(score.std, 0.0)
      elif score.metric == 'RecallAtInf':
        npt.assert_equal(score.value, (1.0 + 1.0) / 2)
        npt.assert_equal(score.std, 0.0)
      elif score.metric == 'InvalidResultRate':
        npt.assert_equal(score.value, 0.0)
        npt.assert_equal(score.std, 0.0)
      elif score.metric == 'NoResultRate':
        npt.assert_equal(score.value, 0.0)
        npt.assert_equal(score.std, 0.0)
      elif score.metric == 'NDCG@10':
        expected = (1.0 / np.log2(3) + 1.0) / 2
        npt.assert_almost_equal(score.value, expected, decimal=4)
      else:
        raise ValueError(f'Unexpected metric: {score.metric}')

  def test_build_scann_index(self):
    embeddings = {
        str(i): types.TextEmbedding(
            embedding=np.array([[1, 2, 3]]) + i,
            spans=np.array([[0, 10]]),  # pyrefly: ignore[bad-argument-type]
            context=types.TextContextParams(id=str(i)),
        )
        for i in range(16)
    }
    searcher, id_by_index_id = retrieval_evaluator.build_index(embeddings, k=2)
    self.assertSequenceEqual(
        id_by_index_id,
        [
            '0',
            '1',
            '10',
            '11',
            '12',
            '13',
            '14',
            '15',
            '2',
            '3',
            '4',
            '5',
            '6',
            '7',
            '8',
            '9',
        ],
    )
    results = searcher.search_batched(
        np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
    )
    self.assertLen(results, 2)
    npt.assert_array_equal(results[0], [[7, 6]])
    npt.assert_array_equal(results[1], [[257.0, 242.0]])

  def test_save_and_load_scann_index(self):
    embeddings = {
        str(i): types.TextEmbedding(
            embedding=np.array([[1, 2, 3]]) + i,
            spans=np.array([[0, 10]]),  # pyrefly: ignore[bad-argument-type]
            context=types.TextContextParams(id=str(i)),
        )
        for i in range(16)
    }
    searcher, id_by_index_id = retrieval_evaluator.build_index(embeddings)
    results = searcher.search_batched(
        np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
    )
    scann_base_dir = self.create_tempdir().full_path
    retrieval_evaluator.save_index(searcher, id_by_index_id, scann_base_dir)
    searcher_loaded, id_by_index_id_loaded = retrieval_evaluator.load_index(
        scann_base_dir
    )
    results_loaded = searcher_loaded.search_batched(
        np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
    )
    self.assertEqual(len(results_loaded), len(results))
    for i in range(len(results)):
      npt.assert_array_equal(results[i], results_loaded[i])
    self.assertSequenceEqual(id_by_index_id_loaded, id_by_index_id)


class BruteForceSearcherTest(absltest.TestCase):
  """Tests for BruteForceSearcher, focusing on the argpartition top-k."""

  def _make_candidates(self):
    """Returns a 6x3 candidate matrix with known dot-product ordering."""
    return np.array(
        [
            [1.0, 0.0, 0.0],  # idx 0
            [0.0, 1.0, 0.0],  # idx 1
            [0.0, 0.0, 1.0],  # idx 2
            [1.0, 1.0, 0.0],  # idx 3  (dot with [1,1,1] = 2)
            [1.0, 1.0, 1.0],  # idx 4  (dot with [1,1,1] = 3)
            [2.0, 0.0, 0.0],  # idx 5  (dot with [1,1,1] = 2)
        ],
        dtype=np.float32,
    )

  def test_search_batched_top_k_ranking(self):
    """Top-k results are sorted descending by dot product."""
    candidates = self._make_candidates()
    searcher = BruteForceSearcher(candidates, num_neighbors=3)
    query = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    ids, scores = searcher.search_batched(query)
    # idx 4 has dot=3, idx 3 and 5 have dot=2
    npt.assert_array_equal(ids[0, 0], 4)
    npt.assert_almost_equal(scores[0, 0], 3.0)
    # Next two should be indices 3 and 5 (both dot=2), order may vary.
    self.assertSetEqual(set(ids[0, 1:].tolist()), {3, 5})
    npt.assert_almost_equal(scores[0, 1], 2.0)
    npt.assert_almost_equal(scores[0, 2], 2.0)

  def test_search_batched_k_equals_n(self):
    """When k == number of candidates, all are returned sorted."""
    candidates = self._make_candidates()
    n = len(candidates)
    searcher = BruteForceSearcher(candidates, num_neighbors=n)
    query = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    ids, scores = searcher.search_batched(query)
    self.assertEqual(ids.shape, (1, n))
    # Scores must be monotonically non-increasing.
    for i in range(n - 1):
      self.assertGreaterEqual(scores[0, i], scores[0, i + 1])

  def test_search_batched_k_equals_one(self):
    """When k == 1, only the best match is returned."""
    candidates = self._make_candidates()
    searcher = BruteForceSearcher(candidates, num_neighbors=1)
    query = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    ids, scores = searcher.search_batched(query)
    self.assertEqual(ids.shape, (1, 1))
    npt.assert_array_equal(ids[0], [4])
    npt.assert_almost_equal(scores[0], [3.0])

  def test_search_batched_multiple_queries(self):
    """Each query in a batch gets independently correct results."""
    candidates = self._make_candidates()
    searcher = BruteForceSearcher(candidates, num_neighbors=1)
    queries = np.array(
        [
            [1.0, 0.0, 0.0],  # best match: idx 5 (dot=2)
            [0.0, 0.0, 1.0],  # best match: idx 2 (dot=1) or idx 4 (dot=1)
        ],
        dtype=np.float32,
    )
    ids, scores = searcher.search_batched(queries)
    self.assertEqual(ids.shape, (2, 1))
    npt.assert_array_equal(ids[0], [5])
    npt.assert_almost_equal(scores[0], [2.0])
    # For second query [0,0,1]: idx 2 has dot=1, idx 4 has dot=1.
    self.assertIn(ids[1, 0], [2, 4])

  def test_search_single_query(self):
    """search() returns the same ids as search_batched() for a single query."""
    candidates = self._make_candidates()
    searcher = BruteForceSearcher(candidates, num_neighbors=2)
    query = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    single_ids, _ = searcher.search(query)
    batch_ids, _ = searcher.search_batched(query[np.newaxis, :])
    npt.assert_array_equal(single_ids, batch_ids[0])

  def test_scores_match_manual_dot_product(self):
    """Returned scores exactly equal the dot products."""
    candidates = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=np.float32,
    )
    searcher = BruteForceSearcher(candidates, num_neighbors=3)
    query = np.array([[1.0, 1.0]], dtype=np.float32)
    ids, scores = searcher.search_batched(query)
    expected_dots = {0: 3.0, 1: 7.0, 2: 11.0}
    for i in range(3):
      npt.assert_almost_equal(scores[0, i], expected_dots[ids[0, i]])

  def test_serialize_and_load_roundtrip(self):
    """Serialize then load produces identical results."""
    candidates = self._make_candidates()
    searcher = BruteForceSearcher(candidates, num_neighbors=2)
    query = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    ids_before, scores_before = searcher.search_batched(query)

    tmpdir = self.create_tempdir().full_path
    searcher.serialize(tmpdir)
    loaded = BruteForceSearcher.load_searcher(tmpdir)

    ids_after, scores_after = loaded.search_batched(query)
    npt.assert_array_equal(ids_before, ids_after)
    npt.assert_array_almost_equal(scores_before, scores_after)
    self.assertEqual(loaded.num_neighbors, searcher.num_neighbors)

  def test_load_missing_file_raises(self):
    """load_searcher raises FileNotFoundError for missing files."""
    tmpdir = self.create_tempdir().full_path
    with self.assertRaises(FileNotFoundError):
      BruteForceSearcher.load_searcher(tmpdir)

  def test_conforms_to_searcher_protocol(self):
    """BruteForceSearcher satisfies the Searcher protocol."""
    candidates = np.zeros((2, 3), dtype=np.float32)
    searcher = BruteForceSearcher(candidates, num_neighbors=1)
    self.assertIsInstance(searcher, retrieval_evaluator.Searcher)


if __name__ == '__main__':
  absltest.main()
