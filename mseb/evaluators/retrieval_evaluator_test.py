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

import os
import pathlib
import shutil

from absl.testing import absltest
from mseb import types
import numpy as np
import numpy.testing as npt
import pytest

scann_ops_pybind = pytest.importorskip('scann.scann_ops_pybind')
ScannSearcher = scann_ops_pybind.ScannSearcher

retrieval_evaluator = pytest.importorskip('mseb.evaluators.retrieval_evaluator')


@pytest.mark.scann
@pytest.mark.optional
class RetrievalEvaluatorTest(absltest.TestCase):

  def test_compute_predictions(self):
    id_by_index_id = ('bli', 'bla', 'blo', 'blu')
    searcher = (
        scann_ops_pybind.builder(
            db=np.array(
                [
                    [1.0, 2.0, 3.0],
                    [2.0, 3.0, 4.0],
                    [3.0, 4.0, 5.0],
                    [4.0, 5.0, 6.0],
                ],
                np.float32,
            ),
            num_neighbors=2,
            distance_measure='dot_product',
        )
        .score_brute_force()
        .build()
    )
    evaluator = retrieval_evaluator.RetrievalEvaluator(
        searcher=searcher,
        id_by_index_id=id_by_index_id,
    )
    predictions = evaluator.compute_predictions(
        embeddings_by_sound_id={
            '1': types.SoundEmbedding(
                timestamps=np.array([[0.0, 1.0]]),
                embedding=np.array([[1.0, 2.0, 3.0]]),
                context=types.SoundContextParams(
                    id='1', sample_rate=16000, length=16000 * 5
                ),
            ),
            '2': types.SoundEmbedding(
                timestamps=np.array([[0.0, 1.0]]),
                embedding=np.array([[1.0, 2.0, 3.0]]),
                context=types.SoundContextParams(
                    id='2', sample_rate=16000, length=16000 * 5
                ),
            ),
        },
    )
    self.assertLen(predictions, 2)
    self.assertSequenceEqual(predictions['1'], [(32.0, 'blu'), (26.0, 'blo')])
    self.assertSequenceEqual(predictions['2'], [(32.0, 'blu'), (26.0, 'blo')])

  def test_compute_metrics(self):
    evaluator = retrieval_evaluator.RetrievalEvaluator(
        searcher=ScannSearcher(None),  # Not used.
        id_by_index_id=(),  # Not used.
    )
    scores = evaluator.compute_metrics(
        predictions={
            '1': [(1.0, 'bli'), (0.5, 'bla'), (0.25, 'blo')],
            '2': [(1.0, 'bli'), (0.5, 'bla'), (0.25, 'blu')],
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
    self.assertLen(scores, 2)
    for score in scores:
      if score.metric == 'MRR':
        npt.assert_equal(score.value, (0.5 + 1.0) / 2)
        npt.assert_equal(score.std, 1 / 4)
      elif score.metric == 'EM':
        npt.assert_equal(score.value, (0.0 + 1.0) / 2)
        npt.assert_equal(score.std, 1 / 2)
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
            '1': [(1.0, 'bli'), (0.5, 'bla'), (0.25, 'blo')],
            '2': [(1.0, 'bli'), (0.5, 'bla'), (0.25, 'blu')],
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
    self.assertLen(scores, 2)
    for score in scores:
      if score.metric == 'MRR':
        npt.assert_equal(score.value, (0.5 + 1.0) / 2)
        npt.assert_equal(score.std, 1 / 4)
      elif score.metric == 'EM':
        npt.assert_equal(score.value, (0.0 + 1.0) / 2)
        npt.assert_equal(score.std, 1 / 2)
      else:
        raise ValueError(f'Unexpected metric: {score.metric}')

  def test_compute_predictions(self):
    evaluator = retrieval_evaluator.RetrievalEvaluatorPartitioned(
        index_dir=self.index_dir
    )
    predictions = evaluator.compute_predictions(
        embeddings_by_sound_id={
            '1': types.SoundEmbedding(
                timestamps=np.array([[0.0, 1.0]]),
                embedding=np.array([[1.0, 2.0, 3.0]]),
                context=types.SoundContextParams(
                    id='1', sample_rate=16000, length=16000 * 5
                ),
            ),
            '2': types.SoundEmbedding(
                timestamps=np.array([[0.0, 1.0]]),
                embedding=np.array([[1.0, 2.0, 3.0]]),
                context=types.SoundContextParams(
                    id='2', sample_rate=16000, length=16000 * 5
                ),
            ),
        },
    )
    self.assertLen(predictions, 2)
    self.assertSequenceEqual(
        predictions['1'],
        [(32.0, 'blu'), (26.0, 'blo'), (32.0, 'blu'), (26.0, 'blo')],
    )
    self.assertSequenceEqual(
        predictions['2'],
        [(32.0, 'blu'), (26.0, 'blo'), (32.0, 'blu'), (26.0, 'blo')],
    )


@pytest.mark.scann
@pytest.mark.optional
class RetrievalEvaluatorUtilTest(absltest.TestCase):

  def test_get_ranked_doc_ids(self):
    predictions_1 = [(1.0, 'bli'), (0.5, 'bla'), (0.25, 'blo')]
    ranked_doc_ids_1 = retrieval_evaluator.get_ranked_doc_ids(
        predictions_1, top_k=2
    )
    self.assertSequenceEqual(ranked_doc_ids_1, ['bli', 'bla'])

    predictions_2 = [(0.5, 'bli'), (0.25, 'bla'), (1.0, 'blu')]
    ranked_doc_ids_2 = retrieval_evaluator.get_ranked_doc_ids(
        predictions_2, top_k=2
    )
    self.assertSequenceEqual(ranked_doc_ids_2, ['blu', 'bli'])

  def test_compute_metrics(self):
    scores = retrieval_evaluator._compute_metrics(
        predictions={
            '1': [(1.0, 'bli'), (0.5, 'bla'), (0.25, 'blo')],
            '2': [(1.0, 'bli'), (0.5, 'bla'), (0.25, 'blu')],
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
    self.assertLen(scores, 2)
    for score in scores:
      if score.metric == 'MRR':
        npt.assert_equal(score.value, (0.5 + 1.0) / 2)
        npt.assert_equal(score.std, 1 / 4)
      elif score.metric == 'EM':
        npt.assert_equal(score.value, (0.0 + 1.0) / 2)
        npt.assert_equal(score.std, 1 / 2)
      else:
        raise ValueError(f'Unexpected metric: {score.metric}')

  def test_build_scann_index(self):
    embeddings = {
        str(i): types.TextEmbedding(
            embedding=np.array([[1, 2, 3]]) + i,
            spans=np.array([[0, 10]]),
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
            spans=np.array([[0, 10]]),
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


if __name__ == '__main__':
  absltest.main()
