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
import shutil

from absl.testing import absltest
from mseb import types
from mseb.evaluators import retrieval_evaluator
import numpy as np
import numpy.testing as npt
import tensorflow as tf
import tensorflow_recommenders as tfrs


class RetrievalEvaluatorTest(absltest.TestCase):

  def test_compute_predictions(self):
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
        searcher=searcher,
        id_by_index_id=id_by_index_id,
    )
    predictions = evaluator.compute_predictions(
        embeddings={
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
        reference_ids=[
            retrieval_evaluator.RetrievalReferenceId(
                sound_id='1', reference_id='blo'
            ),
            retrieval_evaluator.RetrievalReferenceId(
                sound_id='2', reference_id='blu'
            ),
        ],
    )
    self.assertLen(predictions, 2)
    self.assertSequenceEqual(predictions['1'], ['blu', 'blo'])
    self.assertSequenceEqual(predictions['2'], ['blu', 'blo'])

  def test_evaluate_predictions(self):
    evaluator = retrieval_evaluator.RetrievalEvaluator(
        searcher=tfrs.layers.factorized_top_k.BruteForce(),  # Not used.
        id_by_index_id=(),  # Not used.
    )
    scores = evaluator.evaluate_predictions(
        predictions={
            '1': ['bli', 'bla', 'blo'],
            '2': ['bli', 'bla', 'blu'],
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
        searcher=searcher,
        id_by_index_id=id_by_index_id,
    )
    scores = evaluator(
        embeddings={
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
        reference_ids=[
            retrieval_evaluator.RetrievalReferenceId(
                sound_id='1', reference_id='blo'
            ),
            retrieval_evaluator.RetrievalReferenceId(
                sound_id='2', reference_id='blu'
            ),
        ],
    )
    npt.assert_equal(len(scores), 2)
    for score in scores:
      if score.metric == 'MRR':
        npt.assert_equal(score.value, (0.5 + 1.0) / 2)
        npt.assert_equal(score.std, 1 / 4)
      elif score.metric == 'EM':
        npt.assert_equal(score.value, (0.0 + 1.0) / 2)
        npt.assert_equal(score.std, 1 / 2)
      else:
        raise ValueError(f'Unexpected metric: {score.metric}')


class RetrievalEvaluatorPartitionedTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    testdata_path = path.join(
        pathlib.Path(path.abspath(__file__)).parent.parent, 'testdata'
    )
    num_partitions = 2
    self.index_dir = self.create_tempdir().full_path
    for partition_id in range(num_partitions):
      shutil.copytree(
          path.join(
              testdata_path, 'retrievals', 'svq_passage_retrieval_in_lang'
          ),
          path.join(self.index_dir, str(partition_id)),
      )

  def test_evaluate_predictions(self):
    evaluator = retrieval_evaluator.RetrievalEvaluatorPartitioned(
        index_dir='not_used'
    )
    scores = evaluator.evaluate_predictions(
        predictions={
            '1': ['bli', 'bla', 'blo'],
            '2': ['bli', 'bla', 'blu'],
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

  def test_predictions_sorted_by_score(self):
    evaluator = retrieval_evaluator.RetrievalEvaluatorPartitioned(
        index_dir='not_used', top_k=2
    )
    predictions_with_scores = {
        '1': [('bli', 1.0), ('bla', 0.5), ('blo', 0.25)],
        '2': [('bli', 0.5), ('bla', 0.25), ('blu', 1.0), ('blu', 1.0)],
    }
    predictions = evaluator._predictions_sorted_by_score(
        predictions_with_scores
    )
    self.assertLen(predictions, 2)
    self.assertSequenceEqual(predictions['1'], ['bli', 'bla'])
    self.assertSequenceEqual(predictions['2'], ['blu', 'bli'])

  def test_compute_predictions(self):
    evaluator = retrieval_evaluator.RetrievalEvaluatorPartitioned(
        index_dir=self.index_dir
    )
    predictions = evaluator.compute_predictions(
        embeddings={
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
        reference_ids=[
            retrieval_evaluator.RetrievalReferenceId(
                sound_id='1', reference_id='blo'
            ),
            retrieval_evaluator.RetrievalReferenceId(
                sound_id='2', reference_id='blu'
            ),
        ],
    )
    self.assertLen(predictions, 2)
    self.assertSequenceEqual(predictions['1'], ['blu', 'blo'])
    self.assertSequenceEqual(predictions['2'], ['blu', 'blo'])

  def test_call(self):
    evaluator = retrieval_evaluator.RetrievalEvaluatorPartitioned(
        index_dir=self.index_dir
    )
    scores = evaluator(
        embeddings={
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
        reference_ids=[
            retrieval_evaluator.RetrievalReferenceId(
                sound_id='1', reference_id='blo'
            ),
            retrieval_evaluator.RetrievalReferenceId(
                sound_id='2', reference_id='blu'
            ),
        ],
    )
    npt.assert_equal(len(scores), 2)
    for score in scores:
      if score.metric == 'MRR':
        npt.assert_equal(score.value, (0.5 + 1.0) / 2)  # 2/3
        npt.assert_equal(score.std, 1 / 4)  # 1/3
      elif score.metric == 'EM':
        npt.assert_equal(score.value, (0.0 + 1.0) / 2)
        npt.assert_equal(score.std, 1 / 2)
      else:
        raise ValueError(f'Unexpected metric: {score.metric}')


class ScannIndexTest(absltest.TestCase):

  def test_build_scann_index(self):
    embeddings = {
        str(i): types.TextEmbeddings(
            embeddings=np.array([[1, 2, 3]]) + i,
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
    results = searcher(tf.constant([[4.0, 5.0, 6.0]], dtype=tf.float32))
    self.assertLen(results, 2)
    npt.assert_array_equal(results[0], [[257.0, 242.0]])
    npt.assert_array_equal(results[1], [[7, 6]])

  def test_save_and_load_scann_index(self):
    embeddings = {
        str(i): types.TextEmbeddings(
            embeddings=np.array([[1, 2, 3]]) + i,
            spans=np.array([[0, 10]]),
            context=types.TextContextParams(id=str(i)),
        )
        for i in range(16)
    }
    searcher, id_by_index_id = retrieval_evaluator.build_index(embeddings)
    results = searcher(tf.constant([[4.0, 5.0, 6.0]], dtype=tf.float32))
    scann_base_dir = self.create_tempdir().full_path
    retrieval_evaluator.save_index(searcher, id_by_index_id, scann_base_dir)
    searcher_loaded, id_by_index_id_loaded = retrieval_evaluator.load_index(
        scann_base_dir
    )
    results_loaded = searcher_loaded(
        tf.constant([[4.0, 5.0, 6.0]], dtype=tf.float32)
    )
    self.assertEqual(len(results_loaded), len(results))
    for i in range(len(results)):
      npt.assert_array_equal(results[i], results_loaded[i])
    self.assertSequenceEqual(id_by_index_id_loaded, id_by_index_id)


if __name__ == '__main__':
  absltest.main()
