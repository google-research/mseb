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
from typing import Iterable

from absl.testing import absltest
from absl.testing import flagsaver
from mseb import task as task_lib
from mseb import types
import numpy as np
import pytest

scann_ops_pybind = pytest.importorskip('scann.scann_ops_pybind')
ScannSearcher = scann_ops_pybind.ScannSearcher

retrieval_encoder = pytest.importorskip('mseb.encoders.retrieval_encoder')
retrieval_evaluator = pytest.importorskip('mseb.evaluators.retrieval_evaluator')
retrieval_task = pytest.importorskip('mseb.tasks.retrieval')


class MockRetrievalTask(retrieval_task.RetrievalTask):

  @property
  def index_dir(self) -> str:
    return os.path.join(
        task_lib.TASK_CACHE_BASEPATH.value,
        'retrievals',
        'svq_passage_retrieval_in_lang',
    )

  def get_documents_source(self):
    return

  @staticmethod
  def documents_generator(not_used):
    del not_used
    return [
        types.Text(
            text='bli text',
            context=types.TextContextParams(id='bli', text='bli text'),
        ),
        types.Text(
            text='bla text',
            context=types.TextContextParams(id='bla', text='bla text'),
        ),
        types.Text(
            text='blo text',
            context=types.TextContextParams(id='blo', text='blo text'),
        ),
        types.Text(
            text='blu text',
            context=types.TextContextParams(id='blu', text='blu text'),
        ),
    ]

  def sounds(self) -> Iterable[types.Sound]:
    raise NotImplementedError('Not used in this test.')

  def examples(
      self, sub_task: str
  ) -> Iterable[retrieval_evaluator.RetrievalReferenceId]:
    raise NotImplementedError('Not used in this test.')

  @property
  def sub_tasks(self) -> list[str]:
    """Get the list of sub-tasks for the retrieval task."""
    raise NotImplementedError('Not used in this test.')


@pytest.mark.scann
@pytest.mark.optional
class RetrievalEncoderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, 'testdata'
    )

  def test_encode(self):
    self.enter_context(
        flagsaver.flagsaver((task_lib.TASK_CACHE_BASEPATH, self.testdata_path))
    )
    task = MockRetrievalTask()
    task.setup()

    encoder = retrieval_encoder.RetrievalEncoder(top_k=2)
    encoder.set_task(task)
    self.assertIsNotNone(encoder._index_dir)
    self.assertIsNotNone(encoder._documents)
    self.assertIsNone(encoder._text_by_id)
    encoder.setup()
    self.assertIsNotNone(encoder._evaluator)
    self.assertLen(encoder._text_by_id, 4)

    embeddings = encoder.encode([
        types.TextEmbedding(
            embedding=np.array([[1.0, 2.0, 3.0]]),
            spans=np.array([[0, 10]]),
            context=types.TextContextParams(id='1', text='blu text'),
        ),
        types.TextEmbedding(
            embedding=np.array([[1.0, 2.0, 3.0]]),
            spans=np.array([[0, 10]]),
            context=types.TextContextParams(id='2', text='blo text'),
        ),
    ])
    self.assertLen(embeddings, 2)
    embedding_0 = embeddings[0]
    self.assertIsInstance(embedding_0, types.TextWithTitleAndContext)
    self.assertEqual(embedding_0.text, 'blu text')
    self.assertEqual(
        embedding_0.context_text,
        '{"id": "blu", "text": "blu text"}\n{"id": "blo", "text": "blo text"}',
    )
    embedding_1 = embeddings[1]
    self.assertIsInstance(embedding_1, types.TextWithTitleAndContext)
    self.assertEqual(embedding_1.text, 'blo text')
    self.assertEqual(
        embedding_1.context_text,
        '{"id": "blu", "text": "blu text"}\n{"id": "blo", "text": "blo text"}',
    )

  def test_encode_with_partitioned_index(self):
    num_partitions = 2
    index_dir = self.create_tempdir().full_path
    source_base_dir = os.path.join(
        self.testdata_path, 'retrievals', 'svq_passage_retrieval_in_lang'
    )
    for partition_id in range(num_partitions):
      target_base_dir = os.path.join(
          index_dir,
          'retrievals',
          'svq_passage_retrieval_in_lang',
          str(partition_id),
      )
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
    self.enter_context(
        flagsaver.flagsaver((task_lib.TASK_CACHE_BASEPATH, index_dir))
    )
    self.enter_context(
        flagsaver.flagsaver((retrieval_task._NUM_PARTITIONS, num_partitions))
    )
    task = MockRetrievalTask()
    task.setup()

    encoder = retrieval_encoder.RetrievalEncoder(top_k=2)
    encoder.set_task(task)
    self.assertIsNotNone(encoder._index_dir)
    self.assertIsNotNone(encoder._documents)
    encoder.setup()
    self.assertIsNotNone(encoder._text_by_id)
    self.assertLen(encoder._text_by_id, 4)
    self.assertIsNotNone(encoder._evaluator)

    embeddings = encoder.encode([
        types.TextEmbedding(
            embedding=np.array([[1.0, 2.0, 3.0]]),
            spans=np.array([[0, 10]]),
            context=types.TextContextParams(id='1', text='blu text'),
        ),
        types.TextEmbedding(
            embedding=np.array([[1.0, 2.0, 3.0]]),
            spans=np.array([[0, 10]]),
            context=types.TextContextParams(id='2', text='blo text'),
        ),
    ])
    self.assertLen(embeddings, 2)
    embedding_0 = embeddings[0]
    self.assertIsInstance(embedding_0, types.TextWithTitleAndContext)
    self.assertEqual(embedding_0.text, 'blu text')
    self.assertEqual(
        embedding_0.context_text,
        '{"id": "blu", "text": "blu text"}\n{"id": "blo", "text": "blo text"}',
    )
    embedding_1 = embeddings[1]
    self.assertIsInstance(embedding_1, types.TextWithTitleAndContext)
    self.assertEqual(embedding_1.text, 'blo text')
    self.assertEqual(
        embedding_1.context_text,
        '{"id": "blu", "text": "blu text"}\n{"id": "blo", "text": "blo text"}',
    )


if __name__ == '__main__':
  absltest.main()
