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
from typing import Iterable, Sequence

from absl.testing import absltest
from mseb import runner as runner_lib
from mseb import types
from mseb.evaluators import reranking_evaluator
from mseb.tasks import reranking
import numpy as np


class RerankingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent,
        'testdata',
    )

  def test_reranking_task_compute_scores(self):

    class MockRerankingTask(reranking.RerankingTask):

      @property
      def embeddings_dir(self) -> str:
        return os.path.join(super().embeddings_dir, 'svq_en_us_query_reranking')

      def sounds(self) -> Iterable[types.Sound]:
        raise NotImplementedError()

      def examples(
          self, sub_task: str
      ) -> Iterable[reranking_evaluator.RerankingCandidates]:
        return [
            reranking_evaluator.RerankingCandidates(
                sound_id='utt_11697423627206642872', texts=['ref_1A', 'ref_1B']
            ),
            reranking_evaluator.RerankingCandidates(
                sound_id='utt_15041124811443622614',
                texts=['ref_2A', 'ref_2B', 'ref_2C'],
            ),
        ]

      def candidate_lists(self) -> Iterable[Sequence[types.Text]]:
        raise NotImplementedError()

      @property
      def sub_tasks(self) -> list[str]:
        return ['test']

    embeddings = {
        'utt_11697423627206642872': types.SoundEmbedding(
            context=types.SoundContextParams(
                sample_rate=16000,
                length=10,
                id='utt_11697423627206642872',
                language='en',
            ),
            embedding=np.zeros((1, 3)),
            timestamps=np.zeros((1, 2)),
        ),
        'utt_15041124811443622614': types.SoundEmbedding(
            context=types.SoundContextParams(
                sample_rate=16000,
                length=10,
                id='utt_15041124811443622614',
                language='en',
            ),
            embedding=np.ones((1, 3)),
            timestamps=np.zeros((1, 2)),
        ),
    }

    candidate_embeddings = {
        'ref_1A': types.TextEmbeddings(
            embeddings=np.zeros((1, 3)),
            spans=np.zeros((1, 2)),
            context=types.TextContextParams(id='ref_1A'),
        ),
        'ref_1B': types.TextEmbeddings(
            embeddings=np.ones((1, 3)),
            spans=np.zeros((1, 2)),
            context=types.TextContextParams(id='ref_1B'),
        ),
        'ref_2A': types.TextEmbeddings(
            embeddings=np.zeros((1, 3)),
            spans=np.zeros((1, 2)),
            context=types.TextContextParams(id='ref_2A'),
        ),
        'ref_2B': types.TextEmbeddings(
            embeddings=np.ones((1, 3)),
            spans=np.zeros((1, 2)),
            context=types.TextContextParams(id='ref_2B'),
        ),
        'ref_2C': types.TextEmbeddings(
            embeddings=np.ones((1, 3)),
            spans=np.zeros((1, 2)),
            context=types.TextContextParams(id='ref_2C'),
        ),
    }
    cache_dir = self.create_tempdir().full_path
    runner_lib.save_embeddings(
        output_prefix=os.path.join(
            cache_dir, 'rerankings', 'svq_en_us_query_reranking', 'embeddings'
        ),
        embeddings=candidate_embeddings,
    )

    task = MockRerankingTask(cache_dir=cache_dir)
    task.setup()
    self.assertEqual(task.sub_tasks, ['test'])
    scores = task.compute_scores(embeddings=embeddings)
    self.assertLen(scores, 1)
    self.assertIn('test', scores)
    self.assertEqual(scores['test'][0].metric, 'MAP')
    self.assertEqual(scores['test'][1].metric, 'WER')
    self.assertEqual(scores['test'][2].metric, 'CER')
    self.assertEqual(scores['test'][3].metric, 'MRR')

  def test_reranking_task_setup(self):

    class MockRerankingTask(reranking.RerankingTask):

      def candidate_lists(self) -> Iterable[Sequence[types.Text]]:
        return [
            [
                types.Text(
                    text=f'dummy text {i}',
                    context=types.TextContextParams(
                        id=str(i),
                    ),
                )
                for i in range(3)
            ]
            for _ in range(2)
        ]

      def sounds(self) -> Iterable[types.Sound]:
        raise NotImplementedError()

      def examples(
          self, sub_task: str
      ) -> Iterable[reranking_evaluator.RerankingCandidates]:
        raise NotImplementedError()

      @property
      def sub_tasks(self) -> list[str]:
        return ['not_used']

    task = MockRerankingTask(
        cache_dir=self.create_tempdir().full_path, text_encoder_name='mock_text'
    )
    task.setup(runner_cls=runner_lib.DirectRunner)
    self.assertIsNotNone(task._evaluator)
    self.assertIsNotNone(task._evaluator.candidate_embeddings_by_text)
    self.assertLen(task._evaluator.candidate_embeddings_by_text, 3)
    self.assertEqual(task._evaluator.mrr_at_k, 10)


if __name__ == '__main__':
  absltest.main()
