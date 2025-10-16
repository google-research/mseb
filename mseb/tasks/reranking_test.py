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
from absl.testing import flagsaver
from mseb import runner as runner_lib
from mseb import types
from mseb.encoders import normalized_text_encoder_with_prompt as text_encoder
import numpy as np
import pytest

reranking = pytest.importorskip('mseb.tasks.reranking')
reranking_evaluator = pytest.importorskip('mseb.evaluators.reranking_evaluator')


@pytest.mark.whisper
@pytest.mark.optional
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
                sound_id='utt_11697423627206642872',
                texts=['ref_1A', 'ref_1B'],
                language='en',
            ),
            reranking_evaluator.RerankingCandidates(
                sound_id='utt_15041124811443622614',
                texts=['ref_2A', 'ref_2B', 'ref_2C'],
                language='en',
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
        'ref_1A': types.TextEmbedding(
            embedding=np.zeros((1, 3)),
            spans=np.zeros((1, 2)),
            context=types.TextContextParams(id='ref_1A'),
        ),
        'ref_1B': types.TextEmbedding(
            embedding=np.ones((1, 3)),
            spans=np.zeros((1, 2)),
            context=types.TextContextParams(id='ref_1B'),
        ),
        'ref_2A': types.TextEmbedding(
            embedding=np.zeros((1, 3)),
            spans=np.zeros((1, 2)),
            context=types.TextContextParams(id='ref_2A'),
        ),
        'ref_2B': types.TextEmbedding(
            embedding=np.ones((1, 3)),
            spans=np.zeros((1, 2)),
            context=types.TextContextParams(id='ref_2B'),
        ),
        'ref_2C': types.TextEmbedding(
            embedding=np.ones((1, 3)),
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
    self.enter_context(
        flagsaver.flagsaver((reranking.task.TASK_CACHE_BASEPATH, cache_dir))
    )

    task = MockRerankingTask()
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
                    text='ref_1A',
                    context=types.TextContextParams(id='ref_1A'),
                ),
                types.Text(
                    text='ref_1B',
                    context=types.TextContextParams(id='ref_1B'),
                ),
            ],
            [
                types.Text(
                    text='ref_2A',
                    context=types.TextContextParams(id='ref_2A'),
                ),
                types.Text(
                    text='ref_2B',
                    context=types.TextContextParams(id='ref_2B'),
                ),
                types.Text(
                    text='ref_2C',
                    context=types.TextContextParams(id='ref_2C'),
                ),
            ],
        ]

      def sounds(self) -> Iterable[types.Sound]:
        raise NotImplementedError()

      def examples(
          self, sub_task: str
      ) -> Iterable[reranking_evaluator.RerankingCandidates]:
        assert sub_task == 'test'
        return [
            reranking_evaluator.RerankingCandidates(
                sound_id='utt_11697423627206642872',
                texts=['ref_1A', 'ref_1B'],
                language='en',
            ),
            reranking_evaluator.RerankingCandidates(
                sound_id='utt_15041124811443622614',
                texts=['ref_2A', 'ref_2B', 'ref_2C'],
                language='en',
            ),
        ]

      @property
      def sub_tasks(self) -> list[str]:
        return ['test']

    self.enter_context(
        flagsaver.flagsaver((
            reranking.task.TASK_CACHE_BASEPATH,
            self.create_tempdir().full_path,
        ))
    )
    task = MockRerankingTask()
    task.setup(
        runner=runner_lib.DirectRunner(encoder=text_encoder.MockTextEncoder())
    )
    self.assertIsNotNone(task._evaluator)
    self.assertIsNotNone(task._evaluator.candidate_embeddings_by_sound_id)
    self.assertLen(task._evaluator.candidate_embeddings_by_sound_id, 2)
    self.assertEqual(task._evaluator.mrr_at_k, 10)


if __name__ == '__main__':
  absltest.main()
