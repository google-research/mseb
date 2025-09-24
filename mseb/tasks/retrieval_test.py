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
from typing import Iterable

from absl.testing import absltest
from absl.testing import flagsaver
from mseb import runner as runner_lib
from mseb import types
from mseb.evaluators import retrieval_evaluator
from mseb.tasks import retrieval
import numpy as np


class RetrievalTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent,
        'testdata',
    )

  def test_retrieval_task_compute_scores(self):

    class MockRetrievalTask(retrieval.RetrievalTask):

      @property
      def index_dir(self) -> str:
        return os.path.join(super().index_dir, 'svq_passage_retrieval_in_lang')

      def sounds(self) -> Iterable[types.Sound]:
        return [
            types.Sound(
                waveform=np.zeros(16000),
                context=types.SoundContextParams(
                    sample_rate=16000,
                    length=10,
                    id='utt_11697423627206642872',
                ),
            ),
            types.Sound(
                waveform=np.ones(16000),
                context=types.SoundContextParams(
                    sample_rate=16000,
                    length=10,
                    id='utt_15041124811443622614',
                ),
            ),
        ]

      def examples(
          self, sub_task: str
      ) -> Iterable[retrieval_evaluator.RetrievalReferenceId]:
        return [
            retrieval_evaluator.RetrievalReferenceId(
                sound_id='utt_11697423627206642872', reference_id='ref_1'
            ),
            retrieval_evaluator.RetrievalReferenceId(
                sound_id='utt_15041124811443622614', reference_id='ref_2'
            ),
        ]

      def documents(self) -> Iterable[types.Text]:
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
            ),
            embedding=np.zeros((1, 3)),
            timestamps=np.zeros((1, 2)),
        ),
        'utt_15041124811443622614': types.SoundEmbedding(
            context=types.SoundContextParams(
                sample_rate=16000,
                length=10,
                id='utt_15041124811443622614',
            ),
            embedding=np.ones((1, 3)),
            timestamps=np.zeros((1, 2)),
        ),
    }

    self.enter_context(
        flagsaver.flagsaver((retrieval.task.CACHE_BASEPATH, self.testdata_path))
    )
    task = MockRetrievalTask()
    task.setup()
    self.assertEqual(task.sub_tasks, ['test'])
    scores = task.compute_scores(embeddings=embeddings)
    self.assertLen(scores, 1)
    self.assertIn('test', scores)
    self.assertEqual(scores['test'][0].metric, 'MRR')
    self.assertEqual(scores['test'][1].metric, 'EM')

  def test_retrieval_task_setup(self):

    class MockRetrievalTask(retrieval.RetrievalTask):

      def documents(self) -> Iterable[types.Text]:
        return [
            types.Text(
                text='dummy text',
                context=types.TextContextParams(
                    id=str(i),
                ),
            )
            for i in range(16)
        ]

      def sounds(self) -> Iterable[types.Sound]:
        raise NotImplementedError()

      def examples(
          self, sub_task: str
      ) -> Iterable[retrieval_evaluator.RetrievalReferenceId]:
        raise NotImplementedError()

      @property
      def sub_tasks(self) -> list[str]:
        return ['not_used']

    self.enter_context(
        flagsaver.flagsaver(
            (retrieval.task.CACHE_BASEPATH, self.create_tempdir().full_path)
        )
    )
    task = MockRetrievalTask(text_encoder_name='mock_text')
    task.setup(runner_cls=runner_lib.DirectRunner)
    self.assertIsNotNone(task._evaluator)
    self.assertIsNotNone(task._evaluator.searcher)
    self.assertLen(task._evaluator.id_by_index_id, 16)


if __name__ == '__main__':
  absltest.main()
