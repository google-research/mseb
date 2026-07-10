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
from typing import Iterable, Sequence

from absl.testing import absltest
from absl.testing import flagsaver
from mseb import runner as runner_lib
from mseb import types
from mseb.evaluators import reasoning_evaluator
from mseb.tasks import reasoning
import numpy as np
import pytest

text_encoder_with_prompt = pytest.importorskip(
    'mseb.encoders.text_encoder_with_prompt'
)


@pytest.mark.gecko
@pytest.mark.optional
class ReasoningTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        flagsaver.flagsaver((
            reasoning.task.TASK_CACHE_BASEPATH,
            self.create_tempdir().full_path,
        ))
    )

  def test_reasoning_task_compute_scores(self):

    class MockReasoningTask(reasoning.ReasoningTask):

      @property
      def embeddings_dir(self) -> str:
        return os.path.join(
            super().embeddings_dir, 'svq_en_us_span_reasoning_in_lang'
        )

      def multimodal_inputs(self) -> Iterable[types.Sound]:
        raise NotImplementedError()

      def examples(
          self, sub_task: str
      ) -> Iterable[reasoning_evaluator.ReasoningSpans]:
        return [
            reasoning_evaluator.ReasoningSpans(
                sound_id='utt_11697423627206642872',
                texts=['ref_1A', 'ref_1B'],
                reference_answer=reasoning_evaluator.NO_ANSWER_STR,
            ),
            reasoning_evaluator.ReasoningSpans(
                sound_id='utt_15041124811443622614',
                texts=['ref_2A', 'ref_2B', 'ref_2C'],
                reference_answer='ref_2A',
            ),
        ]

      def span_lists(self) -> Iterable[Sequence[types.Text]]:
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

    span_embeddings = {
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
    runner_lib.save_embeddings(
        output_prefix=os.path.join(
            reasoning.task.TASK_CACHE_BASEPATH.value,
            'reasonings',
            'svq_en_us_span_reasoning_in_lang',
            'embeddings',
        ),
        embeddings=span_embeddings,
    )

    task = MockReasoningTask()
    task.setup()
    self.assertEqual(task.sub_tasks, ['test'])
    scores = task.compute_scores(embeddings=embeddings)
    self.assertLen(scores, 1)
    self.assertIn('test', scores)
    self.assertEqual(scores['test'][0].metric, 'GmeanF1')

  def test_reasoning_task_setup(self):

    class MockReasoningTask(reasoning.ReasoningTask):

      def span_lists(self) -> Iterable[Sequence[types.Text]]:
        return [
            [
                types.Text(
                    text=f'dummy span {i}',
                    context=types.TextContextParams(
                        id=f'dummy span {i}',
                    ),
                )
                for i in range(3)
            ]
            for _ in range(2)
        ]

      def multimodal_inputs(self) -> Iterable[types.Sound]:
        raise NotImplementedError()

      def examples(
          self, sub_task: str
      ) -> Iterable[reasoning_evaluator.ReasoningSpans]:
        assert sub_task == 'test'
        return [
            reasoning_evaluator.ReasoningSpans(
                sound_id='sound_1',
                reference_answer=reasoning_evaluator.NO_ANSWER_STR,
                texts=['dummy span 0', 'dummy span 1', 'dummy span 2'],
            ),
            reasoning_evaluator.ReasoningSpans(
                sound_id='sound_2',
                reference_answer='ref_2A',
                texts=['dummy span 1', 'dummy span 2', 'dummy span 0'],
            ),
        ]

      @property
      def sub_tasks(self) -> list[str]:
        return ['test']

    task = MockReasoningTask()
    task.setup(
        runner=runner_lib.DirectRunner(
            encoder=text_encoder_with_prompt.MockTextEncoder()
        )
    )
    self.assertIsNotNone(task._evaluator)
    self.assertIsNotNone(task._evaluator.span_embeddings_by_sound_id)
    self.assertLen(task._evaluator.span_embeddings_by_sound_id, 2)

  def test_reasoning_task_compute_scores_with_prediction_output(self):

    class MockReasoningTask(reasoning.ReasoningTask):

      @property
      def embeddings_dir(self) -> str:
        raise FileNotFoundError()  # setup() logic relies on this exception.

      def multimodal_inputs(self) -> Iterable[types.Sound]:
        raise NotImplementedError()

      def examples(
          self, sub_task: str
      ) -> Iterable[reasoning_evaluator.ReasoningSpans]:
        return [
            reasoning_evaluator.ReasoningSpans(
                sound_id='utt_11697423627206642872',
                texts=['ref_1A', 'ref_1B'],
                reference_answer=reasoning_evaluator.NO_ANSWER_STR,
            ),
            reasoning_evaluator.ReasoningSpans(
                sound_id='utt_15041124811443622614',
                texts=['ref_2A', 'ref_2B', 'ref_2C'],
                reference_answer='ref_2A',
            ),
        ]

      def span_lists(self) -> Iterable[Sequence[types.Text]]:
        raise NotImplementedError()

      @property
      def sub_tasks(self) -> list[str]:
        return ['test']

    embeddings = {
        'utt_11697423627206642872': types.TextPrediction(
            prediction='ref_1A',
            context=types.PredictionContextParams(
                id='utt_11697423627206642872',
            ),
        ),
        'utt_15041124811443622614': types.TextPrediction(
            prediction='ref_2A',
            context=types.PredictionContextParams(
                id='utt_15041124811443622614',
            ),
        ),
    }

    task = MockReasoningTask()
    task.setup()
    self.assertEqual(task.sub_tasks, ['test'])
    scores = task.compute_scores(embeddings=embeddings)
    self.assertLen(scores, 1)
    self.assertIn('test', scores)
    self.assertEqual(scores['test'][0].metric, 'GmeanF1')

  def test_multimodal_objects_for_setup_yields_spans(self):

    class MockReasoningTask(reasoning.ReasoningTask):

      def span_lists(self):
        return [
            [
                types.Text(
                    text='span_1A',
                    context=types.TextContextParams(id='span_1A'),
                ),
                types.Text(
                    text='span_1B',
                    context=types.TextContextParams(id='span_1B'),
                ),
            ],
            [
                types.Text(
                    text='span_2A',
                    context=types.TextContextParams(id='span_2A'),
                ),
            ],
        ]

      def multimodal_inputs(self):
        raise NotImplementedError()

      def examples(self, sub_task):
        raise NotImplementedError()

      @property
      def sub_tasks(self):
        return ['test']

    task = MockReasoningTask()
    objects = list(task.multimodal_objects_for_setup())
    self.assertLen(objects, 3)
    self.assertEqual(objects[0].text, 'span_1A')
    self.assertEqual(objects[2].text, 'span_2A')

  def test_setup_with_embeddings_cache(self):

    class MockReasoningTask(reasoning.ReasoningTask):

      def span_lists(self):
        return [
            [
                types.Text(
                    text='span_A',
                    context=types.TextContextParams(id='span_A'),
                ),
                types.Text(
                    text='span_B',
                    context=types.TextContextParams(id='span_B'),
                ),
            ],
        ]

      def multimodal_inputs(self):
        raise NotImplementedError()

      def examples(self, sub_task):
        return [
            reasoning_evaluator.ReasoningSpans(
                sound_id='sound_1',
                texts=['span_A', 'span_B'],
                reference_answer='span_A',
            ),
        ]

      @property
      def sub_tasks(self):
        return ['test']

    embeddings_cache = {
        'span_A': types.TextEmbedding(
            embedding=np.zeros((1, 3)),
            spans=np.zeros((1, 2)),
            context=types.TextContextParams(id='span_A'),
        ),
        'span_B': types.TextEmbedding(
            embedding=np.ones((1, 3)),
            spans=np.zeros((1, 2)),
            context=types.TextContextParams(id='span_B'),
        ),
    }
    task = MockReasoningTask()
    task.setup(embeddings_cache=embeddings_cache)
    self.assertIsNotNone(task._evaluator)
    self.assertIsNotNone(task._evaluator.span_embeddings_by_sound_id)
    self.assertLen(task._evaluator.span_embeddings_by_sound_id, 1)


if __name__ == '__main__':
  absltest.main()
