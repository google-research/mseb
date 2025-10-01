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
from mseb.evaluators import classification_evaluator
from mseb.tasks import classification
import numpy as np


class ClassificationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent,
        'testdata',
    )

  def test_classification_task_compute_scores(self):

    class MockClassificationTask(classification.ClassificationTask):

      @property
      def weights_dir(self) -> str:
        return os.path.join(
            super().weights_dir, 'speech_massive_en_us_intent_classification'
        )

      def sounds(self) -> Iterable[types.Sound]:
        raise NotImplementedError()

      def examples(
          self, sub_task: str
      ) -> Iterable[classification_evaluator.ClassificationReference]:
        return [
            classification_evaluator.ClassificationReference(
                example_id='utt_11697423627206642872',
                label_id='label_1',
            ),
            classification_evaluator.ClassificationReference(
                example_id='utt_15041124811443622614',
                label_id='label_2',
            ),
        ]

      def class_labels(self) -> Iterable[Sequence[types.Text]]:
        return ['label_1', 'label_2', 'label_3']

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
            embedding=np.zeros((1, 2)),
            timestamps=np.zeros((1, 2)),
        ),
        'utt_15041124811443622614': types.SoundEmbedding(
            context=types.SoundContextParams(
                sample_rate=16000,
                length=10,
                id='utt_15041124811443622614',
                language='en',
            ),
            embedding=np.ones((1, 2)),
            timestamps=np.zeros((1, 2)),
        ),
    }

    cache_dir = self.create_tempdir().full_path
    weights = np.ones((3, 2 + 1))
    classification_evaluator.save_linear_classifier(
        MockClassificationTask().class_labels(),
        weights,
        os.path.join(
            cache_dir,
            'classifications',
            'speech_massive_en_us_intent_classification',
        ),
    )

    self.enter_context(
        flagsaver.flagsaver(
            (classification.task.TASK_CACHE_BASEPATH, cache_dir)
        )
    )

    task = MockClassificationTask()
    task.setup()
    self.assertEqual(task.sub_tasks, ['test'])
    scores = task.compute_scores(embeddings=embeddings)
    self.assertLen(scores, 1)
    self.assertIn('test', scores)
    self.assertLen(scores['test'], 6)
    self.assertEqual(scores['test'][0].metric, 'Accuracy')
    self.assertEqual(scores['test'][1].metric, 'Top-5 Accuracy')
    self.assertEqual(scores['test'][2].metric, 'Balanced Accuracy')
    self.assertEqual(scores['test'][3].metric, 'Weighted Precision')
    self.assertEqual(scores['test'][4].metric, 'Weighted Recall')
    self.assertEqual(scores['test'][5].metric, 'Weighted F1-Score')

  def test_classification_task_setup(self):

    class MockClassificationTask(classification.ClassificationTask):

      def class_labels(self) -> Iterable[Sequence[types.Text]]:
        return ['label_1', 'label_2', 'label_3']

      def sounds(self) -> Iterable[types.Sound]:
        raise NotImplementedError()

      def examples(
          self, sub_task: str
      ) -> Iterable[classification_evaluator.ClassificationReference]:
        raise NotImplementedError()

      @property
      def sub_tasks(self) -> list[str]:
        return ['test']

    self.enter_context(
        flagsaver.flagsaver((
            classification.task.TASK_CACHE_BASEPATH,
            self.create_tempdir().full_path,
        ))
    )
    task = MockClassificationTask()
    task.setup(
        runner=runner_lib.DirectRunner(encoder=text_encoder.MockTextEncoder())
    )
    self.assertIsNotNone(task._evaluator)
    self.assertIsNotNone(task._evaluator.id_by_label)
    self.assertLen(task._evaluator.id_by_label, 3)
    self.assertIsNotNone(task._evaluator.weights)
    self.assertEqual(task._evaluator.top_k_value, 5)


if __name__ == '__main__':
  absltest.main()
