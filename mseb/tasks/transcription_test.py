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

from typing import Iterable

from absl.testing import absltest
from mseb import types
import numpy as np
import pytest


transcription = pytest.importorskip('mseb.tasks.transcription')
transcription_evaluator = pytest.importorskip(
    'mseb.evaluators.transcription_evaluator'
)


class MockTranscriptionTask(transcription.TranscriptionTask):

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
  ) -> Iterable[transcription_evaluator.TranscriptTruth]:
    return [
        transcription_evaluator.TranscriptTruth(
            sound_id='utt_11697423627206642872',
            text='This is a test.',
            language='en_us',
        ),
        transcription_evaluator.TranscriptTruth(
            sound_id='utt_15041124811443622614',
            text='This is another test.',
            language='en_us',
        ),
    ]

  @property
  def sub_tasks(self) -> list[str]:
    return ['test']


@pytest.mark.whisper
@pytest.mark.optional
class RetrievalTest(absltest.TestCase):

  def test_transcription_task_compute_scores(self):
    embeddings = {
        'utt_11697423627206642872': types.SoundEmbedding(
            context=types.SoundContextParams(
                sample_rate=16000,
                length=10,
                id='utt_11697423627206642872',
            ),
            embedding=np.array(['This is a test.']),
            timestamps=np.zeros((1, 2)),
        ),
        'utt_15041124811443622614': types.SoundEmbedding(
            context=types.SoundContextParams(
                sample_rate=16000,
                length=10,
                id='utt_15041124811443622614',
            ),
            embedding=np.array(['This is a different test.']),
            timestamps=np.zeros((1, 2)),
        ),
    }

    task = MockTranscriptionTask()
    task.setup()
    self.assertEqual(task.sub_tasks, ['test'])
    scores = task.compute_scores(embeddings=embeddings)
    self.assertLen(scores, len(task.sub_tasks))
    self.assertIn('test', scores)
    self.assertEqual(scores['test'][0].metric, 'WER')
    self.assertEqual(scores['test'][1].metric, 'SER')
    self.assertEqual(scores['test'][2].metric, 'NoResultRate')

  def test_transcription_task_setup(self):
    task = MockTranscriptionTask()
    task.setup()
    self.assertIsNotNone(task._evaluator)


if __name__ == '__main__':
  absltest.main()
