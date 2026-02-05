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

from absl.testing import absltest
from mseb import types
import numpy as np
import numpy.testing as npt
import pytest


transcription_evaluator = pytest.importorskip(
    'mseb.evaluators.transcription_evaluator'
)


@pytest.mark.whisper
@pytest.mark.optional
class TranscriptionEvaluatorTest(absltest.TestCase):

  def test_compute_predictions(self):
    evaluator = transcription_evaluator.TranscriptionEvaluator()
    transcript_by_sound_id = evaluator.compute_predictions(
        embeddings_by_sound_id={
            'test': types.SoundEmbedding(
                embedding=np.array(['This is a test.']),
                timestamps=np.array([[0.0, 1.0]]),
                context=types.SoundContextParams(
                    id='test',
                    sample_rate=16000,
                    length=100,
                    language='en',
                ),
            ),
        },
    )
    self.assertLen(transcript_by_sound_id, 1)
    self.assertIn('test', transcript_by_sound_id)
    transcript = transcript_by_sound_id['test']
    self.assertIsInstance(transcript, types.TextPrediction)
    self.assertEqual(transcript.prediction, 'This is a test.')
    self.assertEqual(
        transcript.context,
        types.PredictionContextParams(id='test'),
    )

  def test_compute_metrics(self):
    evaluator = transcription_evaluator.TranscriptionEvaluator()
    scores = evaluator.compute_metrics(
        transcript_by_sound_id={
            'test': types.TextPrediction(
                prediction='This is a test.',
                context=types.PredictionContextParams(id='test'),
            )
        },
        transcript_truths=[
            transcription_evaluator.TranscriptTruth(
                sound_id='test',
                text='This is a test.',
                language='en',
            ),
        ],
    )
    npt.assert_equal(len(scores), 3)
    self.assertIn('WER', scores[0].metric)
    npt.assert_equal(scores[0].value, 0)
    npt.assert_equal(scores[0].std, 0)
    self.assertIn('SER', scores[1].metric)
    npt.assert_equal(scores[1].value, 0)
    npt.assert_equal(scores[1].std, 0)
    self.assertIn('NoResultRate', scores[2].metric)
    npt.assert_equal(scores[2].value, 0)
    npt.assert_equal(scores[2].std, 0)


if __name__ == '__main__':
  absltest.main()
