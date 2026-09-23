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
                timestamps=np.array([[0.0, 1.0]]),  # pyrefly: ignore[bad-argument-type]
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
                prediction='This is toast.',
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
    npt.assert_equal(len(scores), 5)
    self.assertIn('WER', scores[0].metric)
    npt.assert_equal(scores[0].value, 2 / 4)
    npt.assert_equal(scores[0].std, 0)
    self.assertIn('SER', scores[1].metric)
    npt.assert_equal(scores[1].value, 1)
    npt.assert_equal(scores[1].std, 0)
    self.assertIn('NoResultRate', scores[2].metric)
    npt.assert_equal(scores[2].value, 0)
    npt.assert_equal(scores[2].std, 0)
    self.assertIn('UtteranceCount', scores[3].metric)
    npt.assert_equal(scores[3].value, 1)
    self.assertIn('WordCount', scores[4].metric)
    npt.assert_equal(scores[4].value, 4)

  def test_compute_metrics_with_empty_transcript_truth(self):
    evaluator = transcription_evaluator.TranscriptionEvaluator()
    scores = evaluator.compute_metrics(
        transcript_by_sound_id={
            'test1': types.TextPrediction(
                prediction='This is a toast.',
                context=types.PredictionContextParams(id='test1'),
            ),
            'test2': types.TextPrediction(
                prediction='',
                context=types.PredictionContextParams(id='test2'),
            ),
            'test3': types.TextPrediction(
                prediction='This should be empty.',
                context=types.PredictionContextParams(id='test3'),
            ),
        },
        transcript_truths=[
            transcription_evaluator.TranscriptTruth(
                sound_id='test1',
                text='This is a test.',
                language='en',
            ),
            transcription_evaluator.TranscriptTruth(
                sound_id='test2',
                text='',
                language='en',
            ),
            transcription_evaluator.TranscriptTruth(
                sound_id='test3',
                text='',
                language='en',
            ),
        ],
    )
    npt.assert_equal(len(scores), 5)
    self.assertIn('WER', scores[0].metric)
    npt.assert_equal(scores[0].value, 5 / 4)
    npt.assert_equal(scores[0].std, 1)
    self.assertIn('SER', scores[1].metric)
    npt.assert_equal(scores[1].value, 2 / 3)
    npt.assert_equal(scores[1].std**2, 2 / 9)
    self.assertIn('NoResultRate', scores[2].metric)
    npt.assert_equal(scores[2].value, 0)
    npt.assert_equal(scores[2].std, 0)
    self.assertIn('UtteranceCount', scores[3].metric)
    npt.assert_equal(scores[3].value, 3)
    self.assertIn('WordCount', scores[4].metric)
    npt.assert_equal(scores[4].value, 4)

  def test_compute_metrics_cjk(self):
    evaluator = transcription_evaluator.TranscriptionEvaluator()

    # Test Chinese (cmn_hans_cn / cmn-hans-cn) - TC2SC and space removal
    scores_zh = evaluator.compute_metrics(
        transcript_by_sound_id={
            'zh_test': types.TextPrediction(
                prediction='繁体字 错误',
                context=types.PredictionContextParams(id='zh_test'),
            )
        },
        transcript_truths=[
            transcription_evaluator.TranscriptTruth(
                sound_id='zh_test',
                text='繁體字 測試。',
                language='cmn-hans-cn',
            ),
        ],
    )
    # Truth:         "繁體字 測試。" ->
    # normalized:    "繁體字 測試" ->
    # TC2SC:         "繁体字 测试" ->
    # space-removal: "繁体字测试" (len 5)
    # Hyp:           "繁体字 错误" ->
    # normalized:    "繁体字 错误" ->
    # space-removal: "繁体字错误" (len 5)
    # Edits: 2 substitutions (测->错, 试->误) -> CER = 2/5 = 0.4
    # We expect 5 baseline scores + 2 CJK scores = 7 scores
    self.assertLen(scores_zh, 7)
    self.assertEqual(scores_zh[5].metric, 'CER')
    self.assertAlmostEqual(scores_zh[5].value, 2 / 5)
    self.assertEqual(scores_zh[6].metric, 'CharCount')
    self.assertEqual(scores_zh[6].value, 5.0)

    # Test Japanese (ja_jp) - Space removal, no TC2SC
    scores_ja = evaluator.compute_metrics(
        transcript_by_sound_id={
            'ja_test': types.TextPrediction(
                prediction='日本語 てすと',
                context=types.PredictionContextParams(id='ja_test'),
            )
        },
        transcript_truths=[
            transcription_evaluator.TranscriptTruth(
                sound_id='ja_test',
                text='日本語 テスト。',
                language='ja_jp',
            ),
        ],
    )
    # Truth:         "日本語 テスト。" ->
    # normalized:    "日本語 テスト" ->
    # space-removal: "日本語テスト" (len 6)
    # Hyp:           "日本語 てすと" ->
    # normalized:    "日本語 てすと" ->
    # space-removal: "日本語てすと" (len 6)
    # Edits: 3 substitutions (テ->て, ス->す, ト->と) -> CER = 3/6 = 0.5
    self.assertLen(scores_ja, 7)
    self.assertEqual(scores_ja[0].metric, 'WER')
    self.assertEqual(scores_ja[1].metric, 'SER')
    self.assertEqual(scores_ja[5].metric, 'CER')
    self.assertAlmostEqual(scores_ja[5].value, 3 / 6)
    self.assertEqual(scores_ja[6].metric, 'CharCount')
    self.assertEqual(scores_ja[6].value, 6.0)

    # Test Korean (ko_kr) - No space removal, no TC2SC
    scores_ko = evaluator.compute_metrics(
        transcript_by_sound_id={
            'ko_test': types.TextPrediction(
                prediction='한국어 태스트',
                context=types.PredictionContextParams(id='ko_test'),
            )
        },
        transcript_truths=[
            transcription_evaluator.TranscriptTruth(
                sound_id='ko_test',
                text='한국어 테스트。',
                language='ko_kr',
            ),
        ],
    )
    # Truth:      "한국어 테스트。" ->
    # normalized: "한국어 테스트" (len 7 space: 한, 국, 어,  , 테, 스, 트)
    # Hyp:        "한국어 태스트" ->
    # normalized: "한국어 태스트" (len 7 including space)
    # Edits: 1 substitution (테->태) -> CER = 1/7
    self.assertLen(scores_ko, 7)
    self.assertEqual(scores_ko[0].metric, 'WER')
    self.assertEqual(scores_ko[1].metric, 'SER')
    self.assertEqual(scores_ko[5].metric, 'CER')
    self.assertAlmostEqual(scores_ko[5].value, 1 / 7)
    self.assertEqual(scores_ko[6].metric, 'CharCount')
    self.assertEqual(scores_ko[6].value, 7.0)

  def test_compute_metrics_non_cjk(self):
    evaluator = transcription_evaluator.TranscriptionEvaluator()
    # Non-CJK should not have CER and CharCount
    scores = evaluator.compute_metrics(
        transcript_by_sound_id={
            'test': types.TextPrediction(
                prediction='This is toast.',
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
    self.assertLen(scores, 5)
    # WER, SER, NoResultRate, UtteranceCount, WordCount
    self.assertEqual(scores[0].metric, 'WER')
    self.assertEqual(scores[1].metric, 'SER')


if __name__ == '__main__':
  absltest.main()
