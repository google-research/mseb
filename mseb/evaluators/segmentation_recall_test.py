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

"""Detection average precision must account for missed reference segments."""

from absl.testing import absltest
from absl.testing import parameterized
from mseb import types
from mseb.evaluators import segmentation_evaluator as se
import numpy as np


def detection_ap(ground_truths, predictions):
  result = se.SegmentationScoringResult(
      per_example_scores=[],
      all_predictions_for_map=list(predictions),
      ground_truths_for_map=ground_truths,
  )
  scores = se.SegmentationEvaluator(tau=0.1).compute_metrics(result)
  return next(score.value for score in scores if score.metric == 'mAP')


class SegmentationRecallTest(parameterized.TestCase):

  @parameterized.parameters(2, 5, 10)
  def test_one_correct_detection_has_partial_recall(self, count):
    references = [
        se.Segment(str(i), 2.0 * i, 2.0 * i + 1) for i in range(count)
    ]
    self.assertAlmostEqual(
        detection_ap({'a': references}, [('a', references[0])]), 1 / count
    )

  @parameterized.parameters(False, True)
  def test_precision_recall_steps_use_all_reference_segments(self, tied):
    refs = [
        se.Segment('a', 0.0, 1.0),
        se.Segment('b', 2.0, 3.0),
        se.Segment('c', 4.0, 5.0),
    ]
    predictions = [
        ('x', se.Segment('a', 0.0, 1.0, 0.9)),
        ('x', se.Segment('wrong', 6.0, 7.0, 0.9 if tied else 0.8)),
        ('x', se.Segment('b', 2.0, 3.0, 0.7)),
    ]
    expected = ((0.5 if tied else 1.0) + 2 / 3) / 3
    self.assertAlmostEqual(detection_ap({'x': refs}, predictions), expected)

  def test_duplicate_detections_do_not_recover_a_missed_reference(self):
    refs = [se.Segment('a', 0.0, 1.0), se.Segment('b', 2.0, 3.0)]
    predictions = [
        ('x', se.Segment('a', 0.0, 1.0, 0.9)),
        ('x', se.Segment('a', 0.0, 1.0, 0.8)),
    ]
    self.assertAlmostEqual(detection_ap({'x': refs}, predictions), 0.5)

  def test_examples_without_predictions_still_contribute_references(self):
    refs = {
        'detected': [se.Segment('a', 0.0, 1.0)],
        'missed': [se.Segment('b', 2.0, 3.0), se.Segment('c', 4.0, 5.0)],
    }
    self.assertAlmostEqual(
        detection_ap(refs, [('detected', refs['detected'][0])]), 1 / 3
    )

  def test_full_recall_keeps_existing_false_positive_penalty(self):
    refs = {'x': [se.Segment('a', 0.0, 1.0), se.Segment('b', 2.0, 3.0)]}
    predictions = [
        ('x', se.Segment('a', 0.0, 1.0, 0.9)),
        ('x', se.Segment('wrong', 6.0, 7.0, 0.8)),
        ('x', se.Segment('b', 2.0, 3.0, 0.7)),
    ]
    self.assertAlmostEqual(detection_ap(refs, predictions), 5 / 6)

  @parameterized.parameters('empty_predictions', 'empty_references', 'no_match')
  def test_no_matched_reference_stays_zero(self, case):
    refs = {'x': [se.Segment('a', 0.0, 1.0)]}
    preds = [('x', se.Segment('wrong', 6.0, 7.0))]
    if case == 'empty_predictions':
      preds = []
    if case == 'empty_references':
      refs = {}
    self.assertEqual(detection_ap(refs, preds), 0.0)

  def test_public_scoring_pipeline_preserves_missing_detection_denominator(
      self,
  ):
    references = [
        se.SegmentationReference(
            'x', [se.Segment('a', 0.0, 1.0), se.Segment('b', 2.0, 3.0)]
        )
    ]
    predictions = {
        'x': types.SoundEmbedding(
            embedding=np.array(['a']),
            timestamps=np.array([[0.0, 1.0]]),
            scores=np.array([0.9]),
            context=types.SoundContextParams(
                id='x', sample_rate=16000, length=48000
            ),
        )
    }
    evaluator = se.SegmentationEvaluator(tau=0.1)
    intermediate = evaluator.compute_scores(predictions, references)
    scores = {
        score.metric: score.value
        for score in evaluator.compute_metrics(intermediate)
    }
    self.assertEqual(scores['TimestampsAndEmbeddingsAccuracy'], 0.5)
    self.assertEqual(scores['TimestampsAndEmbeddingsHits'], 1.0)
    self.assertAlmostEqual(scores['mAP'], 0.5)


if __name__ == '__main__':
  absltest.main()
