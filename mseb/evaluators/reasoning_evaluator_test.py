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

"""Tests for ReasoningEvaluator class."""

from absl.testing import absltest
from mseb import types
from mseb.evaluators import reasoning_evaluator
import numpy as np
import numpy.testing as npt
from sklearn import mixture


class NormalizeSquadTest(absltest.TestCase):

  def test_lowercases(self):
    self.assertEqual(
        reasoning_evaluator.normalize_squad('Hello World'), 'hello world'
    )

  def test_removes_punctuation(self):
    self.assertEqual(
        reasoning_evaluator.normalize_squad('hello, world!'), 'hello world'
    )

  def test_removes_articles(self):
    self.assertEqual(
        reasoning_evaluator.normalize_squad('the cat sat on a mat'),
        'cat sat on mat',
    )

  def test_removes_extra_whitespace(self):
    self.assertEqual(
        reasoning_evaluator.normalize_squad('  hello   world  '), 'hello world'
    )

  def test_combined_normalization(self):
    self.assertEqual(
        reasoning_evaluator.normalize_squad('The Quick, Brown Fox!'),
        'quick brown fox',
    )

  def test_empty_string(self):
    self.assertEqual(reasoning_evaluator.normalize_squad(''), '')


class ScoreFactoryTest(absltest.TestCase):

  def test_f1_defaults(self):
    score = reasoning_evaluator.f1()
    self.assertEqual(score.metric, 'F1')
    self.assertEqual(score.value, 0.0)
    self.assertIsNone(score.std)

  def test_f1_with_values(self):
    score = reasoning_evaluator.f1(value=0.75, std=0.1)
    self.assertEqual(score.value, 0.75)
    self.assertEqual(score.std, 0.1)

  def test_gmean_f1_defaults(self):
    score = reasoning_evaluator.gmean_f1()
    self.assertEqual(score.metric, 'GmeanF1')
    self.assertEqual(score.value, 0.0)
    self.assertIsNone(score.std)


class FindDecisionBoundaryTest(absltest.TestCase):

  def test_equal_variance_components(self):
    """Two equal-variance Gaussians should have boundary at the midpoint."""
    gmm = mixture.GaussianMixture(n_components=2, covariance_type='full')
    gmm.weights_ = np.array([0.5, 0.5])
    gmm.means_ = np.array([[2.0], [8.0]])
    gmm.covariances_ = np.array([[[1.0]], [[1.0]]])
    gmm.precisions_cholesky_ = np.array([[[1.0]], [[1.0]]])
    boundary = reasoning_evaluator._find_decision_boundary(gmm)
    npt.assert_almost_equal(boundary, 5.0, decimal=3)

  def test_unequal_variance_components(self):
    """Boundary should fall between the two means."""
    gmm = mixture.GaussianMixture(n_components=2, covariance_type='full')
    gmm.weights_ = np.array([0.5, 0.5])
    gmm.means_ = np.array([[0.0], [10.0]])
    gmm.covariances_ = np.array([[[1.0]], [[4.0]]])
    gmm.precisions_cholesky_ = np.array([[[1.0]], [[0.5]]])
    boundary = reasoning_evaluator._find_decision_boundary(gmm)
    # Boundary should be between the two means
    self.assertGreater(boundary, 0.0)
    self.assertLess(boundary, 10.0)


class FindNoAnswerThresholdByGmmTest(absltest.TestCase):

  def test_bimodal_scores(self):
    """GMM should separate two distinct score clusters."""
    low_scores = np.random.RandomState(42).normal(0.1, 0.02, 50).tolist()
    high_scores = np.random.RandomState(42).normal(0.9, 0.02, 50).tolist()
    scores = low_scores + high_scores
    threshold = reasoning_evaluator.find_no_answer_threshold_by_gmm(scores)
    # Threshold should be between the clusters
    self.assertGreater(threshold, 0.2)
    self.assertLess(threshold, 0.8)


class ComputeF1ScoreTest(absltest.TestCase):

  def test_exact_match(self):
    self.assertEqual(
        reasoning_evaluator.compute_f1_score('b l a', 'b l a'), 1.0
    )

  def test_partial_match(self):
    self.assertAlmostEqual(
        reasoning_evaluator.compute_f1_score('b l a', 'b l i'), 2 / 3
    )

  def test_no_match(self):
    self.assertEqual(
        reasoning_evaluator.compute_f1_score('b l a', 'x y z'), 0.0
    )

  def test_no_answer_target_with_real_prediction(self):
    self.assertEqual(
        reasoning_evaluator.compute_f1_score(
            reasoning_evaluator.NO_ANSWER_STR, 'b l a'
        ),
        0.0,
    )

  def test_real_target_with_no_answer_prediction(self):
    self.assertEqual(
        reasoning_evaluator.compute_f1_score(
            'b l a', reasoning_evaluator.NO_ANSWER_STR
        ),
        0.0,
    )

  def test_both_no_answer(self):
    self.assertEqual(
        reasoning_evaluator.compute_f1_score(
            reasoning_evaluator.NO_ANSWER_STR,
            reasoning_evaluator.NO_ANSWER_STR,
        ),
        1.0,
    )

  def test_invalid_answer_prediction(self):
    self.assertEqual(
        reasoning_evaluator.compute_f1_score(
            'b l a', reasoning_evaluator.INVALID_ANSWER_STR
        ),
        0.0,
    )

  def test_invalid_answer_with_no_answer_target(self):
    self.assertEqual(
        reasoning_evaluator.compute_f1_score(
            reasoning_evaluator.NO_ANSWER_STR,
            reasoning_evaluator.INVALID_ANSWER_STR,
        ),
        0.0,
    )

  def test_superset_prediction(self):
    """Prediction has extra tokens beyond the target."""
    self.assertAlmostEqual(
        reasoning_evaluator.compute_f1_score('a b', 'a b c'),
        2 * (2 / 3) * 1.0 / ((2 / 3) + 1.0),
    )


def _make_text_embedding(text_id: str, embedding: list[float]):
  return types.TextEmbedding(
      embedding=np.array([embedding], dtype=np.float32),
      spans=np.array([[0, -1]]),  # pyrefly: ignore[bad-argument-type]
      context=types.TextContextParams(id=text_id),
  )


def _make_sound_embedding(sound_id: str, embedding: list[float]):
  return types.SoundEmbedding(
      embedding=np.array([embedding]),
      timestamps=np.array([[0.0, 1.0]]),  # pyrefly: ignore[bad-argument-type]
      context=types.SoundContextParams(
          id=sound_id, sample_rate=16000, length=100, language='en'
      ),
  )


class ComputePredictionsTest(absltest.TestCase):

  def test_basic_prediction(self):
    ev = reasoning_evaluator.ReasoningEvaluator(
        span_embeddings_by_sound_id={
            'test': [
                _make_text_embedding('b l i', [3.0, 4.0]),
                _make_text_embedding('b l a', [5.0, 6.0]),
                _make_text_embedding('x y z', [1.0, 2.0]),
            ]
        },
        no_answer_threshold=0.5,
    )
    predictions = ev.compute_predictions(
        embeddings_by_sound_id={
            'test': _make_sound_embedding('test', [2.5, 3.0]),
        },
    )
    self.assertLen(predictions, 1)
    self.assertEqual(predictions['test'].prediction, 'b l a')

  def test_empty_spans_returns_no_answer(self):
    """When a sound_id has no span embeddings, predict NO_ANSWER_STR."""
    ev = reasoning_evaluator.ReasoningEvaluator(
        span_embeddings_by_sound_id={'test': []},
        no_answer_threshold=0.5,
    )
    predictions = ev.compute_predictions(
        embeddings_by_sound_id={
            'test': _make_sound_embedding('test', [1.0, 0.0]),
        },
    )
    self.assertEqual(
        predictions['test'].prediction, reasoning_evaluator.NO_ANSWER_STR
    )

  def test_score_below_threshold_returns_no_answer(self):
    """When the best score is below threshold, predict NO_ANSWER_STR."""
    ev = reasoning_evaluator.ReasoningEvaluator(
        span_embeddings_by_sound_id={
            'test': [_make_text_embedding('answer', [1.0, 0.0])],
        },
        no_answer_threshold=999.0,  # Very high threshold
    )
    predictions = ev.compute_predictions(
        embeddings_by_sound_id={
            'test': _make_sound_embedding('test', [1.0, 0.0]),
        },
    )
    self.assertEqual(
        predictions['test'].prediction, reasoning_evaluator.NO_ANSWER_STR
    )

  def test_multiple_sound_ids(self):
    ev = reasoning_evaluator.ReasoningEvaluator(
        span_embeddings_by_sound_id={
            's1': [_make_text_embedding('ans1', [1.0, 0.0])],
            's2': [_make_text_embedding('ans2', [0.0, 1.0])],
        },
        no_answer_threshold=-999.0,  # Very low threshold
    )
    predictions = ev.compute_predictions(
        embeddings_by_sound_id={
            's1': _make_sound_embedding('s1', [1.0, 0.0]),
            's2': _make_sound_embedding('s2', [0.0, 1.0]),
        },
    )
    self.assertLen(predictions, 2)
    self.assertEqual(predictions['s1'].prediction, 'ans1')
    self.assertEqual(predictions['s2'].prediction, 'ans2')


class ComputeMetricsTest(absltest.TestCase):

  def test_single_correct_prediction(self):
    ev = reasoning_evaluator.ReasoningEvaluator(span_embeddings_by_sound_id={})
    scores = ev.compute_metrics(
        predictions={
            'test': types.TextPrediction(
                prediction='b l a',
                context=types.PredictionContextParams(id='test'),
            ),
        },
        spans_batch=[
            reasoning_evaluator.ReasoningSpans(
                sound_id='test',
                texts=['b l i', 'b l a', 'x y z'],
                reference_answer='b l i',
            ),
        ],
    )
    npt.assert_equal(len(scores), 4)
    self.assertEqual(scores[0].metric, 'GmeanF1')
    npt.assert_equal(scores[0].value, 2 / 3)
    self.assertEqual(scores[1].metric, 'F1')
    npt.assert_equal(scores[1].value, 2 / 3)
    self.assertEqual(scores[2].metric, 'InvalidResultRate')
    npt.assert_equal(scores[2].value, 0)
    self.assertEqual(scores[3].metric, 'MissingResultRate')
    npt.assert_equal(scores[3].value, 0)

  def test_invalid_and_missing_answers(self):
    ev = reasoning_evaluator.ReasoningEvaluator(span_embeddings_by_sound_id={})
    scores = ev.compute_metrics(
        predictions={
            'test': types.TextPrediction(
                prediction=reasoning_evaluator.INVALID_ANSWER_STR,
                context=types.PredictionContextParams(id='test'),
            ),
            'test2': types.TextPrediction(
                prediction=reasoning_evaluator.NO_RESPONSE_STR,
                context=types.PredictionContextParams(id='test'),
            ),
        },
        spans_batch=[
            reasoning_evaluator.ReasoningSpans(
                sound_id='test',
                texts=['b l i', 'b l a', 'x y z'],
                reference_answer='b l i',
            ),
            reasoning_evaluator.ReasoningSpans(
                sound_id='test2',
                texts=['b l i', 'b l a', 'x y z'],
                reference_answer='b l i',
            ),
        ],
    )
    npt.assert_equal(len(scores), 4)
    npt.assert_equal(scores[0].value, 0)  # GmeanF1
    npt.assert_equal(scores[1].value, 0)  # F1
    npt.assert_equal(scores[2].value, 0.5)  # InvalidResultRate
    npt.assert_equal(scores[3].value, 0.5)  # MissingResultRate

  def test_mixed_no_answer_and_real_answers(self):
    """GmeanF1 should be geometric mean of no-answer F1 and real F1."""
    ev = reasoning_evaluator.ReasoningEvaluator(span_embeddings_by_sound_id={})
    scores = ev.compute_metrics(
        predictions={
            'real': types.TextPrediction(
                prediction='b l a',
                context=types.PredictionContextParams(id='real'),
            ),
            'no_ans': types.TextPrediction(
                prediction=reasoning_evaluator.NO_ANSWER_STR,
                context=types.PredictionContextParams(id='no_ans'),
            ),
        },
        spans_batch=[
            reasoning_evaluator.ReasoningSpans(
                sound_id='real',
                texts=['b l a'],
                reference_answer='b l a',
            ),
            reasoning_evaluator.ReasoningSpans(
                sound_id='no_ans',
                texts=[],
                reference_answer=reasoning_evaluator.NO_ANSWER_STR,
            ),
        ],
    )
    # Both real F1 and no-answer F1 are 1.0, so gmean = 1.0
    npt.assert_almost_equal(scores[0].value, 1.0)  # GmeanF1
    npt.assert_almost_equal(scores[1].value, 1.0)  # F1

  def test_gmean_f1_penalizes_trivial_no_answer(self):
    """Predicting NO_ANSWER for everything should get GmeanF1=0."""
    ev = reasoning_evaluator.ReasoningEvaluator(span_embeddings_by_sound_id={})
    scores = ev.compute_metrics(
        predictions={
            'real': types.TextPrediction(
                prediction=reasoning_evaluator.NO_ANSWER_STR,
                context=types.PredictionContextParams(id='real'),
            ),
            'no_ans': types.TextPrediction(
                prediction=reasoning_evaluator.NO_ANSWER_STR,
                context=types.PredictionContextParams(id='no_ans'),
            ),
        },
        spans_batch=[
            reasoning_evaluator.ReasoningSpans(
                sound_id='real',
                texts=['b l a'],
                reference_answer='b l a',
            ),
            reasoning_evaluator.ReasoningSpans(
                sound_id='no_ans',
                texts=[],
                reference_answer=reasoning_evaluator.NO_ANSWER_STR,
            ),
        ],
    )
    # Real F1=0 (wrong), no-answer F1=1.0 (correct)
    # gmean = 0^0.5 * 1.0^0.5 = 0
    npt.assert_equal(scores[0].value, 0.0)  # GmeanF1
    # Regular F1 = (0 + 1.0) / 2 = 0.5
    npt.assert_equal(scores[1].value, 0.5)  # F1

  def test_all_no_answer_references(self):
    """All references are NO_ANSWER_STR, all predictions correct."""
    ev = reasoning_evaluator.ReasoningEvaluator(span_embeddings_by_sound_id={})
    scores = ev.compute_metrics(
        predictions={
            's1': types.TextPrediction(
                prediction=reasoning_evaluator.NO_ANSWER_STR,
                context=types.PredictionContextParams(id='s1'),
            ),
        },
        spans_batch=[
            reasoning_evaluator.ReasoningSpans(
                sound_id='s1',
                texts=[],
                reference_answer=reasoning_evaluator.NO_ANSWER_STR,
            ),
        ],
    )
    # weight=0 (no real answers), weight_no_answer=1.0
    # mean_real=0.0, mean_no_answer=1.0
    # gmean = 0.0^0 * 1.0^1 = 1.0
    npt.assert_almost_equal(scores[0].value, 1.0)  # GmeanF1


if __name__ == '__main__':
  absltest.main()
