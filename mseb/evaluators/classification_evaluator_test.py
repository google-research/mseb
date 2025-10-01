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

import logging

from absl.testing import absltest
from mseb import types
from mseb.evaluators import classification_evaluator
import numpy as np


class ClassificationEvaluatorTest(absltest.TestCase):
  """Tests for the ClassificationEvaluator class."""

  def setUp(self):
    super().setUp()
    self.embedding_table = np.array([
        [1.0, 0.0, 0.0, 0.0],  # Class 'cat'
        [0.0, 1.0, 0.0, 0.0],  # Class 'dog'
        [0.0, 0.0, 1.0, 0.0],  # Class 'bird'
    ], dtype=np.float32)
    self.class_labels = ['cat', 'dog', 'bird']

  def test_initialization_invalid_k_raises_error(self):
    with self.assertRaises(ValueError):
      classification_evaluator.ClassificationEvaluator(
          weights=self.embedding_table,
          class_labels=self.class_labels,
          top_k_value=0,
      )

  def test_initialization_large_k_logs_warning(self):
    with self.assertLogs(level='WARNING') as log:
      logging.getLogger().warning(
          'Dummy message to activate logger.'
      )  # Ensure logger is active
      classification_evaluator.ClassificationEvaluator(
          weights=self.embedding_table,
          class_labels=self.class_labels,
          top_k_value=3,  # k is equal to the number of classes
      )
    self.assertIn('will always be 100%', log.output[1])

  def test_compute_predictions_malformed_embedding_raises_error(self):
    evaluator = classification_evaluator.ClassificationEvaluator(
        class_labels=self.class_labels, weights=self.embedding_table
    )
    malformed_embeddings = {
        'id_1': types.SoundEmbedding(
            embedding=np.array([]),
            timestamps=np.array([]),
            context=types.SoundContextParams(
                id='id_1',
                sample_rate=16000,
                length=1
            )
        ),
    }
    with self.assertRaisesRegex(ValueError, 'Found missing or malformed'):
      evaluator.compute_predictions(malformed_embeddings)

  def test_compute_metrics_perfect_score(self):
    """Tests a scenario where all predictions are correct."""
    evaluator = classification_evaluator.ClassificationEvaluator(
        class_labels=self.class_labels,
        weights=self.embedding_table,
        top_k_value=2,
    )
    # Each embedding perfectly matches a class embedding.
    scores = {
        'ex1': np.array([1.0, 0.1, 0.2]),  # Should be 'cat'
        'ex2': np.array([0.1, 1.0, 0.2]),  # Should be 'dog'
    }
    references = [
        classification_evaluator.ClassificationReference('ex1', 'cat'),
        classification_evaluator.ClassificationReference('ex2', 'dog'),
    ]
    results = evaluator.compute_metrics(scores, references)
    # For a perfect score, all metrics should be 1.0
    for score in results:
      self.assertAlmostEqual(
          score.value, 1.0,
          msg=f'Metric {score.metric} failed.'
      )

  def test_compute_metrics_top_k_accuracy(self):
    evaluator = classification_evaluator.ClassificationEvaluator(
        class_labels=self.class_labels,
        weights=self.embedding_table,
        top_k_value=2,
    )
    # The highest score is for 'dog', but the second
    # highest is 'cat' (the true label).
    scores = {'ex1': np.array([0.9, 1.0, 0.2])}  # True: cat, Pred: dog
    references = [
        classification_evaluator.ClassificationReference('ex1', 'cat')
    ]
    results = {
        s.metric: s.value for s in evaluator.compute_metrics(scores, references)
    }
    self.assertAlmostEqual(results['Accuracy'], 0.0)
    self.assertAlmostEqual(results['Top-2 Accuracy'], 1.0)

  def test_compute_metrics_balanced_accuracy(self):
    evaluator = classification_evaluator.ClassificationEvaluator(
        class_labels=self.class_labels, weights=self.embedding_table
    )
    # Model always predicts 'cat'. Dataset is 3 'cat', 1 'dog'.
    scores = {
        'cat1': np.array([1.0, 0.1, 0.2]),
        'cat2': np.array([1.0, 0.1, 0.2]),
        'cat3': np.array([1.0, 0.1, 0.2]),
        'dog1': np.array([1.0, 0.1, 0.2]),
    }
    references = [
        classification_evaluator.ClassificationReference('cat1', 'cat'),
        classification_evaluator.ClassificationReference('cat2', 'cat'),
        classification_evaluator.ClassificationReference('cat3', 'cat'),
        classification_evaluator.ClassificationReference('dog1', 'dog'),
    ]

    results = {
        s.metric: s.value for s in evaluator.compute_metrics(scores, references)
    }
    # Model gets 3 out of 4 correct.
    self.assertAlmostEqual(results['Accuracy'], 0.75)
    # Recall for 'cat' is 3/3 = 1.0. Recall for 'dog' is 0/1 = 0.0.
    # Balanced accuracy = (1.0 + 0.0) / 2 = 0.5
    # (Note: scikit-learn averages recall over all PRESENT classes)
    self.assertAlmostEqual(results['Balanced Accuracy'], 0.5)

  def test_multimodality_with_text_embeddings(self):
    evaluator = classification_evaluator.ClassificationEvaluator(
        class_labels=self.class_labels, weights=self.embedding_table
    )
    text_embeddings = {
        'id_1': types.TextEmbedding(
            embedding=np.array([[1., 0., 0., 0.]]),
            spans=np.array([[0, 1]]),
            context=types.TextContextParams(id='id_1')
        ),
    }
    # Check that predictions are computed without error
    scores = evaluator.compute_predictions(text_embeddings)
    # The embedding for 'id_1' matches 'cat' perfectly
    expected_scores = np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(scores['id_1'], expected_scores)

  def test_save_and_load_linear_classifier(self):
    base_dir = self.create_tempdir().full_path
    classification_evaluator.save_linear_classifier(
        self.class_labels, self.embedding_table, base_dir
    )
    class_labels_loaded, weights_loaded = (
        classification_evaluator.load_linear_classifier(base_dir)
    )
    self.assertSequenceEqual(self.class_labels, class_labels_loaded)
    np.testing.assert_allclose(self.embedding_table, weights_loaded)


class MultiLabelClassificationEvaluatorTest(absltest.TestCase):
  """Tests for the MultiLabelClassificationEvaluator class."""

  def setUp(self):
    super().setUp()
    self.embedding_table = np.array([
        [1.0, 0.0, 0.0],  # Class 'cat'
        [0.0, 1.0, 0.0],  # Class 'dog'
        [0.0, 0.0, 1.0],  # Class 'bird'
    ], dtype=np.float32)
    self.id_by_class_index = ['cat', 'dog', 'bird']
    self.evaluator = classification_evaluator.MultiLabelClassificationEvaluator(
        weights=self.embedding_table,
        id_by_class_index=self.id_by_class_index,
    )

  def test_compute_metrics_perfect_score(self):
    # Example 1 has high scores for 'cat' and 'dog'
    # Example 2 has a high score for 'bird'
    scores = {
        'ex1': np.array([0.9, 0.8, 0.1]),
        'ex2': np.array([0.2, 0.1, 0.9]),
    }
    references = [
        classification_evaluator.MultiLabelClassificationReference(
            example_id='ex1',
            label_ids=['cat', 'dog']
        ),
        classification_evaluator.MultiLabelClassificationReference(
            example_id='ex',
            label_ids=['bird']
        ),
    ]

    results = {
        s.metric: s.value
        for s in self.evaluator.compute_metrics(
            scores,
            references,
            threshold=0.5
        )
    }

    # For perfect predictions (above threshold), most scores should be 1.0
    self.assertAlmostEqual(results['mAP'], 0.666666, places=5)
    self.assertAlmostEqual(results['Micro F1'], 1.0)
    self.assertAlmostEqual(results['Macro F1'], 0.666666, places=5)
    self.assertAlmostEqual(results['Subset Accuracy'], 1.0)
    self.assertAlmostEqual(results['Hamming Loss'], 0.0)

  def test_compute_metrics_mixed_score(self):
    # ex1: Predicts 'cat' (correct), misses 'dog' (FN), predicts 'bird' (FP)
    # ex2: Predicts 'bird' (correct)
    scores = {
        'ex1': np.array([0.9, 0.2, 0.7]),
        'ex2': np.array([0.1, 0.3, 0.8]),
    }
    references = [
        classification_evaluator.MultiLabelClassificationReference(
            example_id='ex1',
            label_ids=['cat', 'dog']
        ),
        classification_evaluator.MultiLabelClassificationReference(
            example_id='ex2',
            label_ids=['bird']
        ),
    ]

    results = {
        s.metric: s.value
        for s in self.evaluator.compute_metrics(
            scores,
            references,
            threshold=0.5
        )
    }

    # y_true = [[1, 1, 0], [0, 0, 1]]
    # y_pred = [[1, 0, 1], [0, 0, 1]]
    # TP=2, FP=1, FN=1, TN=2
    self.assertAlmostEqual(results['mAP'], 0.833333, places=5)
    self.assertAlmostEqual(results['Micro F1'], 0.666666, places=5)
    self.assertAlmostEqual(results['Macro F1'], 0.555555, places=5)
    self.assertAlmostEqual(results['Subset Accuracy'], 0.5)
    self.assertAlmostEqual(results['Hamming Loss'], 2.0 / 6.0, places=5)

  def test_compute_metrics_empty_labels(self):
    scores = {'ex1': np.array([0.1, 0.2, 0.3])}  # Model predicts nothing
    references = [
        classification_evaluator.MultiLabelClassificationReference(
            example_id='ex1',
            label_ids=[]
        ),
    ]

    results = {
        s.metric: s.value
        for s in self.evaluator.compute_metrics(
            scores,
            references,
            threshold=0.5
        )
    }

    # With no positive labels, precision is undefined, but F1 should be 1.0
    # because the model correctly predicted nothing.
    self.assertAlmostEqual(results['mAP'], 0.0)  # No positive class to rank
    self.assertAlmostEqual(results['Micro F1'], 0.0)
    self.assertAlmostEqual(results['Macro F1'], 0.0)
    self.assertAlmostEqual(results['Subset Accuracy'], 1.0)
    self.assertAlmostEqual(results['Hamming Loss'], 0.0)

  def test_no_matching_references(self):
    scores = {'ex1': np.array([0.9, 0.8, 0.1])}
    references = [
        classification_evaluator.MultiLabelClassificationReference(
            example_id='non_existent_id',
            label_ids=['cat']
        ),
    ]
    results = self.evaluator.compute_metrics(scores, references)
    self.assertEqual(results, [])


if __name__ == '__main__':
  absltest.main()
