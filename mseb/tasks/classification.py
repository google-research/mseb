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

"""Classification super task."""

import abc
import dataclasses
import logging
import os
from typing import Iterable

import jaxtyping
from mseb import runner as runner_lib
from mseb import task
from mseb import types
from mseb.evaluators import classification_evaluator
import numpy as np


logger = logging.getLogger(__name__)


class ClassificationTask(task.MSEBTask):
  """Classification task.

  The task assumes a linear classifier with weights. The setup method supports
  to options:
  1. The weights are set to the class label embeddings (i.e., the class labels
     are interpreted as text and encoded by a text encoder).
  2. The weights (with the bias in the last column) are loaded from a previously
     fine-tuned  and saved linear classifier.
  The input to the classifier is the embedding of the sound.
  """

  def __init__(self):
    super().__init__()
    self._evaluator = None

  @property
  def weights_dir(self) -> str:
    """The directory where the weights of the linear classifier are stored."""
    return os.path.join(task.TASK_CACHE_BASEPATH.value, 'classifications')

  def setup(self, runner: runner_lib.EncoderRunner | None = None):
    """Create the weights of the linear classifier."""
    if runner is not None:
      assert hasattr(
          runner, '_output_path'
      ), 'Runner must have an _output_path attribute.'
      runner._output_path = self.weights_dir  # pylint: disable=protected-access
      class_labels = tuple(self.class_labels())
      class_label_embeddings = runner.run([
          types.Text(
              text=class_label, context=types.TextContextParams(id=class_label)
          )
          for class_label in class_labels
      ])
      weights = []
      for class_label in class_labels:
        embedding = class_label_embeddings[class_label]
        assert isinstance(embedding, types.TextEmbedding)
        weight: jaxtyping.Float[jaxtyping.Array, '1 D'] = embedding.embedding
        weight = np.pad(
            weight, ((0, 0), (0, 1)), 'constant', constant_values=0.0
        )
        weights.append(weight)
      weights = np.array(weights, dtype=np.float32)
      classification_evaluator.save_linear_classifier(
          class_labels, weights, self.weights_dir
      )
    else:
      try:
        class_labels, weights = (
            classification_evaluator.load_linear_classifier(self.weights_dir)
        )
      except FileNotFoundError:
        raise ValueError(
            'Weights not found in cache directory. Did you create the cache by'
            ' running run_task_setup?'
        ) from FileNotFoundError

    self._evaluator = classification_evaluator.ClassificationEvaluator(
        class_labels=class_labels, weights=weights
    )

  def compute_scores(
      self, embeddings: types.MultiModalEmbeddingCache
  ) -> dict[str, list[types.Score]]:
    if self._evaluator is None:
      raise ValueError('Evaluator is not initialized. Did you call setup?')

    embeddings_one = {}
    for k, v in embeddings.items():
      assert hasattr(v, 'embedding')
      embeddings_one[k] = dataclasses.replace(
          v,
          embedding=np.pad(
              v.embedding, ((0, 0), (0, 1)), 'constant', constant_values=1.0
          ),
      )

    scores = {}
    for sub_task in self.sub_tasks:
      scores[sub_task] = self._evaluator.compute_metrics(
          self._evaluator.compute_predictions(embeddings_one),
          tuple(self.examples(sub_task)),
      )
    return scores

  @abc.abstractmethod
  def examples(
      self, sub_task: str
  ) -> Iterable[classification_evaluator.ClassificationReference]:
    """Get (utt_id, reference label_id) examples from dataset for a given sub-task."""

  @property
  @abc.abstractmethod
  def sub_tasks(self) -> list[str]:
    """Get the list of sub-tasks for the classification task."""

  @abc.abstractmethod
  def class_labels(self) -> Iterable[str]:
    """Get the list of class labels for the classification task."""
