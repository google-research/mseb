"""Brain encoding task.

Evaluates speech embeddings by using them to predict fMRI brain responses
via ridge regression encoding models.
"""

import abc
from typing import Sequence

from mseb import task
from mseb import types
from mseb.evaluators import brain_encoding_evaluator
import numpy as np


class BrainEncodingTask(task.MSEBTask):
  """Brain encoding model evaluation task.

  This task evaluates how well speech embeddings can predict fMRI brain
  responses using ridge regression. The key workflow:

  1. Sound stimuli are yielded via `multimodal_inputs()` for encoding.
  2. Precomputed (or freshly computed) embeddings are passed to
     `compute_scores()`.
  3. Embeddings are aligned to fMRI TRs, FIR delays are applied, and ridge
     regression predicts held-out brain responses.
  4. Per-voxel Pearson correlations are returned as scores.

  Subclasses must provide the fMRI data, stimulus-to-TR mappings, and
  train/test splits.
  """

  def __init__(
      self,
      tr_duration: float = 2.0,
      delays: Sequence[int] = (1, 2, 3, 4),
      alphas: Sequence[float] | None = None,
  ):
    """Initializes the BrainEncodingTask.

    Args:
      tr_duration: Duration of one fMRI TR in seconds.
      delays: FIR delay values in TRs for the hemodynamic response.
      alphas: Ridge regularization candidates.
    """
    super().__init__()
    self._evaluator = brain_encoding_evaluator.BrainEncodingEvaluator(
        tr_duration=tr_duration,
        delays=delays,
        alphas=alphas,
    )

  @abc.abstractmethod
  def train_examples(
      self, sub_task: str
  ) -> Sequence[brain_encoding_evaluator.BrainEncodingExample]:
    """Returns training examples for a given sub-task.

    Each example maps a sound ID to a range of TRs in the fMRI data.

    Args:
      sub_task: The sub-task (e.g., per-subject, per-region) to return examples
        for.

    Returns:
      A list of BrainEncodingExample objects.
    """

  @abc.abstractmethod
  def test_examples(
      self, sub_task: str
  ) -> Sequence[brain_encoding_evaluator.BrainEncodingExample]:
    """Returns test examples for a given sub-task."""

  @abc.abstractmethod
  def fmri_train(self, sub_task: str) -> np.ndarray:
    """Returns training fMRI responses of shape (T_train, V)."""

  @abc.abstractmethod
  def fmri_test(self, sub_task: str) -> np.ndarray:
    """Returns test fMRI responses of shape (T_test, V)."""

  @property
  @abc.abstractmethod
  def sub_tasks(self) -> list[str]:
    """Returns the list of sub-tasks (e.g., per-subject, per-region)."""

  def compute_scores(
      self, embeddings: types.MultiModalEmbeddingCache
  ) -> dict[str, list[types.Score]]:
    """Runs brain encoding evaluation for all sub-tasks.

    Args:
      embeddings: Precomputed embeddings keyed by sound ID. Values should be
        SoundEmbedding objects with frame-level embeddings and timestamps.

    Returns:
      Dictionary mapping sub-task names to lists of Score objects.
    """
    scores = {}
    for sub_task in self.sub_tasks:
      scores[sub_task] = self._evaluator(
          embeddings=embeddings,
          train_examples=self.train_examples(sub_task),
          test_examples=self.test_examples(sub_task),
          fmri_train=self.fmri_train(sub_task),
          fmri_test=self.fmri_test(sub_task),
      )
    return scores
