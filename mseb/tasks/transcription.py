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

"""Transcription super task."""

import abc
from typing import Iterable

from absl import flags
from mseb import runner as runner_lib
from mseb import task
from mseb import types
from mseb.evaluators import transcription_evaluator


CONTEXTUAL_BIAS_KEY = flags.DEFINE_string(
    'contextual_bias_key',
    None,
    'Key to use for the contextual bias.',
)


class TranscriptionTask(task.MSEBTask):
  """Transcription task."""

  def __init__(self):
    super().__init__()
    self._evaluator = None

  def setup(self, runner: runner_lib.EncoderRunner | None = None):
    """Create the evaluator."""
    self._evaluator = transcription_evaluator.TranscriptionEvaluator()

  def compute_scores(
      self, embeddings: types.MultiModalEmbeddingCache
  ) -> dict[str, list[types.Score]]:
    if self._evaluator is None:
      raise ValueError('Evaluator is not initialized. Did you call setup?')

    if not isinstance(next(iter(embeddings.values())), types.TextPrediction):
      transcript_by_sound_id = self._evaluator.compute_predictions(embeddings)
    else:
      transcript_by_sound_id = embeddings

    scores = {}
    for sub_task in self.sub_tasks:
      scores[sub_task] = self._evaluator.compute_metrics(
          transcript_by_sound_id=transcript_by_sound_id,
          transcript_truths=tuple(self.examples(sub_task)),
      )
    return scores

  @abc.abstractmethod
  def examples(
      self, sub_task: str
  ) -> Iterable[transcription_evaluator.TranscriptTruth]:
    """Get (utt_id, transcript_truth) examples from dataset for a given sub-task."""

  @property
  @abc.abstractmethod
  def sub_tasks(self) -> list[str]:
    """Get the list of sub-tasks for the transcription task."""
