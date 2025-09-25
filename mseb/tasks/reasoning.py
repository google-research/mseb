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

"""Reasoning super task."""

import abc
import logging
import os
from typing import Iterable, Sequence

from mseb import runner as runner_lib
from mseb import task
from mseb import types
from mseb.evaluators import reasoning_evaluator


logger = logging.getLogger(__name__)


class ReasoningTask(task.MSEBTask):
  """Reasoning task."""

  def __init__(self, no_answer_threshold: float = 0.5):
    super().__init__()
    self.no_answer_threshold = no_answer_threshold
    self._evaluator = None

  @property
  def embeddings_dir(self) -> str:
    """The directory where the span embeddings cache is stored."""
    return os.path.join(task.CACHE_BASEPATH.value, 'reasonings')

  def setup(
      self, runner: runner_lib.EncoderRunner | None = None
  ):
    """Create the span embeddings cache."""
    if runner is not None:
      assert hasattr(
          runner, '_output_path'
      ), 'Runner must have an _output_path attribute.'
      runner._output_path = self.embeddings_dir  # pylint: disable=protected-access
      unique_spans = {}
      for span_list in self.span_lists():
        for span in span_list:
          unique_spans[span.text] = span
      embeddings_by_text = runner.run(unique_spans.values())
    else:
      try:
        logger.info(
            'Loading span embeddings cache from %s', self.embeddings_dir
        )
        embeddings_by_text = runner_lib.load_embeddings(
            os.path.join(self.embeddings_dir, 'embeddings')
        )
      except FileNotFoundError:
        raise ValueError(
            'Span embeddings cache not found in cache directory. Did you'
            ' create the cache by running run_task_setup?'
        ) from FileNotFoundError

    embeddings_by_sound_id = {}
    for sub_task in self.sub_tasks:
      for spans in self.examples(sub_task):
        embeddings_by_sound_id[spans.sound_id] = [
            embeddings_by_text[text] for text in spans.texts
        ]

    self._evaluator = reasoning_evaluator.ReasoningEvaluator(
        span_embeddings_by_sound_id=embeddings_by_sound_id,
        no_answer_threshold=self.no_answer_threshold,
    )

  def compute_scores(
      self, embeddings: types.MultiModalEmbeddingCache
  ) -> dict[str, list[types.Score]]:
    if self._evaluator is None:
      raise ValueError('Evaluator is not initialized. Did you call setup?')

    sound_embeddings = {}
    for k, v in embeddings.items():
      assert isinstance(v, types.SoundEmbedding)
      sound_embeddings[k] = v

    scores = {}
    for sub_task in self.sub_tasks:
      scores[sub_task] = self._evaluator.compute_metrics(
          self._evaluator.compute_predictions(sound_embeddings),
          tuple(self.examples(sub_task)),
      )
    return scores

  @abc.abstractmethod
  def examples(
      self, sub_task: str
  ) -> Iterable[reasoning_evaluator.ReasoningSpans]:
    """Get (utt_id, spans) examples from dataset for a given sub-task."""

  @property
  @abc.abstractmethod
  def sub_tasks(self) -> list[str]:
    """Get the list of sub-tasks for the reasoning task."""

  @abc.abstractmethod
  def span_lists(self) -> Iterable[Sequence[types.Text]]:
    """Get the list of spans for the reasoning task."""
