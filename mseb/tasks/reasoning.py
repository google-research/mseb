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
from typing import Any, Iterable, Sequence, Type

from mseb import runner as runner_lib
from mseb import task
from mseb import types
from mseb.encoders import encoder_registry
from mseb.evaluators import reasoning_evaluator


logger = logging.getLogger(__name__)


class ReasoningTask(task.MSEBTask):
  """Reasoning task."""

  def __init__(
      self,
      text_encoder_name: str | None = None,
      no_answer_threshold: float = 0.5,
  ):
    super().__init__()
    self.text_encoder_name = text_encoder_name
    self.no_answer_threshold = no_answer_threshold
    self._evaluator = None

  @property
  def embeddings_dir(self) -> str:
    """The directory where the span embeddings cache is stored."""
    return os.path.join(task.CACHE_BASEPATH.value, 'reasonings')

  def setup(
      self, runner_cls: Type[runner_lib.EncoderRunner] | None = None, **kwargs
  ):
    """Create the span embeddings cache."""
    if runner_cls is not None:
      if self.text_encoder_name is None:
        raise ValueError('Text encoder name is not set.')
      text_encoder = encoder_registry.get_encoder_metadata(
          self.text_encoder_name
      ).load()
      kwargs: dict[str, Any] = {'output_path': self.embeddings_dir, **kwargs}
      runner = runner_cls(encoder=text_encoder, **kwargs)
      unique_spans = {}
      for span_list in self.span_lists():
        for span in span_list:
          unique_spans[span.text] = span
      embeddings = runner.run(unique_spans.values())
    else:
      try:
        logger.info(
            'Loading span embeddings cache from %s', self.embeddings_dir
        )
        embeddings = runner_lib.load_embeddings(
            os.path.join(self.embeddings_dir, 'embeddings')
        )
      except FileNotFoundError:
        raise ValueError(
            'Span embeddings cache not found in cache directory. Did you'
            ' create the cache by running run_task_setup?'
        ) from FileNotFoundError

    self._evaluator = reasoning_evaluator.ReasoningEvaluator(
        span_embeddings_by_text=embeddings,
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
      scores[sub_task] = self._evaluator(
          sound_embeddings, tuple(self.examples(sub_task))
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
