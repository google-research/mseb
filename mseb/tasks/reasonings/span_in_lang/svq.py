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

"""SVQ span in-lang reasoning tasks."""

import os
from typing import Iterable, Sequence

from mseb import types
from mseb.datasets import simple_voice_questions as svq
from mseb.evaluators import reasoning_evaluator
from mseb.tasks import reasoning


class SVQSpanInLangReasoning(reasoning.ReasoningTask):
  """SVQ span in-lang reasoning task."""

  def __init__(self, locale: str, no_answer_threshold: float = 0.5):
    super().__init__(no_answer_threshold=no_answer_threshold)
    self.locale = locale

  @property
  def embeddings_dir(self) -> str:
    return os.path.join(
        super().embeddings_dir, f'svq_{self.locale}_span_reasoning_in_lang'
    )

  @property
  def sub_tasks(self) -> list[str]:
    return ['span_reasoning_in_lang']

  def sounds(self) -> Iterable[types.SoundWithTitleAndContext]:
    svq_dataset = svq.SimpleVoiceQuestionsDataset()
    for example in svq_dataset.get_task_data(
        'span_reasoning_in_lang'
    ).itertuples():
      if example.locale == self.locale:
        sound = svq_dataset.get_sound_by_id(example.utt_id)
        yield types.SoundWithTitleAndContext(
            waveform=sound.waveform,
            title_text=example.page_title,
            context_text=example.passage_text,
            context=sound.context,
        )

  def examples(
      self, sub_task: str
  ) -> Iterable[reasoning_evaluator.ReasoningSpans]:
    svq_dataset = svq.SimpleVoiceQuestionsDataset()
    for example in svq_dataset.get_task_data(sub_task).itertuples():
      if example.locale == self.locale:
        yield reasoning_evaluator.ReasoningSpans(
            sound_id=example.utt_id,
            reference_answer=example.span,
            texts=example.spans,
        )

  def span_lists(self) -> Iterable[Sequence[types.Text]]:
    svq_dataset = svq.SimpleVoiceQuestionsDataset()
    for example in svq_dataset.get_task_data(
        'span_reasoning_in_lang'
    ).itertuples():
      if example.locale == self.locale:
        yield [
            types.Text(
                text=span,
                context=types.TextContextParams(id=span),
            )
            for span in example.spans
        ]


class SVQEnUsSpanInLangReasoning(SVQSpanInLangReasoning):
  """SVQ span in-lang reasoning for en-US."""

  metadata = types.TaskMetadata(
      name='SVQEnUsSpanInLangReasoning',
      description='Span in-lang reasoning task.',
      reference='TODO',
      type='SpanInLangReasoning',
      category='speech',
      main_score='F1',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[reasoning_evaluator.f1()],
      eval_splits=['test'],
      eval_langs=['en-US'],
      domains=['speech'],
      task_subtypes=['reasoning'],
  )

  def __init__(self):
    super().__init__(locale='en_us', no_answer_threshold=0.5)
