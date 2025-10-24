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

"""SVQ span cross-lang reasoning tasks."""

import os
from typing import Iterable, Sequence

from mseb import types
from mseb.datasets import simple_voice_questions as svq
from mseb.evaluators import reasoning_evaluator
from mseb.tasks import reasoning


class SVQSpanCrossLangReasoning(reasoning.ReasoningTask):
  """SVQ span cross-lang reasoning task."""

  locale: str | None = None

  @property
  def embeddings_dir(self) -> str:
    assert self.locale is not None
    return os.path.join(
        super().embeddings_dir, f'svq_{self.locale}_span_reasoning_cross_lang'
    )

  @property
  def sub_tasks(self) -> list[str]:
    return ['span_reasoning_cross_lang']

  def sounds(self) -> Iterable[types.SoundWithTitleAndContext]:
    svq_dataset = svq.SimpleVoiceQuestionsDataset()
    for example in svq_dataset.get_task_data(
        'span_reasoning_cross_lang',
        dtype={
            'locale': str,
            'utt_id': str,
            'page_title': str,
            'passage_text': str,
        },
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
    for example in svq_dataset.get_task_data(
        sub_task,
        dtype={
            'locale': str,
            'utt_id': str,
            'span': str,
            'spans': Sequence[str],
        },
    ).itertuples():
      if example.locale == self.locale:
        yield reasoning_evaluator.ReasoningSpans(
            sound_id=example.utt_id,
            reference_answer=example.span,
            texts=example.spans,
        )

  def span_lists(self) -> Iterable[Sequence[types.Text]]:
    svq_dataset = svq.SimpleVoiceQuestionsDataset()
    for example in svq_dataset.get_task_data(
        'span_reasoning_cross_lang',
        dtype={'locale': str, 'spans': Sequence[str]},
    ).itertuples():
      if example.locale == self.locale:
        yield [
            types.Text(
                text=span,
                context=types.TextContextParams(id=span),
            )
            for span in example.spans
        ]


class SVQArEgSpanCrossLangReasoning(SVQSpanCrossLangReasoning):
  locale = 'ar_eg'
  metadata = types.TaskMetadata(
      name='SVQArEgSpanCrossLangReasoning',
      description='Span cross-lang reasoning task.',
      reference='TODO',
      type='SpanCrossLangReasoning',
      category='speech',
      main_score='GmeanF1',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[reasoning_evaluator.gmean_f1(), reasoning_evaluator.f1()],
      eval_splits=['test'],
      eval_langs=['ar-EG'],
      domains=['speech'],
      task_subtypes=['reasoning'],
  )


class SVQArXGulfSpanCrossLangReasoning(SVQSpanCrossLangReasoning):
  locale = 'ar_x_gulf'
  metadata = types.TaskMetadata(
      name='SVQArXGulfSpanCrossLangReasoning',
      description='Span cross-lang reasoning task.',
      reference='TODO',
      type='SpanCrossLangReasoning',
      category='speech',
      main_score='GmeanF1',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[reasoning_evaluator.gmean_f1(), reasoning_evaluator.f1()],
      eval_splits=['test'],
      eval_langs=['ar-x-gulf'],
      domains=['speech'],
      task_subtypes=['reasoning'],
  )


class SVQArXLevantSpanCrossLangReasoning(SVQSpanCrossLangReasoning):
  locale = 'ar_x_levant'
  metadata = types.TaskMetadata(
      name='SVQArXLevantSpanCrossLangReasoning',
      description='Span cross-lang reasoning task.',
      reference='TODO',
      type='SpanCrossLangReasoning',
      category='speech',
      main_score='GmeanF1',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[reasoning_evaluator.gmean_f1(), reasoning_evaluator.f1()],
      eval_splits=['test'],
      eval_langs=['ar-x-levant'],
      domains=['speech'],
      task_subtypes=['reasoning'],
  )


class SVQArXMaghrebiSpanCrossLangReasoning(SVQSpanCrossLangReasoning):
  locale = 'ar_x_maghrebi'
  metadata = types.TaskMetadata(
      name='SVQArXMaghrebiSpanCrossLangReasoning',
      description='Span cross-lang reasoning task.',
      reference='TODO',
      type='SpanCrossLangReasoning',
      category='speech',
      main_score='GmeanF1',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[reasoning_evaluator.gmean_f1(), reasoning_evaluator.f1()],
      eval_splits=['test'],
      eval_langs=['ar-x-maghrebi'],
      domains=['speech'],
      task_subtypes=['reasoning'],
  )


class SVQBnBdSpanCrossLangReasoning(SVQSpanCrossLangReasoning):
  locale = 'bn_bd'
  metadata = types.TaskMetadata(
      name='SVQBnBdSpanCrossLangReasoning',
      description='Span cross-lang reasoning task.',
      reference='TODO',
      type='SpanCrossLangReasoning',
      category='speech',
      main_score='GmeanF1',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[reasoning_evaluator.gmean_f1(), reasoning_evaluator.f1()],
      eval_splits=['test'],
      eval_langs=['bn-BD'],
      domains=['speech'],
      task_subtypes=['reasoning'],
  )


class SVQBnInSpanCrossLangReasoning(SVQSpanCrossLangReasoning):
  locale = 'bn_in'
  metadata = types.TaskMetadata(
      name='SVQBnInSpanCrossLangReasoning',
      description='Span cross-lang reasoning task.',
      reference='TODO',
      type='SpanCrossLangReasoning',
      category='speech',
      main_score='GmeanF1',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[reasoning_evaluator.gmean_f1(), reasoning_evaluator.f1()],
      eval_splits=['test'],
      eval_langs=['bn-IN'],
      domains=['speech'],
      task_subtypes=['reasoning'],
  )


class SVQFiFiSpanCrossLangReasoning(SVQSpanCrossLangReasoning):
  locale = 'fi_fi'
  metadata = types.TaskMetadata(
      name='SVQFiFiSpanCrossLangReasoning',
      description='Span cross-lang reasoning task.',
      reference='TODO',
      type='SpanCrossLangReasoning',
      category='speech',
      main_score='GmeanF1',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[reasoning_evaluator.gmean_f1(), reasoning_evaluator.f1()],
      eval_splits=['test'],
      eval_langs=['fi-FI'],
      domains=['speech'],
      task_subtypes=['reasoning'],
  )


class SVQGuInSpanCrossLangReasoning(SVQSpanCrossLangReasoning):
  locale = 'gu_in'
  metadata = types.TaskMetadata(
      name='SVQGuInSpanCrossLangReasoning',
      description='Span cross-lang reasoning task.',
      reference='TODO',
      type='SpanCrossLangReasoning',
      category='speech',
      main_score='GmeanF1',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[reasoning_evaluator.gmean_f1(), reasoning_evaluator.f1()],
      eval_splits=['test'],
      eval_langs=['gu-IN'],
      domains=['speech'],
      task_subtypes=['reasoning'],
  )


class SVQHiInSpanCrossLangReasoning(SVQSpanCrossLangReasoning):
  locale = 'hi_in'
  metadata = types.TaskMetadata(
      name='SVQHiInSpanCrossLangReasoning',
      description='Span cross-lang reasoning task.',
      reference='TODO',
      type='SpanCrossLangReasoning',
      category='speech',
      main_score='GmeanF1',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[reasoning_evaluator.gmean_f1(), reasoning_evaluator.f1()],
      eval_splits=['test'],
      eval_langs=['hi-IN'],
      domains=['speech'],
      task_subtypes=['reasoning'],
  )


class SVQJaJpSpanCrossLangReasoning(SVQSpanCrossLangReasoning):
  locale = 'ja_jp'
  metadata = types.TaskMetadata(
      name='SVQJaJpSpanCrossLangReasoning',
      description='Span cross-lang reasoning task.',
      reference='TODO',
      type='SpanCrossLangReasoning',
      category='speech',
      main_score='GmeanF1',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[reasoning_evaluator.gmean_f1(), reasoning_evaluator.f1()],
      eval_splits=['test'],
      eval_langs=['ja-JP'],
      domains=['speech'],
      task_subtypes=['reasoning'],
  )


class SVQKnInSpanCrossLangReasoning(SVQSpanCrossLangReasoning):
  locale = 'kn_in'
  metadata = types.TaskMetadata(
      name='SVQKnInSpanCrossLangReasoning',
      description='Span cross-lang reasoning task.',
      reference='TODO',
      type='SpanCrossLangReasoning',
      category='speech',
      main_score='GmeanF1',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[reasoning_evaluator.gmean_f1(), reasoning_evaluator.f1()],
      eval_splits=['test'],
      eval_langs=['kn-IN'],
      domains=['speech'],
      task_subtypes=['reasoning'],
  )


class SVQKoKrSpanCrossLangReasoning(SVQSpanCrossLangReasoning):
  locale = 'ko_kr'
  metadata = types.TaskMetadata(
      name='SVQKoKrSpanCrossLangReasoning',
      description='Span cross-lang reasoning task.',
      reference='TODO',
      type='SpanCrossLangReasoning',
      category='speech',
      main_score='GmeanF1',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[reasoning_evaluator.gmean_f1(), reasoning_evaluator.f1()],
      eval_splits=['test'],
      eval_langs=['ko-KR'],
      domains=['speech'],
      task_subtypes=['reasoning'],
  )


class SVQMlInSpanCrossLangReasoning(SVQSpanCrossLangReasoning):
  locale = 'ml_in'
  metadata = types.TaskMetadata(
      name='SVQMlInSpanCrossLangReasoning',
      description='Span cross-lang reasoning task.',
      reference='TODO',
      type='SpanCrossLangReasoning',
      category='speech',
      main_score='GmeanF1',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[reasoning_evaluator.gmean_f1(), reasoning_evaluator.f1()],
      eval_splits=['test'],
      eval_langs=['ml-IN'],
      domains=['speech'],
      task_subtypes=['reasoning'],
  )


class SVQMrInSpanCrossLangReasoning(SVQSpanCrossLangReasoning):
  locale = 'mr_in'
  metadata = types.TaskMetadata(
      name='SVQMrInSpanCrossLangReasoning',
      description='Span cross-lang reasoning task.',
      reference='TODO',
      type='SpanCrossLangReasoning',
      category='speech',
      main_score='GmeanF1',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[reasoning_evaluator.gmean_f1(), reasoning_evaluator.f1()],
      eval_splits=['test'],
      eval_langs=['mr-IN'],
      domains=['speech'],
      task_subtypes=['reasoning'],
  )


class SVQRuRuSpanCrossLangReasoning(SVQSpanCrossLangReasoning):
  locale = 'ru_ru'
  metadata = types.TaskMetadata(
      name='SVQRuRuSpanCrossLangReasoning',
      description='Span cross-lang reasoning task.',
      reference='TODO',
      type='SpanCrossLangReasoning',
      category='speech',
      main_score='GmeanF1',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[reasoning_evaluator.gmean_f1(), reasoning_evaluator.f1()],
      eval_splits=['test'],
      eval_langs=['ru-RU'],
      domains=['speech'],
      task_subtypes=['reasoning'],
  )


class SVQTaInSpanCrossLangReasoning(SVQSpanCrossLangReasoning):
  locale = 'ta_in'
  metadata = types.TaskMetadata(
      name='SVQTaInSpanCrossLangReasoning',
      description='Span cross-lang reasoning task.',
      reference='TODO',
      type='SpanCrossLangReasoning',
      category='speech',
      main_score='GmeanF1',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[reasoning_evaluator.gmean_f1(), reasoning_evaluator.f1()],
      eval_splits=['test'],
      eval_langs=['ta-IN'],
      domains=['speech'],
      task_subtypes=['reasoning'],
  )


class SVQTeInSpanCrossLangReasoning(SVQSpanCrossLangReasoning):
  locale = 'te_in'
  metadata = types.TaskMetadata(
      name='SVQTeInSpanCrossLangReasoning',
      description='Span cross-lang reasoning task.',
      reference='TODO',
      type='SpanCrossLangReasoning',
      category='speech',
      main_score='GmeanF1',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[reasoning_evaluator.gmean_f1(), reasoning_evaluator.f1()],
      eval_splits=['test'],
      eval_langs=['te-IN'],
      domains=['speech'],
      task_subtypes=['reasoning'],
  )


class SVQUrInSpanCrossLangReasoning(SVQSpanCrossLangReasoning):
  locale = 'ur_in'
  metadata = types.TaskMetadata(
      name='SVQUrInSpanCrossLangReasoning',
      description='Span cross-lang reasoning task.',
      reference='TODO',
      type='SpanCrossLangReasoning',
      category='speech',
      main_score='GmeanF1',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[reasoning_evaluator.gmean_f1(), reasoning_evaluator.f1()],
      eval_splits=['test'],
      eval_langs=['ur-IN'],
      domains=['speech'],
      task_subtypes=['reasoning'],
  )


class SVQUrPkSpanCrossLangReasoning(SVQSpanCrossLangReasoning):
  locale = 'ur_pk'
  metadata = types.TaskMetadata(
      name='SVQUrPkSpanCrossLangReasoning',
      description='Span cross-lang reasoning task.',
      reference='TODO',
      type='SpanCrossLangReasoning',
      category='speech',
      main_score='GmeanF1',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[reasoning_evaluator.gmean_f1(), reasoning_evaluator.f1()],
      eval_splits=['test'],
      eval_langs=['ur-PK'],
      domains=['speech'],
      task_subtypes=['reasoning'],
  )
