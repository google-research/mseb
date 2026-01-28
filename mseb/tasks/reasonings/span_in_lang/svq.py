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


_filter_fn_by_sub_task = {
    'span_reasoning_in_lang': lambda x: True,
    'span_reasoning_in_lang:clean': lambda x: x['environment'] == 'clean',
    'span_reasoning_in_lang:media_noise': (
        lambda x: x['environment'] == 'media_noise'
    ),
    'span_reasoning_in_lang:traffic_noise': (
        lambda x: x['environment'] == 'traffic_noise'
    ),
    'span_reasoning_in_lang:background_speech': (
        lambda x: x['environment'] == 'background_speech'
    ),
}


def _base_sub_task(sub_task: str) -> str:
  return sub_task.split(':')[0]


class SVQSpanInLangReasoning(reasoning.ReasoningTask):
  """SVQ span in-lang reasoning task."""

  locale: str | None = None

  def _get_dataset(self) -> svq.SimpleVoiceQuestionsDataset:
    return svq.SimpleVoiceQuestionsDataset()

  @property
  def embeddings_dir(self) -> str:
    assert self.locale is not None
    return os.path.join(
        super().embeddings_dir, f'svq_{self.locale}_span_reasoning_in_lang'
    )

  @property
  def sub_tasks(self) -> list[str]:
    return list(_filter_fn_by_sub_task.keys())

  def sounds(self) -> Iterable[types.SoundWithTitleAndContext]:
    svq_dataset = self._get_dataset()
    for example in svq_dataset.get_task_data(
        'span_reasoning_in_lang',
        dtype={
            'locale': str,
            'utt_id': str,
            'page_title': str,
            reasoning.CONTEXT_KEY.value: str,
        },
    ).to_dict('records'):
      if example['locale'] == self.locale:
        sound = svq_dataset.get_sound(example)
        yield types.SoundWithTitleAndContext(
            waveform=sound.waveform,
            title_text=example['page_title'],
            context_text=example[reasoning.CONTEXT_KEY.value],
            context=sound.context,
        )

  def examples(
      self, sub_task: str
  ) -> Iterable[reasoning_evaluator.ReasoningSpans]:
    filter_fn = _filter_fn_by_sub_task[sub_task]
    svq_dataset = self._get_dataset()
    for example in svq_dataset.get_task_data(
        _base_sub_task(sub_task),
        dtype={
            'locale': str,
            'utt_id': str,
            'span': str,
            'spans': Sequence[str],
        },
    ).to_dict('records'):
      if example['locale'] == self.locale and filter_fn(example):
        yield reasoning_evaluator.ReasoningSpans(
            sound_id=example['utt_id'],
            reference_answer=example['span'],
            texts=example['spans'],
        )

  def span_lists(self) -> Iterable[Sequence[types.Text]]:
    svq_dataset = self._get_dataset()
    for example in svq_dataset.get_task_data(
        'span_reasoning_in_lang', dtype={'locale': str, 'spans': Sequence[str]}
    ).to_dict('records'):
      if example['locale'] == self.locale:
        yield [
            types.Text(
                text=span,
                context=types.TextContextParams(id=span),
            )
            for span in example['spans']
        ]


class SVQArEgSpanInLangReasoning(SVQSpanInLangReasoning):
  locale = 'ar_eg'
  metadata = types.TaskMetadata(
      name='SVQArEgSpanInLangReasoning',
      description='Span in-lang reasoning task.',
      reference='TODO',
      type='SpanInLangReasoning',
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


class SVQArXGulfSpanInLangReasoning(SVQSpanInLangReasoning):
  locale = 'ar_x_gulf'
  metadata = types.TaskMetadata(
      name='SVQArXGulfSpanInLangReasoning',
      description='Span in-lang reasoning task.',
      reference='TODO',
      type='SpanInLangReasoning',
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


class SVQArXLevantSpanInLangReasoning(SVQSpanInLangReasoning):
  locale = 'ar_x_levant'
  metadata = types.TaskMetadata(
      name='SVQArXLevantSpanInLangReasoning',
      description='Span in-lang reasoning task.',
      reference='TODO',
      type='SpanInLangReasoning',
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


class SVQArXMaghrebiSpanInLangReasoning(SVQSpanInLangReasoning):
  locale = 'ar_x_maghrebi'
  metadata = types.TaskMetadata(
      name='SVQArXMaghrebiSpanInLangReasoning',
      description='Span in-lang reasoning task.',
      reference='TODO',
      type='SpanInLangReasoning',
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


class SVQBnBdSpanInLangReasoning(SVQSpanInLangReasoning):
  locale = 'bn_bd'
  metadata = types.TaskMetadata(
      name='SVQBnBdSpanInLangReasoning',
      description='Span in-lang reasoning task.',
      reference='TODO',
      type='SpanInLangReasoning',
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


class SVQBnInSpanInLangReasoning(SVQSpanInLangReasoning):
  locale = 'bn_in'
  metadata = types.TaskMetadata(
      name='SVQBnInSpanInLangReasoning',
      description='Span in-lang reasoning task.',
      reference='TODO',
      type='SpanInLangReasoning',
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


class SVQEnAuSpanInLangReasoning(SVQSpanInLangReasoning):
  locale = 'en_au'
  metadata = types.TaskMetadata(
      name='SVQEnAuSpanInLangReasoning',
      description='Span in-lang reasoning task.',
      reference='TODO',
      type='SpanInLangReasoning',
      category='speech',
      main_score='GmeanF1',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[reasoning_evaluator.gmean_f1(), reasoning_evaluator.f1()],
      eval_splits=['test'],
      eval_langs=['en-AU'],
      domains=['speech'],
      task_subtypes=['reasoning'],
  )


class SVQEnGbSpanInLangReasoning(SVQSpanInLangReasoning):
  locale = 'en_gb'
  metadata = types.TaskMetadata(
      name='SVQEnGbSpanInLangReasoning',
      description='Span in-lang reasoning task.',
      reference='TODO',
      type='SpanInLangReasoning',
      category='speech',
      main_score='GmeanF1',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[reasoning_evaluator.gmean_f1(), reasoning_evaluator.f1()],
      eval_splits=['test'],
      eval_langs=['en-GB'],
      domains=['speech'],
      task_subtypes=['reasoning'],
  )


class SVQEnInSpanInLangReasoning(SVQSpanInLangReasoning):
  locale = 'en_in'
  metadata = types.TaskMetadata(
      name='SVQEnInSpanInLangReasoning',
      description='Span in-lang reasoning task.',
      reference='TODO',
      type='SpanInLangReasoning',
      category='speech',
      main_score='GmeanF1',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[reasoning_evaluator.gmean_f1(), reasoning_evaluator.f1()],
      eval_splits=['test'],
      eval_langs=['en-IN'],
      domains=['speech'],
      task_subtypes=['reasoning'],
  )


class SVQEnPhSpanInLangReasoning(SVQSpanInLangReasoning):
  locale = 'en_ph'
  metadata = types.TaskMetadata(
      name='SVQEnPhSpanInLangReasoning',
      description='Span in-lang reasoning task.',
      reference='TODO',
      type='SpanInLangReasoning',
      category='speech',
      main_score='GmeanF1',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[reasoning_evaluator.gmean_f1(), reasoning_evaluator.f1()],
      eval_splits=['test'],
      eval_langs=['en-PH'],
      domains=['speech'],
      task_subtypes=['reasoning'],
  )


class SVQEnUsSpanInLangReasoning(SVQSpanInLangReasoning):
  locale = 'en_us'
  metadata = types.TaskMetadata(
      name='SVQEnUsSpanInLangReasoning',
      description='Span in-lang reasoning task.',
      reference='TODO',
      type='SpanInLangReasoning',
      category='speech',
      main_score='GmeanF1',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[reasoning_evaluator.gmean_f1(), reasoning_evaluator.f1()],
      eval_splits=['test'],
      eval_langs=['en-US'],
      domains=['speech'],
      task_subtypes=['reasoning'],
  )


class SVQFiFiSpanInLangReasoning(SVQSpanInLangReasoning):
  locale = 'fi_fi'
  metadata = types.TaskMetadata(
      name='SVQFiFiSpanInLangReasoning',
      description='Span in-lang reasoning task.',
      reference='TODO',
      type='SpanInLangReasoning',
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


class SVQIdIdSpanInLangReasoning(SVQSpanInLangReasoning):
  locale = 'id_id'
  metadata = types.TaskMetadata(
      name='SVQIdIdSpanInLangReasoning',
      description='Span in-lang reasoning task.',
      reference='TODO',
      type='SpanInLangReasoning',
      category='speech',
      main_score='GmeanF1',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[reasoning_evaluator.gmean_f1(), reasoning_evaluator.f1()],
      eval_splits=['test'],
      eval_langs=['id-ID'],
      domains=['speech'],
      task_subtypes=['reasoning'],
  )


class SVQKoKrSpanInLangReasoning(SVQSpanInLangReasoning):
  locale = 'ko_kr'
  metadata = types.TaskMetadata(
      name='SVQKoKrSpanInLangReasoning',
      description='Span in-lang reasoning task.',
      reference='TODO',
      type='SpanInLangReasoning',
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


class SVQRuRuSpanInLangReasoning(SVQSpanInLangReasoning):
  locale = 'ru_ru'
  metadata = types.TaskMetadata(
      name='SVQRuRuSpanInLangReasoning',
      description='Span in-lang reasoning task.',
      reference='TODO',
      type='SpanInLangReasoning',
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


class SVQSwSpanInLangReasoning(SVQSpanInLangReasoning):
  locale = 'sw'
  metadata = types.TaskMetadata(
      name='SVQSwSpanInLangReasoning',
      description='Span in-lang reasoning task.',
      reference='TODO',
      type='SpanInLangReasoning',
      category='speech',
      main_score='GmeanF1',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[reasoning_evaluator.gmean_f1(), reasoning_evaluator.f1()],
      eval_splits=['test'],
      eval_langs=['sw'],
      domains=['speech'],
      task_subtypes=['reasoning'],
  )


class SVQTeInSpanInLangReasoning(SVQSpanInLangReasoning):
  locale = 'te_in'
  metadata = types.TaskMetadata(
      name='SVQTeInSpanInLangReasoning',
      description='Span in-lang reasoning task.',
      reference='TODO',
      type='SpanInLangReasoning',
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
