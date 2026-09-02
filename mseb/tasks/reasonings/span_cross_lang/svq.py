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

"""SVQ span cross-lang reasoning tasks."""

import functools
import os
from typing import Any, Iterable, Sequence

from mseb import types
from mseb.datasets import simple_voice_questions as svq
from mseb.evaluators import reasoning_evaluator
from mseb.tasks import reasoning

_filter_fn_by_sub_task = {
    'span_reasoning_cross_lang': lambda x: True,
    'span_reasoning_cross_lang:clean': lambda x: x['environment'] == 'clean',
    'span_reasoning_cross_lang:media_noise': (
        lambda x: x['environment'] == 'media_noise'
    ),
    'span_reasoning_cross_lang:traffic_noise': (
        lambda x: x['environment'] == 'traffic_noise'
    ),
    'span_reasoning_cross_lang:background_speech': (
        lambda x: x['environment'] == 'background_speech'
    ),
}


def _base_sub_task(sub_task: str) -> str:
  return sub_task.split(':')[0]


class SVQSpanCrossLangReasoning(reasoning.ReasoningTask):
  """SVQ span cross-lang reasoning task."""

  locale: str | None = None

  @functools.cached_property
  def svq_dataset(self) -> svq.SimpleVoiceQuestionsDataset:
    return svq.SimpleVoiceQuestionsDataset()

  @property
  def embeddings_dir(self) -> str:
    assert self.locale is not None
    name = f'svq_{self.locale}_span_reasoning_cross_lang'
    return os.path.join(super().embeddings_dir, name)

  def _task_data(self, task_data_key: str, dtype: dict[str, Any] | None = None):
    df = self.svq_dataset.get_task_data(task_data_key, dtype=dtype)
    if self.locale:
      df = df[df.locale == self.locale]
    return df

  @property
  def sub_tasks(self) -> list[str]:
    return list(_filter_fn_by_sub_task.keys())

  def multimodal_inputs(self) -> Iterable[types.SoundWithTitleAndContext]:
    df = self._task_data(
        'span_reasoning_cross_lang',
        dtype={
            'locale': str,
            'utt_id': str,
            'page_title': str,
            reasoning.CONTEXT_KEY.value: str,
        },
    )
    for example in df.to_dict('records'):
      sound = self.svq_dataset.get_sound(example)
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
    df = self._task_data(
        'span_reasoning_cross_lang',
        dtype={
            'locale': str,
            'utt_id': str,
            'span': str,
            'spans': Sequence[str],
        },
    )
    for example in df.to_dict('records'):
      if filter_fn(example):
        yield reasoning_evaluator.ReasoningSpans(
            sound_id=example['utt_id'],
            reference_answer=example['span'],
            texts=example['spans'],
        )

  def span_lists(self) -> Iterable[Sequence[types.Text]]:
    df = self._task_data(
        'span_reasoning_cross_lang',
        dtype={
            'locale': str,
            'spans': Sequence[str],
        },
    )
    for example in df.to_dict('records'):
      yield [
          types.Text(
              text=span,
              context=types.TextContextParams(id=span),
          )
          for span in example['spans']
      ]


# Locale -> (ClassName suffix, eval_lang)
_SVQ_LOCALES = {
    'ar_eg': ('ArEg', 'ar-EG'),
    'ar_x_gulf': ('ArXGulf', 'ar-x-gulf'),
    'ar_x_levant': ('ArXLevant', 'ar-x-levant'),
    'ar_x_maghrebi': ('ArXMaghrebi', 'ar-x-maghrebi'),
    'bn_bd': ('BnBd', 'bn-BD'),
    'bn_in': ('BnIn', 'bn-IN'),
    'fi_fi': ('FiFi', 'fi-FI'),
    'gu_in': ('GuIn', 'gu-IN'),
    'hi_in': ('HiIn', 'hi-IN'),
    'ja_jp': ('JaJp', 'ja-JP'),
    'kn_in': ('KnIn', 'kn-IN'),
    'ko_kr': ('KoKr', 'ko-KR'),
    'ml_in': ('MlIn', 'ml-IN'),
    'mr_in': ('MrIn', 'mr-IN'),
    'ru_ru': ('RuRu', 'ru-RU'),
    'ta_in': ('TaIn', 'ta-IN'),
    'te_in': ('TeIn', 'te-IN'),
    'ur_in': ('UrIn', 'ur-IN'),
    'ur_pk': ('UrPk', 'ur-PK'),
}


def _make_task_class(
    base_cls,
    locale,
    suffix,
    eval_lang,
    description,
):
  """Dynamically create a locale-specific task class."""
  class_name = f'SVQ{suffix}{base_cls.__name__[len("SVQ"):]}'
  cls = type(
      class_name,
      (base_cls,),
      {
          'locale': locale,
          'metadata': types.TaskMetadata(
              name=class_name,
              description=description,
              reference='https://huggingface.co/datasets/google/svq',
              documentation_file='svq_retrieval.md',
              dataset_documentation_file='dataset_svq.md',
              type='SpanCrossLangReasoning',
              category='speech',
              main_score='GmeanF1',
              revision='1.0.0',
              dataset=types.Dataset(
                  name='SVQ',
                  path='https://huggingface.co/datasets/google/svq',
                  revision='1.0.0',
              ),
              scores=[reasoning_evaluator.gmean_f1(), reasoning_evaluator.f1()],
              eval_splits=['test'],
              eval_langs=[eval_lang],
              domains=['speech'],
              task_subtypes=['reasoning'],
          ),
      },
  )
  return cls


# Generate all locale-specific classes and register them in the module.
# Default size.
for _locale, (_suffix, _eval_lang) in _SVQ_LOCALES.items():
  _cls = _make_task_class(  # pylint: disable=invalid-name
      base_cls=SVQSpanCrossLangReasoning,
      locale=_locale,
      suffix=_suffix,
      eval_lang=_eval_lang,
      description='Span cross-lang reasoning task.',
  )
  globals()[_cls.__name__] = _cls
