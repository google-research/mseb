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


def _get_environment(sub_task: str) -> str:
  """Returns the environment for the given sub_task.

  Examples:
    'span_reasoning_cross_lang:clean' -> 'clean'
    'span_reasoning_cross_lang' -> '*'

  Args:
    sub_task: The sub_task name.

  Returns:
    The environment for the given sub_task.
  """
  sub_task_parts = sub_task.split(':')
  if len(sub_task_parts) == 1:
    return '*'
  elif len(sub_task_parts) == 2:
    return sub_task_parts[1]
  else:
    raise ValueError(f'Invalid sub_task: {sub_task}')


class SVQSpanCrossLangReasoning(reasoning.ReasoningTask):
  """SVQ span cross-lang reasoning task."""

  locale: str | None = None
  size: str | None = None

  @functools.cached_property
  def svq_dataset(self) -> svq.SimpleVoiceQuestionsDataset:
    return svq.SimpleVoiceQuestionsDataset()

  @property
  def embeddings_dir(self) -> str:
    assert self.locale is not None
    name = f'svq_{self.locale}_span_reasoning_cross_lang'
    if self.size is not None:
      name += f'_{self.size}'
    return os.path.join(super().embeddings_dir, name)

  @property
  def sub_tasks(self) -> list[str]:
    return [
        'span_reasoning_cross_lang',
        'span_reasoning_cross_lang:clean',
        'span_reasoning_cross_lang:media_noise',
        'span_reasoning_cross_lang:traffic_noise',
        'span_reasoning_cross_lang:background_speech',
    ]

  def _task_data(self, task_data_key: str, dtype: dict[str, Any] | None = None):
    df = self.svq_dataset.get_task_data(task_data_key, dtype=dtype)
    df = df[df['reasonings/span_cross_lang']]
    if self.size is not None:
      mask = df['span_context_id_cross_lang'].map(
          getattr(svq, f'is_member_of_{self.size}')
      )
      df = df[mask]
    return df

  def multimodal_inputs(self) -> Iterable[types.SoundWithTitleAndContext]:
    default_context_key = 'span_context_title_cross_lang'
    context_key = reasoning.CONTEXT_KEY.value or default_context_key
    df = self._task_data(
        f'utts_{self.locale}_*',
        dtype={
            'locale': str,
            'utt_id': str,
            'span_context_title_cross_lang': str,
            context_key: str,
        },
    )
    for example in df.to_dict('records'):
      sound = self.svq_dataset.get_sound(example)
      yield types.SoundWithTitleAndContext(
          waveform=sound.waveform,
          title_text=example['span_context_title_cross_lang'],
          context_text=example[context_key],
          context=sound.context,
      )

  def examples(
      self, sub_task: str
  ) -> Iterable[reasoning_evaluator.ReasoningSpans]:
    df = self._task_data(
        f'utts_{self.locale}_{_get_environment(sub_task)}',
        dtype={
            'locale': str,
            'utt_id': str,
            'span_cross_lang': str,
            'spans_cross_lang': Sequence[str],
        },
    )
    for example in df.to_dict('records'):
      yield reasoning_evaluator.ReasoningSpans(
          sound_id=example['utt_id'],
          reference_answer=example['span_cross_lang'],
          texts=example['spans_cross_lang'],
      )

  def span_lists(self) -> Iterable[Sequence[types.Text]]:
    df = self._task_data(
        f'utts_{self.locale}_*',
        dtype={
            'locale': str,
            'spans_cross_lang': Sequence[str],
        },
    )
    for example in df.to_dict('records'):
      yield [
          types.Text(
              text=span,
              context=types.TextContextParams(id=span),
          )
          for span in example['spans_cross_lang']
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
    size,
    suffix,
    eval_lang,
    description,
):
  """Dynamically create a locale-specific task class."""
  class_name = f'SVQ{suffix}{base_cls.__name__[len("SVQ"):]}'
  if size is not None:
    class_name += size.capitalize()
  cls = type(
      class_name,
      (base_cls,),
      {
          'locale': locale,
          'size': size,
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
      size=None,
      suffix=_suffix,
      eval_lang=_eval_lang,
      description='Span cross-lang reasoning task.',
  )
  globals()[_cls.__name__] = _cls


# Compact size.
for _locale, (_suffix, _eval_lang) in _SVQ_LOCALES.items():
  _cls = _make_task_class(  # pylint: disable=invalid-name
      base_cls=SVQSpanCrossLangReasoning,
      locale=_locale,
      size='compact',
      suffix=_suffix,
      eval_lang=_eval_lang,
      description='Span cross-lang reasoning task.',
  )
  globals()[_cls.__name__] = _cls

# Debug size.
for _locale, (_suffix, _eval_lang) in {
    'en_us': ('EnUs', 'en-US'),
    'fi_fi': ('FiFi', 'fi-FI'),
}.items():
  _cls = _make_task_class(  # pylint: disable=invalid-name
      base_cls=SVQSpanCrossLangReasoning,
      locale=_locale,
      size='debug',
      suffix=_suffix,
      eval_lang=_eval_lang,
      description='Span cross-lang reasoning task.',
  )
  globals()[_cls.__name__] = _cls
