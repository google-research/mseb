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

"""SVQ passage in-lang retrieval tasks."""

import functools
import os
from typing import Any, Iterable

from mseb import task as task_lib
from mseb import types
from mseb.datasets import simple_voice_questions as svq
from mseb.evaluators import retrieval_evaluator
from mseb.tasks import retrieval
from mseb.tasks.retrievals import utils

_filter_fn_by_sub_task = {
    'passage_retrieval_in_lang': lambda x: True,
    'passage_retrieval_in_lang:clean': lambda x: x['environment'] == 'clean',
    'passage_retrieval_in_lang:media_noise': (
        lambda x: x['environment'] == 'media_noise'
    ),
    'passage_retrieval_in_lang:traffic_noise': (
        lambda x: x['environment'] == 'traffic_noise'
    ),
    'passage_retrieval_in_lang:background_speech': (
        lambda x: x['environment'] == 'background_speech'
    ),
}


def _base_sub_task(sub_task: str) -> str:
  return sub_task.split(':')[0]


class SVQPassageInLangRetrieval(retrieval.RetrievalTask):
  """SVQ passage in-lang retrieval."""

  locale: str | None = None

  @functools.cached_property
  def svq_dataset(self) -> svq.SimpleVoiceQuestionsDataset:
    return svq.SimpleVoiceQuestionsDataset()

  @property
  def index_dir(self) -> str:
    return os.path.join(super().index_dir, 'svq_passage_retrieval_in_lang')

  @property
  def sub_tasks(self) -> list[str]:
    return list(_filter_fn_by_sub_task.keys())

  def _task_data(self, task_data_key: str, dtype: dict[str, Any] | None = None):
    df = self.svq_dataset.get_task_data(task_data_key, dtype=dtype)
    if self.locale:
      df = df[df.locale == self.locale]
    return df

  def get_documents_source(self) -> svq.SimpleVoiceQuestionsDataset:
    return self.svq_dataset

  @staticmethod
  def documents_generator(svq_dataset: Any) -> Iterable[types.Text]:
    """Yields Text documents from the given SVQ dataset index."""
    df = svq_dataset.get_task_data(
        'passage_retrieval_in_lang_index',
        dtype={'id': str, 'title': str, 'context': str},
    )
    for example in df.to_dict('records'):
      yield types.Text(
          text=example['context'],
          context=types.TextContextParams(
              id=example['id'],
              title=example['title'],
          ),
      )

  def multimodal_inputs(self) -> Iterable[types.Sound]:
    truncation = None
    backfill = None
    df = self._task_data(
        'passage_retrieval_in_lang',
        dtype={
            'locale': str,
            'utt_id': str,
            task_lib.TRANSCRIPT_KEY.value: str,
        },
    )
    for example in df.to_dict('records'):
      sound = self.svq_dataset.get_sound({'utt_id': example['utt_id']})
      sound.context.text = example[task_lib.TRANSCRIPT_KEY.value]
      if retrieval.RETRIEVED_ITEMS_KEY.value:
        if backfill is None:
          backfill_df = self._task_data(
              'passage_retrieval_in_lang', dtype={'utt_id': str}
          )
          backfill = utils.BackFillRetrievedItemTexts(
              self.documents(),
              utils.BackFillRetrievedItemTexts.get_empty_text_by_id([
                  x.get(retrieval.RETRIEVED_ITEMS_KEY.value)
                  for x in backfill_df.to_dict('records')
              ]),
          )
        context_text = backfill.backfill(
            example.get(retrieval.RETRIEVED_ITEMS_KEY.value)
        )
        if utils.MAX_CONTEXT_TOKENS.value and utils.TOKENIZER_NAME.value:
          if truncation is None:
            truncation = utils.ListPredictionTruncation(
                max_tokens=utils.MAX_CONTEXT_TOKENS.value,
                tokenizer_name=utils.TOKENIZER_NAME.value,
            )
          context_text = truncation.maybe_truncate(context_text)
        sound = types.SoundWithTitleAndContext(
            waveform=sound.waveform,
            context=sound.context,
            context_text=context_text,
        )
      yield sound

  def examples(
      self, sub_task: str
  ) -> Iterable[retrieval_evaluator.RetrievalReferenceId]:
    filter_fn = _filter_fn_by_sub_task[sub_task]
    df = self._task_data(
        _base_sub_task(sub_task),
        dtype={'locale': str, 'utt_id': str, 'passage_id': str},
    )
    for example in df.to_dict('records'):
      if filter_fn(example):
        yield retrieval_evaluator.RetrievalReferenceId(
            sound_id=example['utt_id'], reference_id=example['passage_id']
        )


# Locale -> (ClassName suffix, eval_lang)
_SVQ_LOCALES = {
    'ar_eg': ('ArEg', 'ar-EG'),
    'ar_x_gulf': ('ArXGulf', 'ar-x-gulf'),
    'ar_x_levant': ('ArXLevant', 'ar-x-levant'),
    'ar_x_maghrebi': ('ArXMaghrebi', 'ar-x-maghrebi'),
    'bn_bd': ('BnBd', 'bn-BD'),
    'bn_in': ('BnIn', 'bn-IN'),
    'en_au': ('EnAu', 'en-AU'),
    'en_gb': ('EnGb', 'en-GB'),
    'en_in': ('EnIn', 'en-IN'),
    'en_ph': ('EnPh', 'en-PH'),
    'en_us': ('EnUs', 'en-US'),
    'fi_fi': ('FiFi', 'fi-FI'),
    'id_id': ('IdId', 'id-ID'),
    'ko_kr': ('KoKr', 'ko-KR'),
    'ru_ru': ('RuRu', 'ru-RU'),
    'sw': ('Sw', 'sw'),
    'te_in': ('TeIn', 'te-IN'),
}


def _make_task_class(base_cls, locale, suffix, eval_lang, description):
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
              type='PassageInLangRetrieval',
              category='speech',
              main_score='MRR',
              revision='1.0.0',
              dataset=types.Dataset(
                  name='SVQ',
                  path='https://huggingface.co/datasets/google/svq',
                  revision='1.0.0',
              ),
              scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
              eval_splits=['test'],
              eval_langs=[eval_lang],
              domains=['speech'],
              task_subtypes=['retrieval'],
          ),
      },
  )
  return cls


# Generate all locale-specific classes and register them in the module.
# Default size.
for _locale, (_suffix, _eval_lang) in _SVQ_LOCALES.items():
  _cls = _make_task_class(  # pylint: disable=invalid-name
      base_cls=SVQPassageInLangRetrieval,
      locale=_locale,
      suffix=_suffix,
      eval_lang=_eval_lang,
      description='Passage in-lang retrieval task.',
  )
  globals()[_cls.__name__] = _cls
