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

"""SVQ document in-lang retrieval tasks."""

import os
from typing import Any, Iterable

from mseb import task as task_lib
from mseb import types
from mseb.datasets import simple_voice_questions as svq
from mseb.evaluators import retrieval_evaluator
from mseb.tasks import retrieval
from mseb.tasks.retrievals import utils
import tensorflow_datasets as tfds

_filter_fn_by_sub_task = {
    'document_retrieval_in_lang': lambda x: True,
    'document_retrieval_in_lang:clean': lambda x: x['environment'] == 'clean',
    'document_retrieval_in_lang:media_noise': (
        lambda x: x['environment'] == 'media_noise'
    ),
    'document_retrieval_in_lang:traffic_noise': (
        lambda x: x['environment'] == 'traffic_noise'
    ),
    'document_retrieval_in_lang:background_speech': (
        lambda x: x['environment'] == 'background_speech'
    ),
}


def _base_sub_task(sub_task: str) -> str:
  return sub_task.split(':')[0]


class SVQDocumentInLangRetrieval(retrieval.RetrievalTask):
  """SVQ document in-lang retrieval."""

  locale: str | None = None

  def _get_svq_dataset(self) -> svq.SimpleVoiceQuestionsDataset:
    return svq.SimpleVoiceQuestionsDataset()

  @property
  def index_dir(self) -> str:
    return os.path.join(
        super().index_dir, f'svq_{self.language}_document_retrieval_in_lang'
    )

  @property
  def language(self) -> str:
    assert self.locale is not None
    return self.locale.split('_')[0]

  @property
  def sub_tasks(self) -> list[str]:
    return list(_filter_fn_by_sub_task.keys())

  def get_documents_source(self) -> Any:
    return f'wikipedia/20190301.{self.language}'

  @staticmethod
  def documents_generator(dataset: Any) -> Iterable[types.Text]:
    ds = tfds.load(dataset, split='train')
    for example in ds.as_numpy_iterator():
      title = example['title'].decode('utf-8')
      yield types.Text(
          text=example['text'].decode('utf-8'),
          context=types.TextContextParams(id=title, title=title),
      )

  def multimodal_inputs(self) -> Iterable[types.Sound]:
    truncation = None
    backfill = None
    svq_dataset = self._get_svq_dataset()
    for example in svq_dataset.get_task_data(
        'document_retrieval_in_lang',
        dtype={
            'locale': str,
            'utt_id': str,
            task_lib.TRANSCRIPT_KEY.value: str,
        },
    ).to_dict('records'):
      if example['locale'] == self.locale:
        sound = svq_dataset.get_sound({'utt_id': example['utt_id']})
        sound.context.text = example[task_lib.TRANSCRIPT_KEY.value]
        if retrieval.RETRIEVED_ITEMS_KEY.value:
          if backfill is None:
            backfill = utils.BackFillRetrievedItemTexts(
                self.documents(),
                utils.BackFillRetrievedItemTexts.get_empty_text_by_id([
                    x.get(retrieval.RETRIEVED_ITEMS_KEY.value)
                    for x in svq_dataset.get_task_data(
                        'document_retrieval_in_lang', dtype={'utt_id': str}
                    ).to_dict('records')
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
    svq_dataset = self._get_svq_dataset()
    for example in svq_dataset.get_task_data(
        _base_sub_task(sub_task),
        dtype={'locale': str, 'utt_id': str, 'page_title': str},
    ).to_dict('records'):
      if example['locale'] == self.locale and filter_fn(example):
        yield retrieval_evaluator.RetrievalReferenceId(
            sound_id=example['utt_id'], reference_id=example['page_title']
        )


class SVQDocumentInLangRetrievalSmallIndex(SVQDocumentInLangRetrieval):
  """SVQ document in-lang retrieval with small index."""

  @property
  def index_dir(self) -> str:
    return os.path.join(
        super(SVQDocumentInLangRetrieval, self).index_dir,
        'svq_document_retrieval_in_lang_small_index',
    )

  def get_documents_source(self) -> Any:
    return self._get_svq_dataset()

  @staticmethod
  def documents_generator(dataset: Any) -> Iterable[types.Text]:
    for example in dataset.get_task_data(
        'document_retrieval_in_lang_small_index',
        dtype={'title': str, 'text': str},
    ).to_dict('records'):
      yield types.Text(
          text=example['text'],
          context=types.TextContextParams(
              id=example['title'],
              title=example['title'],
          ),
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
              type='DocumentInLangRetrieval',
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
for _locale, (_suffix, _eval_lang) in _SVQ_LOCALES.items():
  # Full index variant.
  _cls = _make_task_class(  # pylint: disable=invalid-name
      base_cls=SVQDocumentInLangRetrieval,
      locale=_locale,
      suffix=_suffix,
      eval_lang=_eval_lang,
      description='Document in-lang retrieval task.',
  )
  globals()[_cls.__name__] = _cls

  # Small index variant.
  _cls = _make_task_class(  # pylint: disable=invalid-name
      base_cls=SVQDocumentInLangRetrievalSmallIndex,
      locale=_locale,
      suffix=_suffix,
      eval_lang=_eval_lang,
      description='Document in-lang retrieval task with small index.',
  )
  globals()[_cls.__name__] = _cls
