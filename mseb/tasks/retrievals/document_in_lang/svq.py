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

  def get_documents_source(self) -> str:
    return f'wikipedia/20190301.{self.language}'

  @staticmethod
  def documents_generator(wikipedia_dataset: Any) -> Iterable[types.Text]:
    ds = tfds.load(wikipedia_dataset, split='train')
    for example in ds.as_numpy_iterator():
      title = example['title'].decode('utf-8')
      yield types.Text(
          text=example['text'].decode('utf-8'),
          context=types.TextContextParams(id=title, title=title),
      )

  def sounds(self) -> Iterable[types.Sound]:
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
          sound = types.SoundWithTitleAndContext(
              waveform=sound.waveform,
              context=sound.context,
              context_text=example.get(retrieval.RETRIEVED_ITEMS_KEY.value),
          )
        yield sound

  def examples(
      self, sub_task: str
  ) -> Iterable[retrieval_evaluator.RetrievalReferenceId]:
    filter_fn = _filter_fn_by_sub_task[sub_task]
    svq_dataset = self._get_svq_dataset()
    for example in svq_dataset.get_task_data(
        _base_sub_task(sub_task),
        dtype={'locale': str, 'utt_id': str, 'page_title': str}
    ).to_dict('records'):
      if example['locale'] == self.locale and filter_fn(example):
        yield retrieval_evaluator.RetrievalReferenceId(
            sound_id=example['utt_id'], reference_id=example['page_title']
        )


class SVQArEgDocumentInLangRetrieval(SVQDocumentInLangRetrieval):
  locale = 'ar_eg'
  metadata = types.TaskMetadata(
      name='SVQArEgDocumentInLangRetrieval',
      description='Document in-lang retrieval task.',
      reference='TODO',
      type='DocumentInLangRetrieval',
      category='speech',
      main_score='MRR',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
      eval_splits=['test'],
      eval_langs=['ar-EG'],
      domains=['speech'],
      task_subtypes=['retrieval'],
  )


class SVQArXGulfDocumentInLangRetrieval(SVQDocumentInLangRetrieval):
  locale = 'ar_x_gulf'
  metadata = types.TaskMetadata(
      name='SVQArXGulfDocumentInLangRetrieval',
      description='Document in-lang retrieval task.',
      reference='TODO',
      type='DocumentInLangRetrieval',
      category='speech',
      main_score='MRR',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
      eval_splits=['test'],
      eval_langs=['ar-x-gulf'],
      domains=['speech'],
      task_subtypes=['retrieval'],
  )


class SVQArXLevantDocumentInLangRetrieval(SVQDocumentInLangRetrieval):
  locale = 'ar_x_levant'
  metadata = types.TaskMetadata(
      name='SVQArXLevantDocumentInLangRetrieval',
      description='Document in-lang retrieval task.',
      reference='TODO',
      type='DocumentInLangRetrieval',
      category='speech',
      main_score='MRR',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
      eval_splits=['test'],
      eval_langs=['ar-x-levant'],
      domains=['speech'],
      task_subtypes=['retrieval'],
  )


class SVQArXMaghrebiDocumentInLangRetrieval(SVQDocumentInLangRetrieval):
  locale = 'ar_x_maghrebi'
  metadata = types.TaskMetadata(
      name='SVQArXMaghrebiDocumentInLangRetrieval',
      description='Document in-lang retrieval task.',
      reference='TODO',
      type='DocumentInLangRetrieval',
      category='speech',
      main_score='MRR',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
      eval_splits=['test'],
      eval_langs=['ar-x-maghrebi'],
      domains=['speech'],
      task_subtypes=['retrieval'],
  )


class SVQBnBdDocumentInLangRetrieval(SVQDocumentInLangRetrieval):
  locale = 'bn_bd'
  metadata = types.TaskMetadata(
      name='SVQBnBdDocumentInLangRetrieval',
      description='Document in-lang retrieval task.',
      reference='TODO',
      type='DocumentInLangRetrieval',
      category='speech',
      main_score='MRR',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
      eval_splits=['test'],
      eval_langs=['bn-BD'],
      domains=['speech'],
      task_subtypes=['retrieval'],
  )


class SVQBnInDocumentInLangRetrieval(SVQDocumentInLangRetrieval):
  locale = 'bn_in'
  metadata = types.TaskMetadata(
      name='SVQBnInDocumentInLangRetrieval',
      description='Document in-lang retrieval task.',
      reference='TODO',
      type='DocumentInLangRetrieval',
      category='speech',
      main_score='MRR',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
      eval_splits=['test'],
      eval_langs=['bn-IN'],
      domains=['speech'],
      task_subtypes=['retrieval'],
  )


class SVQEnAuDocumentInLangRetrieval(SVQDocumentInLangRetrieval):
  locale = 'en_au'
  metadata = types.TaskMetadata(
      name='SVQEnAuDocumentInLangRetrieval',
      description='Document in-lang retrieval task.',
      reference='TODO',
      type='DocumentInLangRetrieval',
      category='speech',
      main_score='MRR',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
      eval_splits=['test'],
      eval_langs=['en-AU'],
      domains=['speech'],
      task_subtypes=['retrieval'],
  )


class SVQEnGbDocumentInLangRetrieval(SVQDocumentInLangRetrieval):
  locale = 'en_gb'
  metadata = types.TaskMetadata(
      name='SVQEnGbDocumentInLangRetrieval',
      description='Document in-lang retrieval task.',
      reference='TODO',
      type='DocumentInLangRetrieval',
      category='speech',
      main_score='MRR',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
      eval_splits=['test'],
      eval_langs=['en-GB'],
      domains=['speech'],
      task_subtypes=['retrieval'],
  )


class SVQEnInDocumentInLangRetrieval(SVQDocumentInLangRetrieval):
  locale = 'en_in'
  metadata = types.TaskMetadata(
      name='SVQEnInDocumentInLangRetrieval',
      description='Document in-lang retrieval task.',
      reference='TODO',
      type='DocumentInLangRetrieval',
      category='speech',
      main_score='MRR',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
      eval_splits=['test'],
      eval_langs=['en-IN'],
      domains=['speech'],
      task_subtypes=['retrieval'],
  )


class SVQEnPhDocumentInLangRetrieval(SVQDocumentInLangRetrieval):
  locale = 'en_ph'
  metadata = types.TaskMetadata(
      name='SVQEnPhDocumentInLangRetrieval',
      description='Document in-lang retrieval task.',
      reference='TODO',
      type='DocumentInLangRetrieval',
      category='speech',
      main_score='MRR',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
      eval_splits=['test'],
      eval_langs=['en-PH'],
      domains=['speech'],
      task_subtypes=['retrieval'],
  )


class SVQEnUsDocumentInLangRetrieval(SVQDocumentInLangRetrieval):
  locale = 'en_us'
  metadata = types.TaskMetadata(
      name='SVQEnUsDocumentInLangRetrieval',
      description='Document in-lang retrieval task.',
      reference='TODO',
      type='DocumentInLangRetrieval',
      category='speech',
      main_score='MRR',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
      eval_splits=['test'],
      eval_langs=['en-US'],
      domains=['speech'],
      task_subtypes=['retrieval'],
  )


class SVQFiFiDocumentInLangRetrieval(SVQDocumentInLangRetrieval):
  locale = 'fi_fi'
  metadata = types.TaskMetadata(
      name='SVQFiFiDocumentInLangRetrieval',
      description='Document in-lang retrieval task.',
      reference='TODO',
      type='DocumentInLangRetrieval',
      category='speech',
      main_score='MRR',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
      eval_splits=['test'],
      eval_langs=['fi-FI'],
      domains=['speech'],
      task_subtypes=['retrieval'],
  )


class SVQIdIdDocumentInLangRetrieval(SVQDocumentInLangRetrieval):
  locale = 'id_id'
  metadata = types.TaskMetadata(
      name='SVQIdIdDocumentInLangRetrieval',
      description='Document in-lang retrieval task.',
      reference='TODO',
      type='DocumentInLangRetrieval',
      category='speech',
      main_score='MRR',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
      eval_splits=['test'],
      eval_langs=['id-ID'],
      domains=['speech'],
      task_subtypes=['retrieval'],
  )


class SVQKoKrDocumentInLangRetrieval(SVQDocumentInLangRetrieval):
  locale = 'ko_kr'
  metadata = types.TaskMetadata(
      name='SVQKoKrDocumentInLangRetrieval',
      description='Document in-lang retrieval task.',
      reference='TODO',
      type='DocumentInLangRetrieval',
      category='speech',
      main_score='MRR',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
      eval_splits=['test'],
      eval_langs=['ko-KR'],
      domains=['speech'],
      task_subtypes=['retrieval'],
  )


class SVQRuRuDocumentInLangRetrieval(SVQDocumentInLangRetrieval):
  locale = 'ru_ru'
  metadata = types.TaskMetadata(
      name='SVQRuRuDocumentInLangRetrieval',
      description='Document in-lang retrieval task.',
      reference='TODO',
      type='DocumentInLangRetrieval',
      category='speech',
      main_score='MRR',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
      eval_splits=['test'],
      eval_langs=['ru-RU'],
      domains=['speech'],
      task_subtypes=['retrieval'],
  )


class SVQSwDocumentInLangRetrieval(SVQDocumentInLangRetrieval):
  locale = 'sw'
  metadata = types.TaskMetadata(
      name='SVQSwDocumentInLangRetrieval',
      description='Document in-lang retrieval task.',
      reference='TODO',
      type='DocumentInLangRetrieval',
      category='speech',
      main_score='MRR',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
      eval_splits=['test'],
      eval_langs=['sw'],
      domains=['speech'],
      task_subtypes=['retrieval'],
  )


class SVQTeInDocumentInLangRetrieval(SVQDocumentInLangRetrieval):
  locale = 'te_in'
  metadata = types.TaskMetadata(
      name='SVQTeInDocumentInLangRetrieval',
      description='Document in-lang retrieval task.',
      reference='TODO',
      type='DocumentInLangRetrieval',
      category='speech',
      main_score='MRR',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
      eval_splits=['test'],
      eval_langs=['te-IN'],
      domains=['speech'],
      task_subtypes=['retrieval'],
  )
