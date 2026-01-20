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

"""SVQ document cross-lang retrieval tasks."""

import os
from typing import Any, Iterable

from mseb import task as task_lib
from mseb import types
from mseb.datasets import simple_voice_questions as svq
from mseb.evaluators import retrieval_evaluator
from mseb.tasks import retrieval
import tensorflow_datasets as tfds


class SVQDocumentCrossLangRetrieval(retrieval.RetrievalTask):
  """SVQ document cross-lang retrieval."""

  locale: str | None = None

  def _get_svq_dataset(self) -> svq.SimpleVoiceQuestionsDataset:
    return svq.SimpleVoiceQuestionsDataset()

  @property
  def index_dir(self) -> str:
    return os.path.join(super().index_dir, 'svq_en_document_retrieval_in_lang')

  @property
  def sub_tasks(self) -> list[str]:
    return ['document_retrieval_cross_lang']

  def sounds(self) -> Iterable[types.Sound]:
    svq_dataset = self._get_svq_dataset()
    for example in svq_dataset.get_task_data(
        'document_retrieval_cross_lang',
        dtype={
            'locale': str,
            'utt_id': str,
            task_lib.TRANSCRIPT_KEY.value: str,
        },
    ).to_dict('records'):
      if example['locale'] == self.locale:
        sound = svq_dataset.get_sound({'utt_id': example['utt_id']})
        # Add the ground truth query for headroom analysis.
        sound.context.text = example[task_lib.TRANSCRIPT_KEY.value]
        if retrieval.RETRIEVED_ITEMS_KEY.value:
          sound = types.SoundWithTitleAndContext(
              waveform=sound.waveform,
              context=sound.context,
              context_text=example.get(retrieval.RETRIEVED_ITEMS_KEY.value)
          )
        yield sound

  def examples(
      self, sub_task: str
  ) -> Iterable[retrieval_evaluator.RetrievalReferenceId]:
    svq_dataset = self._get_svq_dataset()
    for example in svq_dataset.get_task_data(
        sub_task, dtype={'locale': str, 'utt_id': str, 'page_title': str}
    ).to_dict('records'):
      if example['locale'] == self.locale:
        yield retrieval_evaluator.RetrievalReferenceId(
            sound_id=example['utt_id'], reference_id=example['page_title']
        )

  def get_documents_source(self) -> str:
    return 'wikipedia/20190301.en'

  @staticmethod
  def documents_generator(wikipedia_dataset: Any) -> Iterable[types.Text]:
    ds = tfds.load(wikipedia_dataset, split='train')
    for example in ds.as_numpy_iterator():
      title = example['title'].decode('utf-8')
      yield types.Text(
          text=example['text'].decode('utf-8'),
          context=types.TextContextParams(id=title, title=title),
      )


class SVQArEgDocumentCrossLangRetrieval(SVQDocumentCrossLangRetrieval):
  locale = 'ar_eg'
  metadata = types.TaskMetadata(
      name='SVQArEgDocumentCrossLangRetrieval',
      description='Document cross-lang retrieval task.',
      reference='TODO',
      type='DocumentCrossLangRetrieval',
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


class SVQArXGulfDocumentCrossLangRetrieval(SVQDocumentCrossLangRetrieval):
  locale = 'ar_x_gulf'
  metadata = types.TaskMetadata(
      name='SVQArXGulfDocumentCrossLangRetrieval',
      description='Document cross-lang retrieval task.',
      reference='TODO',
      type='DocumentCrossLangRetrieval',
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


class SVQArXLevantDocumentCrossLangRetrieval(SVQDocumentCrossLangRetrieval):
  locale = 'ar_x_levant'
  metadata = types.TaskMetadata(
      name='SVQArXLevantDocumentCrossLangRetrieval',
      description='Document cross-lang retrieval task.',
      reference='TODO',
      type='DocumentCrossLangRetrieval',
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


class SVQArXMaghrebiDocumentCrossLangRetrieval(SVQDocumentCrossLangRetrieval):
  locale = 'ar_x_maghrebi'
  metadata = types.TaskMetadata(
      name='SVQArXMaghrebiDocumentCrossLangRetrieval',
      description='Document cross-lang retrieval task.',
      reference='TODO',
      type='DocumentCrossLangRetrieval',
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


class SVQBnBdDocumentCrossLangRetrieval(SVQDocumentCrossLangRetrieval):
  locale = 'bn_bd'
  metadata = types.TaskMetadata(
      name='SVQBnBdDocumentCrossLangRetrieval',
      description='Document cross-lang retrieval task.',
      reference='TODO',
      type='DocumentCrossLangRetrieval',
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


class SVQBnInDocumentCrossLangRetrieval(SVQDocumentCrossLangRetrieval):
  locale = 'bn_in'
  metadata = types.TaskMetadata(
      name='SVQBnInDocumentCrossLangRetrieval',
      description='Document cross-lang retrieval task.',
      reference='TODO',
      type='DocumentCrossLangRetrieval',
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


class SVQFiFiDocumentCrossLangRetrieval(SVQDocumentCrossLangRetrieval):
  locale = 'fi_fi'
  metadata = types.TaskMetadata(
      name='SVQFiFiDocumentCrossLangRetrieval',
      description='Document cross-lang retrieval task.',
      reference='TODO',
      type='DocumentCrossLangRetrieval',
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


class SVQGuInDocumentCrossLangRetrieval(SVQDocumentCrossLangRetrieval):
  locale = 'gu_in'
  metadata = types.TaskMetadata(
      name='SVQGuInDocumentCrossLangRetrieval',
      description='Document cross-lang retrieval task.',
      reference='TODO',
      type='DocumentCrossLangRetrieval',
      category='speech',
      main_score='MRR',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
      eval_splits=['test'],
      eval_langs=['gu-IN'],
      domains=['speech'],
      task_subtypes=['retrieval'],
  )


class SVQHiInDocumentCrossLangRetrieval(SVQDocumentCrossLangRetrieval):
  locale = 'hi_in'
  metadata = types.TaskMetadata(
      name='SVQHiInDocumentCrossLangRetrieval',
      description='Document cross-lang retrieval task.',
      reference='TODO',
      type='DocumentCrossLangRetrieval',
      category='speech',
      main_score='MRR',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
      eval_splits=['test'],
      eval_langs=['hi-IN'],
      domains=['speech'],
      task_subtypes=['retrieval'],
  )


class SVQJaJpDocumentCrossLangRetrieval(SVQDocumentCrossLangRetrieval):
  locale = 'ja_jp'
  metadata = types.TaskMetadata(
      name='SVQJaJpDocumentCrossLangRetrieval',
      description='Document cross-lang retrieval task.',
      reference='TODO',
      type='DocumentCrossLangRetrieval',
      category='speech',
      main_score='MRR',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
      eval_splits=['test'],
      eval_langs=['ja-JP'],
      domains=['speech'],
      task_subtypes=['retrieval'],
  )


class SVQKnInDocumentCrossLangRetrieval(SVQDocumentCrossLangRetrieval):
  locale = 'kn_in'
  metadata = types.TaskMetadata(
      name='SVQKnInDocumentCrossLangRetrieval',
      description='Document cross-lang retrieval task.',
      reference='TODO',
      type='DocumentCrossLangRetrieval',
      category='speech',
      main_score='MRR',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
      eval_splits=['test'],
      eval_langs=['kn-IN'],
      domains=['speech'],
      task_subtypes=['retrieval'],
  )


class SVQKoKrDocumentCrossLangRetrieval(SVQDocumentCrossLangRetrieval):
  locale = 'ko_kr'
  metadata = types.TaskMetadata(
      name='SVQKoKrDocumentCrossLangRetrieval',
      description='Document cross-lang retrieval task.',
      reference='TODO',
      type='DocumentCrossLangRetrieval',
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


class SVQMlInDocumentCrossLangRetrieval(SVQDocumentCrossLangRetrieval):
  locale = 'ml_in'
  metadata = types.TaskMetadata(
      name='SVQMlInDocumentCrossLangRetrieval',
      description='Document cross-lang retrieval task.',
      reference='TODO',
      type='DocumentCrossLangRetrieval',
      category='speech',
      main_score='MRR',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
      eval_splits=['test'],
      eval_langs=['ml-IN'],
      domains=['speech'],
      task_subtypes=['retrieval'],
  )


class SVQMrInDocumentCrossLangRetrieval(SVQDocumentCrossLangRetrieval):
  locale = 'mr_in'
  metadata = types.TaskMetadata(
      name='SVQMrInDocumentCrossLangRetrieval',
      description='Document cross-lang retrieval task.',
      reference='TODO',
      type='DocumentCrossLangRetrieval',
      category='speech',
      main_score='MRR',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
      eval_splits=['test'],
      eval_langs=['mr-IN'],
      domains=['speech'],
      task_subtypes=['retrieval'],
  )


class SVQRuRuDocumentCrossLangRetrieval(SVQDocumentCrossLangRetrieval):
  locale = 'ru_ru'
  metadata = types.TaskMetadata(
      name='SVQRuRuDocumentCrossLangRetrieval',
      description='Document cross-lang retrieval task.',
      reference='TODO',
      type='DocumentCrossLangRetrieval',
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


class SVQTaInDocumentCrossLangRetrieval(SVQDocumentCrossLangRetrieval):
  locale = 'ta_in'
  metadata = types.TaskMetadata(
      name='SVQTaInDocumentCrossLangRetrieval',
      description='Document cross-lang retrieval task.',
      reference='TODO',
      type='DocumentCrossLangRetrieval',
      category='speech',
      main_score='MRR',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
      eval_splits=['test'],
      eval_langs=['ta-IN'],
      domains=['speech'],
      task_subtypes=['retrieval'],
  )


class SVQTeInDocumentCrossLangRetrieval(SVQDocumentCrossLangRetrieval):
  locale = 'te_in'
  metadata = types.TaskMetadata(
      name='SVQTeInDocumentCrossLangRetrieval',
      description='Document cross-lang retrieval task.',
      reference='TODO',
      type='DocumentCrossLangRetrieval',
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


class SVQUrInDocumentCrossLangRetrieval(SVQDocumentCrossLangRetrieval):
  locale = 'ur_in'
  metadata = types.TaskMetadata(
      name='SVQUrInDocumentCrossLangRetrieval',
      description='Document cross-lang retrieval task.',
      reference='TODO',
      type='DocumentCrossLangRetrieval',
      category='speech',
      main_score='MRR',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
      eval_splits=['test'],
      eval_langs=['ur-IN'],
      domains=['speech'],
      task_subtypes=['retrieval'],
  )


class SVQUrPkDocumentCrossLangRetrieval(SVQDocumentCrossLangRetrieval):
  locale = 'ur_pk'
  metadata = types.TaskMetadata(
      name='SVQUrPkDocumentCrossLangRetrieval',
      description='Document cross-lang retrieval task.',
      reference='TODO',
      type='DocumentCrossLangRetrieval',
      category='speech',
      main_score='MRR',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
      eval_splits=['test'],
      eval_langs=['ur-PK'],
      domains=['speech'],
      task_subtypes=['retrieval'],
  )
