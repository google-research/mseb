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

"""SVQ passage in-lang retrieval tasks."""

import os
from typing import Iterable

from mseb import types
from mseb.datasets import simple_voice_questions as svq
from mseb.evaluators import retrieval_evaluator
from mseb.tasks import retrieval


class SVQPassageInLangRetrieval(retrieval.RetrievalTask):
  """SVQ passage in-lang retrieval."""

  locale: str | None = None

  def _get_dataset(self) -> svq.SimpleVoiceQuestionsDataset:
    return svq.SimpleVoiceQuestionsDataset()

  @property
  def index_dir(self) -> str:
    return os.path.join(super().index_dir, 'svq_passage_retrieval_in_lang')

  @property
  def sub_tasks(self) -> list[str]:
    return ['passage_retrieval_in_lang']

  def documents(self) -> Iterable[types.Text]:
    svq_dataset = self._get_dataset()
    for example in svq_dataset.get_task_data(
        'passage_retrieval_in_lang_index',
        dtype={'id': str, 'title': str, 'context': str},
    ).to_dict('records'):
      yield types.Text(
          text=example['context'],
          context=types.TextContextParams(
              id=example['id'],
              title=example['title'],
          ),
      )

  def sounds(self) -> Iterable[types.Sound]:
    svq_dataset = self._get_dataset()
    for example in svq_dataset.get_task_data(
        'passage_retrieval_in_lang',
        dtype={'locale': str, 'utt_id': str, 'text': str},
    ).to_dict('records'):
      if example['locale'] == self.locale:
        sound = svq_dataset.get_sound({'utt_id': example['utt_id']})
        # Add the ground truth query for headroom analysis.
        sound.context.text = example['text']
        yield sound

  def examples(
      self, sub_task: str
  ) -> Iterable[retrieval_evaluator.RetrievalReferenceId]:
    svq_dataset = self._get_dataset()
    for example in svq_dataset.get_task_data(
        sub_task, dtype={'locale': str, 'utt_id': str, 'passage_id': str}
    ).to_dict('records'):
      if example['locale'] == self.locale:
        yield retrieval_evaluator.RetrievalReferenceId(
            sound_id=example['utt_id'], reference_id=example['passage_id']
        )


class SVQArEgPassageInLangRetrieval(SVQPassageInLangRetrieval):
  locale = 'ar_eg'
  metadata = types.TaskMetadata(
      name='SVQArEgPassageInLangRetrieval',
      description='Passage in-lang retrieval task.',
      reference='TODO',
      type='PassageInLangRetrieval',
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


class SVQArXGulfPassageInLangRetrieval(SVQPassageInLangRetrieval):
  locale = 'ar_x_gulf'
  metadata = types.TaskMetadata(
      name='SVQArXGulfPassageInLangRetrieval',
      description='Passage in-lang retrieval task.',
      reference='TODO',
      type='PassageInLangRetrieval',
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


class SVQArXLevantPassageInLangRetrieval(SVQPassageInLangRetrieval):
  locale = 'ar_x_levant'
  metadata = types.TaskMetadata(
      name='SVQArXLevantPassageInLangRetrieval',
      description='Passage in-lang retrieval task.',
      reference='TODO',
      type='PassageInLangRetrieval',
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


class SVQArXMaghrebiPassageInLangRetrieval(SVQPassageInLangRetrieval):
  locale = 'ar_x_maghrebi'
  metadata = types.TaskMetadata(
      name='SVQArXMaghrebiPassageInLangRetrieval',
      description='Passage in-lang retrieval task.',
      reference='TODO',
      type='PassageInLangRetrieval',
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


class SVQBnBdPassageInLangRetrieval(SVQPassageInLangRetrieval):
  locale = 'bn_bd'
  metadata = types.TaskMetadata(
      name='SVQBnBdPassageInLangRetrieval',
      description='Passage in-lang retrieval task.',
      reference='TODO',
      type='PassageInLangRetrieval',
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


class SVQBnInPassageInLangRetrieval(SVQPassageInLangRetrieval):
  locale = 'bn_in'
  metadata = types.TaskMetadata(
      name='SVQBnInPassageInLangRetrieval',
      description='Passage in-lang retrieval task.',
      reference='TODO',
      type='PassageInLangRetrieval',
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


class SVQEnAuPassageInLangRetrieval(SVQPassageInLangRetrieval):
  locale = 'en_au'
  metadata = types.TaskMetadata(
      name='SVQEnAuPassageInLangRetrieval',
      description='Passage in-lang retrieval task.',
      reference='TODO',
      type='PassageInLangRetrieval',
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


class SVQEnGbPassageInLangRetrieval(SVQPassageInLangRetrieval):
  locale = 'en_gb'
  metadata = types.TaskMetadata(
      name='SVQEnGbPassageInLangRetrieval',
      description='Passage in-lang retrieval task.',
      reference='TODO',
      type='PassageInLangRetrieval',
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


class SVQEnInPassageInLangRetrieval(SVQPassageInLangRetrieval):
  locale = 'en_in'
  metadata = types.TaskMetadata(
      name='SVQEnInPassageInLangRetrieval',
      description='Passage in-lang retrieval task.',
      reference='TODO',
      type='PassageInLangRetrieval',
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


class SVQEnPhPassageInLangRetrieval(SVQPassageInLangRetrieval):
  locale = 'en_ph'
  metadata = types.TaskMetadata(
      name='SVQEnPhPassageInLangRetrieval',
      description='Passage in-lang retrieval task.',
      reference='TODO',
      type='PassageInLangRetrieval',
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


class SVQEnUsPassageInLangRetrieval(SVQPassageInLangRetrieval):
  locale = 'en_us'
  metadata = types.TaskMetadata(
      name='SVQEnUsPassageInLangRetrieval',
      description='Passage in-lang retrieval task.',
      reference='TODO',
      type='PassageInLangRetrieval',
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


class SVQFiFiPassageInLangRetrieval(SVQPassageInLangRetrieval):
  locale = 'fi_fi'
  metadata = types.TaskMetadata(
      name='SVQFiFiPassageInLangRetrieval',
      description='Passage in-lang retrieval task.',
      reference='TODO',
      type='PassageInLangRetrieval',
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


class SVQIdIdPassageInLangRetrieval(SVQPassageInLangRetrieval):
  locale = 'id_id'
  metadata = types.TaskMetadata(
      name='SVQIdIdPassageInLangRetrieval',
      description='Passage in-lang retrieval task.',
      reference='TODO',
      type='PassageInLangRetrieval',
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


class SVQKoKrPassageInLangRetrieval(SVQPassageInLangRetrieval):
  locale = 'ko_kr'
  metadata = types.TaskMetadata(
      name='SVQKoKrPassageInLangRetrieval',
      description='Passage in-lang retrieval task.',
      reference='TODO',
      type='PassageInLangRetrieval',
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


class SVQRuRuPassageInLangRetrieval(SVQPassageInLangRetrieval):
  locale = 'ru_ru'
  metadata = types.TaskMetadata(
      name='SVQRuRuPassageInLangRetrieval',
      description='Passage in-lang retrieval task.',
      reference='TODO',
      type='PassageInLangRetrieval',
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


class SVQSwPassageInLangRetrieval(SVQPassageInLangRetrieval):
  locale = 'sw'
  metadata = types.TaskMetadata(
      name='SVQSwPassageInLangRetrieval',
      description='Passage in-lang retrieval task.',
      reference='TODO',
      type='PassageInLangRetrieval',
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


class SVQTeInPassageInLangRetrieval(SVQPassageInLangRetrieval):
  locale = 'te_in'
  metadata = types.TaskMetadata(
      name='SVQTeInPassageInLangRetrieval',
      description='Passage in-lang retrieval task.',
      reference='TODO',
      type='PassageInLangRetrieval',
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
