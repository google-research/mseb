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

"""SVQ passage cross-lang retrieval tasks."""

import os
from typing import Iterable

from mseb import task as task_lib
from mseb import types
from mseb.datasets import simple_voice_questions as svq
from mseb.evaluators import retrieval_evaluator
from mseb.tasks import retrieval


class SVQPassageCrossLangRetrieval(retrieval.RetrievalTask):
  """SVQ passage cross-lang retrieval."""

  locale: str | None = None

  def _get_dataset(self) -> svq.SimpleVoiceQuestionsDataset:
    return svq.SimpleVoiceQuestionsDataset()

  @property
  def index_dir(self) -> str:
    return os.path.join(super().index_dir, 'svq_passage_retrieval_cross_lang')

  @property
  def sub_tasks(self) -> list[str]:
    return ['passage_retrieval_cross_lang']

  def documents(self) -> Iterable[types.Text]:
    svq_dataset = self._get_dataset()
    for example in svq_dataset.get_task_data(
        'passage_retrieval_cross_lang_index',
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
        'passage_retrieval_cross_lang',
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
    svq_dataset = self._get_dataset()
    for example in svq_dataset.get_task_data(
        sub_task,
        dtype={'locale': str, 'utt_id': str, 'passage_id': str},
    ).to_dict('records'):
      if example['locale'] == self.locale:
        yield retrieval_evaluator.RetrievalReferenceId(
            sound_id=example['utt_id'], reference_id=example['passage_id']
        )


class SVQArEgPassageCrossLangRetrieval(SVQPassageCrossLangRetrieval):
  locale = 'ar_eg'
  metadata = types.TaskMetadata(
      name='SVQArEgPassageCrossLangRetrieval',
      description='Passage cross-lang retrieval task.',
      reference='TODO',
      type='PassageCrossLangRetrieval',
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


class SVQArXGulfPassageCrossLangRetrieval(SVQPassageCrossLangRetrieval):
  locale = 'ar_x_gulf'
  metadata = types.TaskMetadata(
      name='SVQArXGulfPassageCrossLangRetrieval',
      description='Passage cross-lang retrieval task.',
      reference='TODO',
      type='PassageCrossLangRetrieval',
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


class SVQArXLevantPassageCrossLangRetrieval(SVQPassageCrossLangRetrieval):
  locale = 'ar_x_levant'
  metadata = types.TaskMetadata(
      name='SVQArXLevantPassageCrossLangRetrieval',
      description='Passage cross-lang retrieval task.',
      reference='TODO',
      type='PassageCrossLangRetrieval',
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


class SVQArXMaghrebiPassageCrossLangRetrieval(SVQPassageCrossLangRetrieval):
  locale = 'ar_x_maghrebi'
  metadata = types.TaskMetadata(
      name='SVQArXMaghrebiPassageCrossLangRetrieval',
      description='Passage cross-lang retrieval task.',
      reference='TODO',
      type='PassageCrossLangRetrieval',
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


class SVQBnBdPassageCrossLangRetrieval(SVQPassageCrossLangRetrieval):
  locale = 'bn_bd'
  metadata = types.TaskMetadata(
      name='SVQBnBdPassageCrossLangRetrieval',
      description='Passage cross-lang retrieval task.',
      reference='TODO',
      type='PassageCrossLangRetrieval',
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


class SVQBnInPassageCrossLangRetrieval(SVQPassageCrossLangRetrieval):
  locale = 'bn_in'
  metadata = types.TaskMetadata(
      name='SVQBnInPassageCrossLangRetrieval',
      description='Passage cross-lang retrieval task.',
      reference='TODO',
      type='PassageCrossLangRetrieval',
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


class SVQFiFiPassageCrossLangRetrieval(SVQPassageCrossLangRetrieval):
  locale = 'fi_fi'
  metadata = types.TaskMetadata(
      name='SVQFiFiPassageCrossLangRetrieval',
      description='Passage cross-lang retrieval task.',
      reference='TODO',
      type='PassageCrossLangRetrieval',
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


class SVQGuInPassageCrossLangRetrieval(SVQPassageCrossLangRetrieval):
  locale = 'gu_in'
  metadata = types.TaskMetadata(
      name='SVQGuInPassageCrossLangRetrieval',
      description='Passage cross-lang retrieval task.',
      reference='TODO',
      type='PassageCrossLangRetrieval',
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


class SVQHiInPassageCrossLangRetrieval(SVQPassageCrossLangRetrieval):
  locale = 'hi_in'
  metadata = types.TaskMetadata(
      name='SVQHiInPassageCrossLangRetrieval',
      description='Passage cross-lang retrieval task.',
      reference='TODO',
      type='PassageCrossLangRetrieval',
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


class SVQJaJpPassageCrossLangRetrieval(SVQPassageCrossLangRetrieval):
  locale = 'ja_jp'
  metadata = types.TaskMetadata(
      name='SVQJaJpPassageCrossLangRetrieval',
      description='Passage cross-lang retrieval task.',
      reference='TODO',
      type='PassageCrossLangRetrieval',
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


class SVQKnInPassageCrossLangRetrieval(SVQPassageCrossLangRetrieval):
  locale = 'kn_in'
  metadata = types.TaskMetadata(
      name='SVQKnInPassageCrossLangRetrieval',
      description='Passage cross-lang retrieval task.',
      reference='TODO',
      type='PassageCrossLangRetrieval',
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


class SVQKoKrPassageCrossLangRetrieval(SVQPassageCrossLangRetrieval):
  locale = 'ko_kr'
  metadata = types.TaskMetadata(
      name='SVQKoKrPassageCrossLangRetrieval',
      description='Passage cross-lang retrieval task.',
      reference='TODO',
      type='PassageCrossLangRetrieval',
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


class SVQMlInPassageCrossLangRetrieval(SVQPassageCrossLangRetrieval):
  locale = 'ml_in'
  metadata = types.TaskMetadata(
      name='SVQMlInPassageCrossLangRetrieval',
      description='Passage cross-lang retrieval task.',
      reference='TODO',
      type='PassageCrossLangRetrieval',
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


class SVQMrInPassageCrossLangRetrieval(SVQPassageCrossLangRetrieval):
  locale = 'mr_in'
  metadata = types.TaskMetadata(
      name='SVQMrInPassageCrossLangRetrieval',
      description='Passage cross-lang retrieval task.',
      reference='TODO',
      type='PassageCrossLangRetrieval',
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


class SVQRuRuPassageCrossLangRetrieval(SVQPassageCrossLangRetrieval):
  locale = 'ru_ru'
  metadata = types.TaskMetadata(
      name='SVQRuRuPassageCrossLangRetrieval',
      description='Passage cross-lang retrieval task.',
      reference='TODO',
      type='PassageCrossLangRetrieval',
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


class SVQTaInPassageCrossLangRetrieval(SVQPassageCrossLangRetrieval):
  locale = 'ta_in'
  metadata = types.TaskMetadata(
      name='SVQTaInPassageCrossLangRetrieval',
      description='Passage cross-lang retrieval task.',
      reference='TODO',
      type='PassageCrossLangRetrieval',
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


class SVQTeInPassageCrossLangRetrieval(SVQPassageCrossLangRetrieval):
  locale = 'te_in'
  metadata = types.TaskMetadata(
      name='SVQTeInPassageCrossLangRetrieval',
      description='Passage cross-lang retrieval task.',
      reference='TODO',
      type='PassageCrossLangRetrieval',
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


class SVQUrInPassageCrossLangRetrieval(SVQPassageCrossLangRetrieval):
  locale = 'ur_in'
  metadata = types.TaskMetadata(
      name='SVQUrInPassageCrossLangRetrieval',
      description='Passage cross-lang retrieval task.',
      reference='TODO',
      type='PassageCrossLangRetrieval',
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


class SVQUrPkPassageCrossLangRetrieval(SVQPassageCrossLangRetrieval):
  locale = 'ur_pk'
  metadata = types.TaskMetadata(
      name='SVQUrPkPassageCrossLangRetrieval',
      description='Passage cross-lang retrieval task.',
      reference='TODO',
      type='PassageCrossLangRetrieval',
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
