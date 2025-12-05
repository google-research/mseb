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

"""SVQ query reranking tasks."""

import os
from typing import Iterable, Sequence

from mseb import task as task_lib
from mseb import types
from mseb.datasets import simple_voice_questions as svq
from mseb.evaluators import reranking_evaluator
from mseb.tasks import reranking


class SVQQueryReranking(reranking.RerankingTask):
  """SVQ query reranking."""

  locale: str | None = None

  def _get_dataset(self) -> svq.SimpleVoiceQuestionsDataset:
    return svq.SimpleVoiceQuestionsDataset()

  @property
  def embeddings_dir(self) -> str:
    assert self.locale is not None
    return os.path.join(
        super().embeddings_dir, f'svq_{self.locale}_query_reranking'
    )

  @property
  def sub_tasks(self) -> list[str]:
    return ['query_reranking']

  def sounds(self) -> Iterable[types.Sound]:
    svq_dataset = self._get_dataset()
    for example in svq_dataset.get_task_data(
        'query_reranking',
        dtype={
            'locale': str,
            'utt_id': str,
            task_lib.TRANSCRIPT_KEY.value: str,
        },
    ).to_dict('records'):
      if example['locale'] == self.locale:
        sound = svq_dataset.get_sound(example)
        sound.context.text = example[task_lib.TRANSCRIPT_KEY.value]
        yield sound

  def examples(
      self, sub_task: str
  ) -> Iterable[reranking_evaluator.RerankingCandidates]:
    svq_dataset = self._get_dataset()
    for example in svq_dataset.get_task_data(
        sub_task,
        dtype={'locale': str, 'utt_id': str, 'candidates': Sequence[str]},
    ).to_dict('records'):
      if example['locale'] == self.locale:
        yield reranking_evaluator.RerankingCandidates(
            sound_id=example['utt_id'],
            texts=example['candidates'],
            language=example['locale'],
        )

  def candidate_lists(self) -> Iterable[Sequence[types.Text]]:
    svq_dataset = self._get_dataset()
    for example in svq_dataset.get_task_data(
        'query_reranking', dtype={'locale': str, 'candidates': Sequence[str]}
    ).to_dict('records'):
      if example['locale'] == self.locale:
        yield [
            types.Text(
                text=candidate,
                context=types.TextContextParams(id=candidate),
            )
            for candidate in example['candidates']
        ]


class SVQArEgQueryReranking(SVQQueryReranking):
  locale = 'ar_eg'
  metadata = types.TaskMetadata(
      name='SVQArEgQueryReranking',
      description='Query reranking task.',
      reference='TODO',
      type='QueryReranking',
      category='speech',
      main_score='MAP',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          reranking_evaluator.map(),
          reranking_evaluator.mrr(),
          reranking_evaluator.wer(),
          reranking_evaluator.cer(),
      ],
      eval_splits=['test'],
      eval_langs=['ar-EG'],
      domains=['speech'],
      task_subtypes=['reranking'],
  )


class SVQArXGulfQueryReranking(SVQQueryReranking):
  locale = 'ar_x_gulf'
  metadata = types.TaskMetadata(
      name='SVQArXGulfQueryReranking',
      description='Query reranking task.',
      reference='TODO',
      type='QueryReranking',
      category='speech',
      main_score='MAP',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          reranking_evaluator.map(),
          reranking_evaluator.mrr(),
          reranking_evaluator.wer(),
          reranking_evaluator.cer(),
      ],
      eval_splits=['test'],
      eval_langs=['ar-x-gulf'],
      domains=['speech'],
      task_subtypes=['reranking'],
  )


class SVQArXLevantQueryReranking(SVQQueryReranking):
  locale = 'ar_x_levant'
  metadata = types.TaskMetadata(
      name='SVQArXLevantQueryReranking',
      description='Query reranking task.',
      reference='TODO',
      type='QueryReranking',
      category='speech',
      main_score='MAP',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          reranking_evaluator.map(),
          reranking_evaluator.mrr(),
          reranking_evaluator.wer(),
          reranking_evaluator.cer(),
      ],
      eval_splits=['test'],
      eval_langs=['ar-x-levant'],
      domains=['speech'],
      task_subtypes=['reranking'],
  )


class SVQArXMaghrebiQueryReranking(SVQQueryReranking):
  locale = 'ar_x_maghrebi'
  metadata = types.TaskMetadata(
      name='SVQArXMaghrebiQueryReranking',
      description='Query reranking task.',
      reference='TODO',
      type='QueryReranking',
      category='speech',
      main_score='MAP',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          reranking_evaluator.map(),
          reranking_evaluator.mrr(),
          reranking_evaluator.wer(),
          reranking_evaluator.cer(),
      ],
      eval_splits=['test'],
      eval_langs=['ar-x-maghrebi'],
      domains=['speech'],
      task_subtypes=['reranking'],
  )


class SVQBnBdQueryReranking(SVQQueryReranking):
  locale = 'bn_bd'
  metadata = types.TaskMetadata(
      name='SVQBnBdQueryReranking',
      description='Query reranking task.',
      reference='TODO',
      type='QueryReranking',
      category='speech',
      main_score='MAP',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          reranking_evaluator.map(),
          reranking_evaluator.mrr(),
          reranking_evaluator.wer(),
          reranking_evaluator.cer(),
      ],
      eval_splits=['test'],
      eval_langs=['bn-BD'],
      domains=['speech'],
      task_subtypes=['reranking'],
  )


class SVQBnInQueryReranking(SVQQueryReranking):
  locale = 'bn_in'
  metadata = types.TaskMetadata(
      name='SVQBnInQueryReranking',
      description='Query reranking task.',
      reference='TODO',
      type='QueryReranking',
      category='speech',
      main_score='MAP',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          reranking_evaluator.map(),
          reranking_evaluator.mrr(),
          reranking_evaluator.wer(),
          reranking_evaluator.cer(),
      ],
      eval_splits=['test'],
      eval_langs=['bn-IN'],
      domains=['speech'],
      task_subtypes=['reranking'],
  )


class SVQEnAuQueryReranking(SVQQueryReranking):
  locale = 'en_au'
  metadata = types.TaskMetadata(
      name='SVQEnAuQueryReranking',
      description='Query reranking task.',
      reference='TODO',
      type='QueryReranking',
      category='speech',
      main_score='MAP',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          reranking_evaluator.map(),
          reranking_evaluator.mrr(),
          reranking_evaluator.wer(),
          reranking_evaluator.cer(),
      ],
      eval_splits=['test'],
      eval_langs=['en-AU'],
      domains=['speech'],
      task_subtypes=['reranking'],
  )


class SVQEnGbQueryReranking(SVQQueryReranking):
  locale = 'en_gb'
  metadata = types.TaskMetadata(
      name='SVQEnGbQueryReranking',
      description='Query reranking task.',
      reference='TODO',
      type='QueryReranking',
      category='speech',
      main_score='MAP',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          reranking_evaluator.map(),
          reranking_evaluator.mrr(),
          reranking_evaluator.wer(),
          reranking_evaluator.cer(),
      ],
      eval_splits=['test'],
      eval_langs=['en-GB'],
      domains=['speech'],
      task_subtypes=['reranking'],
  )


class SVQEnInQueryReranking(SVQQueryReranking):
  locale = 'en_in'
  metadata = types.TaskMetadata(
      name='SVQEnInQueryReranking',
      description='Query reranking task.',
      reference='TODO',
      type='QueryReranking',
      category='speech',
      main_score='MAP',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          reranking_evaluator.map(),
          reranking_evaluator.mrr(),
          reranking_evaluator.wer(),
          reranking_evaluator.cer(),
      ],
      eval_splits=['test'],
      eval_langs=['en-IN'],
      domains=['speech'],
      task_subtypes=['reranking'],
  )


class SVQEnPhQueryReranking(SVQQueryReranking):
  locale = 'en_ph'
  metadata = types.TaskMetadata(
      name='SVQEnPhQueryReranking',
      description='Query reranking task.',
      reference='TODO',
      type='QueryReranking',
      category='speech',
      main_score='MAP',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          reranking_evaluator.map(),
          reranking_evaluator.mrr(),
          reranking_evaluator.wer(),
          reranking_evaluator.cer(),
      ],
      eval_splits=['test'],
      eval_langs=['en-PH'],
      domains=['speech'],
      task_subtypes=['reranking'],
  )


class SVQEnUsQueryReranking(SVQQueryReranking):
  locale = 'en_us'
  metadata = types.TaskMetadata(
      name='SVQEnUsQueryReranking',
      description='Query reranking task.',
      reference='TODO',
      type='QueryReranking',
      category='speech',
      main_score='MAP',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          reranking_evaluator.map(),
          reranking_evaluator.mrr(),
          reranking_evaluator.wer(),
          reranking_evaluator.cer(),
      ],
      eval_splits=['test'],
      eval_langs=['en-US'],
      domains=['speech'],
      task_subtypes=['reranking'],
  )


class SVQFiFiQueryReranking(SVQQueryReranking):
  locale = 'fi_fi'
  metadata = types.TaskMetadata(
      name='SVQFiFiQueryReranking',
      description='Query reranking task.',
      reference='TODO',
      type='QueryReranking',
      category='speech',
      main_score='MAP',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          reranking_evaluator.map(),
          reranking_evaluator.mrr(),
          reranking_evaluator.wer(),
          reranking_evaluator.cer(),
      ],
      eval_splits=['test'],
      eval_langs=['fi-FI'],
      domains=['speech'],
      task_subtypes=['reranking'],
  )


class SVQGuInQueryReranking(SVQQueryReranking):
  locale = 'gu_in'
  metadata = types.TaskMetadata(
      name='SVQGuInQueryReranking',
      description='Query reranking task.',
      reference='TODO',
      type='QueryReranking',
      category='speech',
      main_score='MAP',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          reranking_evaluator.map(),
          reranking_evaluator.mrr(),
          reranking_evaluator.wer(),
          reranking_evaluator.cer(),
      ],
      eval_splits=['test'],
      eval_langs=['gu-IN'],
      domains=['speech'],
      task_subtypes=['reranking'],
  )


class SVQHiInQueryReranking(SVQQueryReranking):
  locale = 'hi_in'
  metadata = types.TaskMetadata(
      name='SVQHiInQueryReranking',
      description='Query reranking task.',
      reference='TODO',
      type='QueryReranking',
      category='speech',
      main_score='MAP',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          reranking_evaluator.map(),
          reranking_evaluator.mrr(),
          reranking_evaluator.wer(),
          reranking_evaluator.cer(),
      ],
      eval_splits=['test'],
      eval_langs=['hi-IN'],
      domains=['speech'],
      task_subtypes=['reranking'],
  )


class SVQIdIdQueryReranking(SVQQueryReranking):
  locale = 'id_id'
  metadata = types.TaskMetadata(
      name='SVQIdIdQueryReranking',
      description='Query reranking task.',
      reference='TODO',
      type='QueryReranking',
      category='speech',
      main_score='MAP',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          reranking_evaluator.map(),
          reranking_evaluator.mrr(),
          reranking_evaluator.wer(),
          reranking_evaluator.cer(),
      ],
      eval_splits=['test'],
      eval_langs=['id-ID'],
      domains=['speech'],
      task_subtypes=['reranking'],
  )


class SVQJaJpQueryReranking(SVQQueryReranking):
  locale = 'ja_jp'
  metadata = types.TaskMetadata(
      name='SVQJaJpQueryReranking',
      description='Query reranking task.',
      reference='TODO',
      type='QueryReranking',
      category='speech',
      main_score='MAP',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          reranking_evaluator.map(),
          reranking_evaluator.mrr(),
          reranking_evaluator.wer(),
          reranking_evaluator.cer(),
      ],
      eval_splits=['test'],
      eval_langs=['ja-JP'],
      domains=['speech'],
      task_subtypes=['reranking'],
  )


class SVQKnInQueryReranking(SVQQueryReranking):
  locale = 'kn_in'
  metadata = types.TaskMetadata(
      name='SVQKnInQueryReranking',
      description='Query reranking task.',
      reference='TODO',
      type='QueryReranking',
      category='speech',
      main_score='MAP',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          reranking_evaluator.map(),
          reranking_evaluator.mrr(),
          reranking_evaluator.wer(),
          reranking_evaluator.cer(),
      ],
      eval_splits=['test'],
      eval_langs=['kn-IN'],
      domains=['speech'],
      task_subtypes=['reranking'],
  )


class SVQKoKrQueryReranking(SVQQueryReranking):
  locale = 'ko_kr'
  metadata = types.TaskMetadata(
      name='SVQKoKrQueryReranking',
      description='Query reranking task.',
      reference='TODO',
      type='QueryReranking',
      category='speech',
      main_score='MAP',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          reranking_evaluator.map(),
          reranking_evaluator.mrr(),
          reranking_evaluator.wer(),
          reranking_evaluator.cer(),
      ],
      eval_splits=['test'],
      eval_langs=['ko-KR'],
      domains=['speech'],
      task_subtypes=['reranking'],
  )


class SVQMlInQueryReranking(SVQQueryReranking):
  locale = 'ml_in'
  metadata = types.TaskMetadata(
      name='SVQMlInQueryReranking',
      description='Query reranking task.',
      reference='TODO',
      type='QueryReranking',
      category='speech',
      main_score='MAP',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          reranking_evaluator.map(),
          reranking_evaluator.mrr(),
          reranking_evaluator.wer(),
          reranking_evaluator.cer(),
      ],
      eval_splits=['test'],
      eval_langs=['ml-IN'],
      domains=['speech'],
      task_subtypes=['reranking'],
  )


class SVQMrInQueryReranking(SVQQueryReranking):
  locale = 'mr_in'
  metadata = types.TaskMetadata(
      name='SVQMrInQueryReranking',
      description='Query reranking task.',
      reference='TODO',
      type='QueryReranking',
      category='speech',
      main_score='MAP',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          reranking_evaluator.map(),
          reranking_evaluator.mrr(),
          reranking_evaluator.wer(),
          reranking_evaluator.cer(),
      ],
      eval_splits=['test'],
      eval_langs=['mr-IN'],
      domains=['speech'],
      task_subtypes=['reranking'],
  )


class SVQRuRuQueryReranking(SVQQueryReranking):
  locale = 'ru_ru'
  metadata = types.TaskMetadata(
      name='SVQRuRuQueryReranking',
      description='Query reranking task.',
      reference='TODO',
      type='QueryReranking',
      category='speech',
      main_score='MAP',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          reranking_evaluator.map(),
          reranking_evaluator.mrr(),
          reranking_evaluator.wer(),
          reranking_evaluator.cer(),
      ],
      eval_splits=['test'],
      eval_langs=['ru-RU'],
      domains=['speech'],
      task_subtypes=['reranking'],
  )


class SVQSwQueryReranking(SVQQueryReranking):
  locale = 'sw'
  metadata = types.TaskMetadata(
      name='SVQSwQueryReranking',
      description='Query reranking task.',
      reference='TODO',
      type='QueryReranking',
      category='speech',
      main_score='MAP',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          reranking_evaluator.map(),
          reranking_evaluator.mrr(),
          reranking_evaluator.wer(),
          reranking_evaluator.cer(),
      ],
      eval_splits=['test'],
      eval_langs=['sw'],
      domains=['speech'],
      task_subtypes=['reranking'],
  )


class SVQTaInQueryReranking(SVQQueryReranking):
  locale = 'ta_in'
  metadata = types.TaskMetadata(
      name='SVQTaInQueryReranking',
      description='Query reranking task.',
      reference='TODO',
      type='QueryReranking',
      category='speech',
      main_score='MAP',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          reranking_evaluator.map(),
          reranking_evaluator.mrr(),
          reranking_evaluator.wer(),
          reranking_evaluator.cer(),
      ],
      eval_splits=['test'],
      eval_langs=['ta-IN'],
      domains=['speech'],
      task_subtypes=['reranking'],
  )


class SVQTeInQueryReranking(SVQQueryReranking):
  locale = 'te_in'
  metadata = types.TaskMetadata(
      name='SVQTeInQueryReranking',
      description='Query reranking task.',
      reference='TODO',
      type='QueryReranking',
      category='speech',
      main_score='MAP',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          reranking_evaluator.map(),
          reranking_evaluator.mrr(),
          reranking_evaluator.wer(),
          reranking_evaluator.cer(),
      ],
      eval_splits=['test'],
      eval_langs=['te-IN'],
      domains=['speech'],
      task_subtypes=['reranking'],
  )


class SVQUrInQueryReranking(SVQQueryReranking):
  locale = 'ur_in'
  metadata = types.TaskMetadata(
      name='SVQUrInQueryReranking',
      description='Query reranking task.',
      reference='TODO',
      type='QueryReranking',
      category='speech',
      main_score='MAP',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          reranking_evaluator.map(),
          reranking_evaluator.mrr(),
          reranking_evaluator.wer(),
          reranking_evaluator.cer(),
      ],
      eval_splits=['test'],
      eval_langs=['ur-IN'],
      domains=['speech'],
      task_subtypes=['reranking'],
  )


class SVQUrPkQueryReranking(SVQQueryReranking):
  locale = 'ur_pk'
  metadata = types.TaskMetadata(
      name='SVQUrPkQueryReranking',
      description='Query reranking task.',
      reference='TODO',
      type='QueryReranking',
      category='speech',
      main_score='MAP',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          reranking_evaluator.map(),
          reranking_evaluator.mrr(),
          reranking_evaluator.wer(),
          reranking_evaluator.cer(),
      ],
      eval_splits=['test'],
      eval_langs=['ur-PK'],
      domains=['speech'],
      task_subtypes=['reranking'],
  )
