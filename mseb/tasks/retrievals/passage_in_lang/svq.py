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
from typing import Any, Iterable

from mseb import encoder as encoder_lib
from mseb import types
from mseb.datasets import simple_voice_questions as svq
from mseb.encoders import normalized_text_encoder_with_prompt as text_encoder_lib
from mseb.evaluators import retrieval_evaluator
from mseb.tasks import retrieval


class SVQRetrieval(retrieval.RetrievalTask):
  """SVQ retrieval."""

  metadata = types.TaskMetadata(
      name='SVQRetrieval',
      description='Retrieval task.',
      reference='TODO',
      type='Retrieval',
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


class SVQPassageInLangRetrieval(SVQRetrieval):
  """SVQ passage in-lang retrieval."""

  def __init__(
      self,
      cache_dir: str | None = None,
      text_encoder_cls: type[encoder_lib.TextEncoder] | None = None,
      text_encoder_kwargs: dict[str, Any] | None = None,
  ):
    super().__init__(
        cache_dir=cache_dir,
        text_encoder_cls=text_encoder_cls,
        text_encoder_kwargs=text_encoder_kwargs,
    )
    self.cache_dir = os.path.join(
        self.cache_dir, 'svq_passage_retrieval_in_lang'
    )

  @property
  def sub_tasks(self) -> list[str]:
    return ['passage_retrieval_in_lang']

  def documents(self) -> Iterable[types.Text]:
    svq_dataset = svq.SimpleVoiceQuestionsDataset(base_path=self.cache_dir)
    for example in svq_dataset.get_task_data(
        'passage_retrieval_in_lang_index'
    ).itertuples():
      yield types.Text(
          text=example.context,
          context=types.TextContextParams(
              id=example.id,
              title=example.title,
          ),
      )


class SVQEnUsPassageInLangRetrieval(SVQPassageInLangRetrieval):
  """SVQ passage in-lang retrieval for en-US."""

  def sounds(self) -> Iterable[types.Sound]:
    assert 'en-US' in self.metadata.eval_langs
    # TODO(heigold): Race condition or download all data for each locale?
    svq_dataset = svq.SimpleVoiceQuestionsDataset(base_path=self.cache_dir)
    for example in svq_dataset.get_task_data(
        'passage_retrieval_in_lang'
    ).itertuples():
      if example.locale == 'en_us':
        yield svq_dataset.get_sound_by_id(example.utt_id)

  def examples(
      self, sub_task: str
  ) -> Iterable[retrieval_evaluator.RetrievalReferenceId]:
    assert 'en-US' in self.metadata.eval_langs
    # TODO(heigold): Race condition or download all data for each locale?
    svq_dataset = svq.SimpleVoiceQuestionsDataset(base_path=self.cache_dir)
    for example in svq_dataset.get_task_data(sub_task).itertuples():
      if example.locale == 'en_us':
        yield retrieval_evaluator.RetrievalReferenceId(
            sound_id=example.utt_id, reference_id=example.passage_id
        )


class SVQEnUsPassageInLangRetrievalGecko(SVQEnUsPassageInLangRetrieval):

  def __init__(self, cache_dir: str | None = None):
    super().__init__(
        cache_dir=cache_dir,
        text_encoder_cls=text_encoder_lib.GeckoTextEncoder,
        text_encoder_kwargs={'model_path': '@gecko/gecko-1b-i18n-tpu/2'},
    )


