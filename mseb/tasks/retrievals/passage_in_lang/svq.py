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

  def __init__(
      self,
      locale: str,
      cache_dir: str | None = None,
      text_encoder_name: str | None = None,
  ):
    super().__init__(cache_dir=cache_dir, text_encoder_name=text_encoder_name)
    self.locale = locale

  @property
  def index_dir(self) -> str:
    return os.path.join(
        super().index_dir,
        'svq_passage_retrieval_in_lang',
        self.text_encoder_name,
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

  def sounds(self) -> Iterable[types.Sound]:
    svq_dataset = svq.SimpleVoiceQuestionsDataset(base_path=self.cache_dir)
    for example in svq_dataset.get_task_data(
        'passage_retrieval_in_lang'
    ).itertuples():
      if example.locale == self.locale:
        sound = svq_dataset.get_sound_by_id(example.utt_id)
        # Add the ground truth query for headroom analysis.
        sound.context.text = example.text
        yield sound

  def examples(
      self, sub_task: str
  ) -> Iterable[retrieval_evaluator.RetrievalReferenceId]:
    svq_dataset = svq.SimpleVoiceQuestionsDataset(base_path=self.cache_dir)
    for example in svq_dataset.get_task_data(sub_task).itertuples():
      if example.locale == self.locale:
        yield retrieval_evaluator.RetrievalReferenceId(
            sound_id=example.utt_id, reference_id=example.passage_id
        )


class SVQEnUsPassageInLangRetrievalGecko(SVQPassageInLangRetrieval):
  """SVQ passage in-lang retrieval for en-US using Gecko."""

  metadata = types.TaskMetadata(
      name='SVQEnUsPassageInLangRetrievalGecko',
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

  def __init__(self, cache_dir: str | None = None):
    super().__init__(
        locale='en_us', cache_dir=cache_dir, text_encoder_name='gecko_text'
    )


