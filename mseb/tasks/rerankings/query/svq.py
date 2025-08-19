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

from mseb import types
from mseb.datasets import simple_voice_questions as svq
from mseb.evaluators import reranking_evaluator
from mseb.tasks import reranking


class SVQQueryReranking(reranking.RerankingTask):
  """SVQ query reranking."""

  def __init__(
      self,
      cache_dir: str | None = None,
      text_encoder_name: str | None = None,
  ):
    super().__init__(cache_dir=cache_dir, text_encoder_name=text_encoder_name)
    self.cache_dir = os.path.join(self.cache_dir, 'svq_query_reranking')

  @property
  def sub_tasks(self) -> list[str]:
    return ['query_reranking']


class SVQEnUsQueryReranking(SVQQueryReranking):
  """SVQ query reranking for en-US."""

  def sounds(self) -> Iterable[types.Sound]:
    svq_dataset = svq.SimpleVoiceQuestionsDataset(base_path=self.cache_dir)
    for example in svq_dataset.get_task_data('query_reranking').itertuples():
      if example.locale == 'en_us':
        sound = svq_dataset.get_sound_by_id(example.utt_id)
        # Add the ground truth query for headroom analysis.
        sound.context.text = example.text
        yield sound

  def examples(
      self, sub_task: str
  ) -> Iterable[reranking_evaluator.RerankingCandidates]:
    svq_dataset = svq.SimpleVoiceQuestionsDataset(base_path=self.cache_dir)
    for example in svq_dataset.get_task_data(sub_task).itertuples():
      if example.locale == 'en_us':
        yield reranking_evaluator.RerankingCandidates(
            sound_id=example.utt_id, texts=example.candidates
        )

  def candidate_lists(self) -> Iterable[Sequence[types.Text]]:
    svq_dataset = svq.SimpleVoiceQuestionsDataset(base_path=self.cache_dir)
    for example in svq_dataset.get_task_data('query_reranking').itertuples():
      if example.locale == 'en_us':
        yield [
            types.Text(
                text=candidate,
                context=types.TextContextParams(id=candidate),
            )
            for candidate in example.candidates
        ]


class SVQEnUsQueryRerankingGecko(SVQEnUsQueryReranking):
  """SVQ query reranking for en-US using Gecko."""

  metadata = types.TaskMetadata(
      name='SVQEnUsQueryRerankingGecko',
      description='Query reranking task.',
      reference='TODO',
      type='QueryReranking',
      category='speech',
      main_score='WER',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          reranking_evaluator.wer(),
          reranking_evaluator.cer(),
          reranking_evaluator.mrr(),
      ],
      eval_splits=['test'],
      eval_langs=['en-US'],
      domains=['speech'],
      task_subtypes=['reranking'],
  )

  def __init__(self, cache_dir: str | None = None):
    super().__init__(cache_dir=cache_dir, text_encoder_name='gecko_text')


