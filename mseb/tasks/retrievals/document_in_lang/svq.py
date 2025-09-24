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

"""SVQ document in-lang retrieval tasks."""

import os
from typing import Iterable

from mseb import svq_data
from mseb import types
from mseb.datasets import simple_voice_questions as svq
from mseb.evaluators import retrieval_evaluator
from mseb.tasks import retrieval
import tensorflow_datasets as tfds


class SVQDocumentInLangRetrieval(retrieval.RetrievalTask):
  """SVQ document in-lang retrieval."""

  def __init__(
      self,
      locale: str,
      text_encoder_name: str | None = None,
      num_partitions: int = 1,
  ):
    """Initializes the SVQ document in-lang retrieval task.

    Args:
      locale: The locale of the task.
      text_encoder_name: The name of the text encoder to build the index.
      num_partitions: The number of index partitions to use.
    """
    super().__init__(
        text_encoder_name=text_encoder_name, num_partitions=num_partitions
    )
    self.locale = locale

  @property
  def index_dir(self) -> str:
    return os.path.join(
        super().index_dir,
        f'svq_{self.language}_document_retrieval_in_lang',
        self.text_encoder_name,
    )

  @property
  def language(self) -> str:
    return self.locale.split('_')[0]

  @property
  def sub_tasks(self) -> list[str]:
    return ['document_retrieval_in_lang']

  def sounds(self) -> Iterable[types.Sound]:
    svq_dataset = svq.SimpleVoiceQuestionsDataset(
        base_path=svq_data.SVQ_BASEPATH.value
    )
    for example in svq_dataset.get_task_data(
        'document_retrieval_in_lang'
    ).itertuples():
      if example.locale == self.locale:
        sound = svq_dataset.get_sound_by_id(example.utt_id)
        # Add the ground truth query for headroom analysis.
        sound.context.text = example.text
        yield sound

  def examples(
      self, sub_task: str
  ) -> Iterable[retrieval_evaluator.RetrievalReferenceId]:
    svq_dataset = svq.SimpleVoiceQuestionsDataset(
        base_path=svq_data.SVQ_BASEPATH.value
    )
    for example in svq_dataset.get_task_data(sub_task).itertuples():
      if example.locale == self.locale:
        yield retrieval_evaluator.RetrievalReferenceId(
            sound_id=example.utt_id, reference_id=example.page_title
        )

  def documents(self) -> Iterable[types.Text]:
    ds = tfds.load(
        f'wikipedia/20190301.{self.language}',
        split='train',
    )
    for example in ds.as_numpy_iterator():
      title = example['title'].decode('utf-8')
      yield types.Text(
          text=example['text'].decode('utf-8'),
          context=types.TextContextParams(id=title, title=title),
      )


class SVQEnUsDocumentInLangRetrievalGecko(SVQDocumentInLangRetrieval):
  """SVQ document in-lang retrieval for en-US using Gecko."""

  metadata = types.TaskMetadata(
      name='SVQEnUsDocumentInLangRetrievalGecko',
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

  def __init__(self):
    super().__init__(locale='en_us', text_encoder_name='gecko_text')


