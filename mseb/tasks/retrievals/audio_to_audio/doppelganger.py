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

"""Doppelganger cross-domain audio retrieval tasks."""

import os
from typing import Iterable

from mseb import types
from mseb.datasets import doppelganger
from mseb.evaluators import retrieval_evaluator
from mseb.tasks import retrieval


_GALLERY_SIZE = 3065
_SUB_TASKS = ['exact_source', 'category']


class _DoppelgangerRetrieval(retrieval.RetrievalTask):
  """Shared implementation for both synthetic-real retrieval directions."""

  query_domain: str
  document_domain: str

  def __init__(self):
    # Full-gallery rankings are required for category-level average precision.
    super().__init__(top_k=_GALLERY_SIZE)
    self._dataset = None

  def _get_dataset(self) -> doppelganger.DoppelgangerDataset:
    if self._dataset is None:
      self._dataset = doppelganger.DoppelgangerDataset(split='test')
    return self._dataset

  @property
  def index_dir(self) -> str:
    direction = f'{self.query_domain}_to_{self.document_domain}'
    return os.path.join(super().index_dir, f'doppelganger_{direction}')

  @property
  def sub_tasks(self) -> list[str]:
    return _SUB_TASKS

  def get_documents_source(
      self,
  ) -> tuple[doppelganger.DoppelgangerDataset, str]:
    return self._get_dataset(), self.document_domain

  @staticmethod
  def documents_generator(
      documents_source: tuple[doppelganger.DoppelgangerDataset, str],
  ) -> Iterable[types.Sound]:
    dataset, domain = documents_source
    for record in dataset.get_task_data().to_dict('records'):
      if domain == 'real':
        yield dataset.get_real_sound(record)
      else:
        yield dataset.get_synthetic_sound(record)

  def multimodal_inputs(self) -> Iterable[types.Sound]:
    dataset = self._get_dataset()
    for record in dataset.get_task_data().to_dict('records'):
      if self.query_domain == 'real':
        yield dataset.get_real_sound(record)
      else:
        yield dataset.get_synthetic_sound(record)

  def examples(
      self, sub_task: str
  ) -> Iterable[retrieval_evaluator.RetrievalReferenceId]:
    if sub_task not in _SUB_TASKS:
      raise ValueError(f'Unknown Doppelganger sub-task: {sub_task}.')

    dataset = self._get_dataset()
    records = dataset.get_task_data().to_dict('records')
    document_ids_by_event: dict[str, list[str]] = {}
    if sub_task == 'category':
      for record in records:
        document_ids_by_event.setdefault(record['event_id'], []).append(
            dataset.sound_id(record['pair_id'], self.document_domain)
        )

    for record in records:
      if sub_task == 'exact_source':
        reference_id: str | list[str] = dataset.sound_id(
            record['pair_id'], self.document_domain
        )
      else:
        reference_id = document_ids_by_event[record['event_id']]
      yield retrieval_evaluator.RetrievalReferenceId(
          sound_id=dataset.sound_id(record['pair_id'], self.query_domain),
          reference_id=reference_id,
      )


_DATASET_METADATA = types.Dataset(
    name='Doppelganger',
    path='https://huggingface.co/datasets/elliottash/doppelganger',
    revision='mseb-v1',
)

_SCORES = [
    retrieval_evaluator.map(),
    retrieval_evaluator.mrr(),
    retrieval_evaluator.em(),
]


class DoppelgangerSyntheticToRealRetrieval(_DoppelgangerRetrieval):
  """Retrieves real sources using their synthetic twins as queries."""

  query_domain = 'synthetic'
  document_domain = 'real'

  metadata = types.TaskMetadata(
      name='DoppelgangerSyntheticToRealRetrieval',
      description=(
          'Retrieve real sound effects from audio-conditioned synthetic '
          'queries, using exact-source and UCS-category relevance.'
      ),
      reference='https://arxiv.org/abs/2607.04337',
      type='AudioRetrieval',
      category='audio',
      main_score='MAP',
      revision='1.0.0',
      dataset=_DATASET_METADATA,
      scores=_SCORES,
      eval_splits=['test'],
      eval_langs=['und'],
      domains=['audio'],
      task_subtypes=['retrieval', 'cross-domain'],
      documentation_file='doppelganger_retrieval.md',
      dataset_documentation_file='dataset_doppelganger.md',
  )


class DoppelgangerRealToSyntheticRetrieval(_DoppelgangerRetrieval):
  """Retrieves synthetic twins using their real sources as queries."""

  query_domain = 'real'
  document_domain = 'synthetic'

  metadata = types.TaskMetadata(
      name='DoppelgangerRealToSyntheticRetrieval',
      description=(
          'Retrieve audio-conditioned synthetic twins from real sound '
          'queries, using exact-source and UCS-category relevance.'
      ),
      reference='https://arxiv.org/abs/2607.04337',
      type='AudioRetrieval',
      category='audio',
      main_score='MAP',
      revision='1.0.0',
      dataset=_DATASET_METADATA,
      scores=_SCORES,
      eval_splits=['test'],
      eval_langs=['und'],
      domains=['audio'],
      task_subtypes=['retrieval', 'cross-domain'],
      documentation_file='doppelganger_retrieval.md',
      dataset_documentation_file='dataset_doppelganger.md',
  )
