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

"""Flickr8k image retrieval task.

Audio-to-image retrieval on the Flickr8k Audio dataset. Given a spoken caption,
retrieve the matching image from the Flickr8k test set.

The task yields:
  - multimodal_inputs(): spoken captions as Sound objects (queries)
  - documents(): Flickr8k images as Image objects (documents for index building)

Evaluation uses the standard RetrievalTask ScaNN-based pipeline: build an index
over image embeddings, query with audio embeddings, compute MRR/EM/Recall.
"""

import os
from typing import Any, Iterable

from mseb import types
from mseb.datasets import flickr8k
from mseb.evaluators import retrieval_evaluator
from mseb.tasks import retrieval


class Flickr8kImageRetrieval(retrieval.RetrievalTask):
  """Flickr8k audio-to-image retrieval task."""

  def __init__(self):
    super().__init__()
    self._dataset = None

  def _get_dataset(self) -> flickr8k.Flickr8kDataset:
    if self._dataset is None:
      self._dataset = flickr8k.Flickr8kDataset(split='test')
    return self._dataset

  @property
  def index_dir(self) -> str:
    return os.path.join(super().index_dir, 'flickr8k_image_retrieval')

  @property
  def sub_tasks(self) -> list[str]:
    return ['image_retrieval']

  def get_documents_source(self) -> flickr8k.Flickr8kDataset:
    return self._get_dataset()

  @staticmethod
  def documents_generator(
      dataset: Any,
  ) -> Iterable[types.Image]:
    """Yields Image objects for each unique image in the dataset."""
    for record in dataset.get_unique_images():
      yield dataset.get_image(record)

  def multimodal_inputs(self) -> Iterable[types.Sound]:
    """Yields Sound objects for each spoken caption in the dataset."""
    dataset = self._get_dataset()
    for record in dataset.get_task_data().to_dict('records'):
      yield dataset.get_sound(record)

  def examples(
      self, sub_task: str
  ) -> Iterable[retrieval_evaluator.RetrievalReferenceId]:
    """Yields (sound_id, reference_id) pairs mapping captions to images."""
    dataset = self._get_dataset()
    for record in dataset.get_task_data().to_dict('records'):
      yield retrieval_evaluator.RetrievalReferenceId(
          sound_id=record['uttid'],
          reference_id=record['image'],
      )

  metadata = types.TaskMetadata(
      name='Flickr8kEnImageRetrieval',
      description=(
          'Audio-to-image retrieval on Flickr8k Audio: retrieve the correct'
          ' Flickr8k image given a spoken English caption.'
      ),
      reference='https://groups.csail.mit.edu/sls/downloads/flickraudio/',
      type='ImageRetrieval',
      category='speech',
      main_score='MRR',
      revision='1.0.0',
      dataset=types.Dataset(
          name='Flickr8kAudio',
          path=(
              '/cns/mb-d/home/gem-embed-cns-owner/audio/real_recording/'
              'audio_image/flickr8k_en'
          ),
          revision='1.0.0',
      ),
      scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
      eval_splits=['test'],
      eval_langs=['en'],
      domains=['speech', 'image'],
      task_subtypes=['retrieval'],
  )
