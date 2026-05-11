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

"""SpokenCOCO image retrieval task.

Audio-to-image retrieval on the SpokenCOCO dataset. Given a spoken caption,
retrieve the matching image from the MS COCO val set.

The task yields:
  - sounds(): spoken captions as Sound objects (queries)
  - images(): COCO images as Image objects (documents for index building)

Evaluation uses the standard RetrievalTask ScaNN-based pipeline: build an index
over image embeddings, query with audio embeddings, compute MRR/EM/Recall.
"""

import os
from typing import Iterable

from mseb import types
from mseb.datasets import spoken_coco
from mseb.evaluators import retrieval_evaluator
from mseb.tasks import retrieval


class SpokenCocoImageRetrieval(retrieval.RetrievalTask):
  """SpokenCOCO audio-to-image retrieval task."""

  def __init__(self):
    super().__init__()
    self._dataset = None

  def _get_dataset(self) -> spoken_coco.SpokenCocoDataset:
    if self._dataset is None:
      self._dataset = spoken_coco.SpokenCocoDataset(split='val')
    return self._dataset

  @property
  def index_dir(self) -> str:
    return os.path.join(super().index_dir, 'spoken_coco_image_retrieval')

  @property
  def sub_tasks(self) -> list[str]:
    return ['image_retrieval']

  def get_documents_source(self) -> spoken_coco.SpokenCocoDataset:
    return self._get_dataset()

  @staticmethod
  def documents_generator(
      dataset: spoken_coco.SpokenCocoDataset,
  ) -> Iterable[types.Image]:
    """Yields Image objects for each unique image in the dataset."""
    for record in dataset.get_unique_images():
      yield dataset.get_image(record)

  def sounds(self) -> Iterable[types.Sound]:
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
      name='SpokenCocoEnImageRetrieval',
      description=(
          'Audio-to-image retrieval on SpokenCOCO: retrieve the correct'
          ' MS COCO image given a spoken English caption.'
      ),
      reference='https://groups.csail.mit.edu/sls/downloads/placesaudio/',
      type='ImageRetrieval',
      category='speech',
      main_score='MRR',
      revision='1.0.0',
      dataset=types.Dataset(
          name='SpokenCOCO',
          path=(
              'https://data.csail.mit.edu/placesaudio/SpokenCOCO.tar.gz'
              'audio_image/spoken_coco_en'
          ),
          revision='1.0.0',
      ),
      scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
      eval_splits=['val'],
      eval_langs=['en'],
      domains=['speech', 'image'],
      task_subtypes=['retrieval'],
  )
