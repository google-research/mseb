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

"""SpokenCOCO speech retrieval task.

Image-to-speech retrieval on the SpokenCOCO dataset. Given an image from
MS COCO, retrieve the matching spoken caption.

The task yields:
  - images(): COCO images as Image objects (queries)
  - documents(): spoken captions as Sound objects (documents for index building)

Evaluation uses the standard RetrievalTask ScaNN-based pipeline: build an index
over audio embeddings, query with image embeddings, compute MRR/EM/Recall.
"""

import os
from typing import Iterable

from mseb import types
from mseb.datasets import spoken_coco
from mseb.evaluators import retrieval_evaluator
from mseb.tasks import retrieval


class SpokenCocoSpeechRetrieval(retrieval.RetrievalTask):
  """SpokenCOCO image-to-speech retrieval task."""

  def __init__(self):
    super().__init__()
    self._dataset = None

  def _get_dataset(self) -> spoken_coco.SpokenCocoDataset:
    if self._dataset is None:
      self._dataset = spoken_coco.SpokenCocoDataset(split='val')
    return self._dataset

  @property
  def index_dir(self) -> str:
    return os.path.join(super().index_dir, 'spoken_coco_speech_retrieval')

  @property
  def sub_tasks(self) -> list[str]:
    return ['speech_retrieval']

  def get_documents_source(self) -> spoken_coco.SpokenCocoDataset:
    return self._get_dataset()

  @staticmethod
  def documents_generator(
      dataset: spoken_coco.SpokenCocoDataset,
  ) -> Iterable[types.Sound]:
    """Yields Sound objects for each spoken caption in the dataset."""
    for record in dataset.get_task_data().to_dict('records'):
      yield dataset.get_sound(record)

  def multimodal_inputs(self) -> Iterable[types.Image]:
    """Yields Image objects for each unique image in the dataset."""
    dataset = self._get_dataset()
    for record in dataset.get_unique_images():
      yield dataset.get_image(record)

  def examples(
      self, sub_task: str
  ) -> Iterable[retrieval_evaluator.RetrievalReferenceId]:
    """Yields (image_id, reference_id) pairs mapping images to captions.

    For image-to-speech retrieval, each image maps to multiple spoken captions.
    The reference_id is the caption uttid.

    Args:
      sub_task: The subtask to get examples for.

    Yields:
      A RetrievalReferenceId object for each image-caption pair.
    """
    dataset = self._get_dataset()
    utt_ids_by_img_id = {}
    for record in dataset.get_task_data().to_dict('records'):
      if record['image'] not in utt_ids_by_img_id:
        utt_ids_by_img_id[record['image']] = []
      utt_ids_by_img_id[record['image']].append(record['uttid'])
    for img_id, utt_ids in utt_ids_by_img_id.items():
      yield retrieval_evaluator.RetrievalReferenceId(
          sound_id=img_id, reference_id=utt_ids
      )

  metadata = types.TaskMetadata(
      name='SpokenCocoEnSpeechRetrieval',
      description=(
          'Image-to-speech retrieval on SpokenCOCO: retrieve the correct'
          ' spoken English caption given an MS COCO image.'
      ),
      reference='https://groups.csail.mit.edu/sls/downloads/placesaudio/',
      type='SpeechRetrieval',
      category='image',
      main_score='MRR',
      revision='1.0.0',
      dataset=types.Dataset(
          name='SpokenCOCO',
          path='https://data.csail.mit.edu/placesaudio/SpokenCOCO.tar.gz',
          revision='1.0.0',
      ),
      scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
      eval_splits=['val'],
      eval_langs=['en'],
      domains=['image', 'speech'],
      task_subtypes=['retrieval'],
  )
