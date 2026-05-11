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

"""Flickr8k speech retrieval task.

Image-to-speech retrieval on the Flickr8k Audio dataset. Given an image from
Flickr8k, retrieve the matching spoken caption.

The task yields:
  - images(): Flickr8k images as Image objects (queries)
  - documents(): spoken captions as Sound objects (documents for index building)

Evaluation uses the standard RetrievalTask ScaNN-based pipeline: build an index
over audio embeddings, query with image embeddings, compute MRR/EM/Recall.
"""

import os
from typing import Any, Iterable

from mseb import types
from mseb.datasets import flickr8k
from mseb.evaluators import retrieval_evaluator
from mseb.tasks import retrieval


class Flickr8kSpeechRetrieval(retrieval.RetrievalTask):
  """Flickr8k image-to-speech retrieval task."""

  def __init__(self):
    super().__init__()
    self._dataset = None

  def _get_dataset(self) -> flickr8k.Flickr8kDataset:
    if self._dataset is None:
      self._dataset = flickr8k.Flickr8kDataset(split='test')
    return self._dataset

  @property
  def index_dir(self) -> str:
    return os.path.join(super().index_dir, 'flickr8k_speech_retrieval')

  @property
  def sub_tasks(self) -> list[str]:
    return ['speech_retrieval']

  def get_documents_source(self) -> flickr8k.Flickr8kDataset:
    return self._get_dataset()

  @staticmethod
  def documents_generator(
      dataset: Any,
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
      name='Flickr8kEnSpeechRetrieval',
      description=(
          'Image-to-speech retrieval on Flickr8k Audio: retrieve the correct'
          ' spoken English caption given a Flickr8k image.'
      ),
      reference='https://groups.csail.mit.edu/sls/downloads/flickraudio/',
      type='SpeechRetrieval',
      category='image',
      main_score='MRR',
      revision='1.0.0',
      dataset=types.Dataset(
          name='Flickr8kAudio',
          path='https://groups.csail.mit.edu/sls/downloads/flickraudio/',
          revision='1.0.0',
      ),
      scores=[retrieval_evaluator.mrr(), retrieval_evaluator.em()],
      eval_splits=['test'],
      eval_langs=['en'],
      domains=['image', 'speech'],
      task_subtypes=['retrieval'],
  )
