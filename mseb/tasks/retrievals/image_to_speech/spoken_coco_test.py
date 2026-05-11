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

"""Tests for the SpokenCOCO speech retrieval task (image-to-speech)."""

import json
import os

from absl.testing import absltest
from mseb import types
from mseb.datasets import spoken_coco
import numpy as np
from PIL import Image as PILImage
import pytest
from scipy.io import wavfile

retrieval_evaluator = pytest.importorskip('mseb.evaluators.retrieval_evaluator')

spoken_coco_task = pytest.importorskip(
    'mseb.tasks.retrievals.image_to_speech.spoken_coco'
)


def _create_mock_dataset(base_dir: str) -> None:
  """Creates a minimal SpokenCOCO file structure in base_dir."""
  image_dir = os.path.join(base_dir, 'val2014')
  os.makedirs(image_dir)
  for filename in (
      'COCO_val2014_000000000001.jpg',
      'COCO_val2014_000000000002.jpg',
  ):
    img = PILImage.fromarray(np.zeros((16, 24, 3), dtype=np.uint8))
    img.save(os.path.join(image_dir, filename))

  wav_dir = os.path.join(base_dir, 'wavs', 'val', '0')
  os.makedirs(wav_dir)
  sample_rate = 16000
  for filename in (
      'speaker1-utt1_1_100.wav',
      'speaker2-utt2_1_200.wav',
      'speaker3-utt3_2_300.wav',
  ):
    waveform = np.zeros(sample_rate // 2, dtype=np.float32)
    wavfile.write(os.path.join(wav_dir, filename), sample_rate, waveform)

  metadata = {
      'data': [
          {
              'image': 'val2014/COCO_val2014_000000000001.jpg',
              'captions': [
                  {
                      'text': 'A DOG SITTING ON A COUCH',
                      'speaker': 'speaker1',
                      'uttid': 'speaker1-utt1_1_100',
                      'wav': 'wavs/val/0/speaker1-utt1_1_100.wav',
                  },
                  {
                      'text': 'A BROWN DOG ON A RED SOFA',
                      'speaker': 'speaker2',
                      'uttid': 'speaker2-utt2_1_200',
                      'wav': 'wavs/val/0/speaker2-utt2_1_200.wav',
                  },
              ],
          },
          {
              'image': 'val2014/COCO_val2014_000000000002.jpg',
              'captions': [
                  {
                      'text': 'A CAT SLEEPING ON A BED',
                      'speaker': 'speaker3',
                      'uttid': 'speaker3-utt3_2_300',
                      'wav': 'wavs/val/0/speaker3-utt3_2_300.wav',
                  },
              ],
          },
      ]
  }
  with open(os.path.join(base_dir, 'SpokenCOCO_val.json'), 'w') as f:
    json.dump(metadata, f)


def _make_task_with_mock_data(
    testdata_dir: str,
) -> spoken_coco_task.SpokenCocoSpeechRetrieval:
  """Creates a task instance backed by mock data."""
  task = spoken_coco_task.SpokenCocoSpeechRetrieval()
  task._dataset = spoken_coco.SpokenCocoDataset(
      base_path=testdata_dir, split='val'
  )
  return task


@pytest.mark.scann
@pytest.mark.optional
class SpokenCocoSpeechRetrievalTest(absltest.TestCase):
  """Tests for the SpokenCocoSpeechRetrieval task class."""

  def setUp(self):
    super().setUp()
    self.testdata_dir = self.create_tempdir().full_path
    _create_mock_dataset(self.testdata_dir)
    self.task = _make_task_with_mock_data(self.testdata_dir)

  def test_metadata(self):
    meta = self.task.metadata
    self.assertEqual(meta.name, 'SpokenCocoEnSpeechRetrieval')
    self.assertEqual(meta.type, 'SpeechRetrieval')
    self.assertEqual(meta.category, 'image')
    self.assertEqual(meta.main_score, 'MRR')
    self.assertIn('en', meta.eval_langs)
    self.assertIn('val', meta.eval_splits)

  def test_sub_tasks(self):
    self.assertEqual(self.task.sub_tasks, ['speech_retrieval'])

  def test_multimodal_inputs(self):
    """multimodal_inputs() should yield one Image per unique image (queries)."""
    images = list(self.task.multimodal_inputs())
    self.assertLen(images, 2)
    for img in images:
      self.assertIsInstance(img, types.Image)
    self.assertEqual(
        images[0].context.id, 'val2014/COCO_val2014_000000000001.jpg'
    )
    self.assertEqual(
        images[1].context.id, 'val2014/COCO_val2014_000000000002.jpg'
    )

  def test_documents(self):
    """documents() should yield one Sound per spoken caption (index)."""
    documents = list(self.task.documents())
    self.assertLen(documents, 3)
    for doc in documents:
      self.assertIsInstance(doc, types.Sound)
    self.assertEqual(documents[0].context.id, 'speaker1-utt1_1_100')
    self.assertEqual(documents[1].context.id, 'speaker2-utt2_1_200')
    self.assertEqual(documents[2].context.id, 'speaker3-utt3_2_300')

  def test_examples(self):
    """examples() maps image_id -> caption uttid."""
    examples = list(self.task.examples('speech_retrieval'))
    self.assertLen(examples, 2)

    e0 = examples[0]
    self.assertIsInstance(e0, retrieval_evaluator.RetrievalReferenceId)
    # sound_id is the image (query), reference_id is the caption (document).
    self.assertEqual(e0.sound_id, 'val2014/COCO_val2014_000000000001.jpg')
    self.assertFalse(
        set(e0.reference_id) - {'speaker1-utt1_1_100', 'speaker2-utt2_1_200'}
    )

    e1 = examples[1]
    self.assertEqual(e1.sound_id, 'val2014/COCO_val2014_000000000002.jpg')
    self.assertFalse(set(e1.reference_id) - {'speaker3-utt3_2_300'})

  def test_images_and_examples_have_matching_ids(self):
    """image IDs from multimodal_inputs() should cover sound_ids in examples()."""
    image_ids = {img.context.id for img in self.task.multimodal_inputs()}
    example_image_ids = {
        e.sound_id for e in self.task.examples('speech_retrieval')
    }
    self.assertEqual(image_ids, example_image_ids)

  def test_documents_and_examples_have_matching_ids(self):
    """document IDs should cover reference_ids in examples()."""
    doc_ids = {d.context.id for d in self.task.documents()}
    reference_ids = [
        e.reference_id for e in self.task.examples('speech_retrieval')
    ]
    for reference_id in reference_ids:
      reference_id = set(reference_id)
      self.assertTrue(
          reference_id.issubset(doc_ids),
          f'Missing document IDs: {reference_id - doc_ids}',
      )


if __name__ == '__main__':
  absltest.main()
