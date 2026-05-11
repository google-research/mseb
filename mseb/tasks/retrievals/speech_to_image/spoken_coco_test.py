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

"""Tests for the SpokenCOCO image retrieval task."""

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
    'mseb.tasks.retrievals.speech_to_image.spoken_coco'
)


@pytest.mark.scann
@pytest.mark.optional
def _create_mock_dataset(base_dir: str) -> None:
  """Creates a minimal SpokenCOCO file structure in base_dir."""
  # Create image directory.
  image_dir = os.path.join(base_dir, 'val2014')
  os.makedirs(image_dir)

  # Create two small RGB images.
  for filename in (
      'COCO_val2014_000000000001.jpg',
      'COCO_val2014_000000000002.jpg',
  ):
    img = PILImage.fromarray(np.zeros((16, 24, 3), dtype=np.uint8))
    img.save(os.path.join(image_dir, filename))

  # Create WAV directory.
  wav_dir = os.path.join(base_dir, 'wavs', 'val', '0')
  os.makedirs(wav_dir)

  # Create small WAV files.
  sample_rate = 16000
  duration_samples = sample_rate // 2
  for filename in (
      'speaker1-utt1_1_100.wav',
      'speaker2-utt2_1_200.wav',
      'speaker3-utt3_2_300.wav',
  ):
    waveform = np.zeros(duration_samples, dtype=np.float32)
    wavfile.write(os.path.join(wav_dir, filename), sample_rate, waveform)

  # Create the metadata JSON.
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
) -> spoken_coco_task.SpokenCocoImageRetrieval:
  """Creates a task instance backed by mock data."""
  task = spoken_coco_task.SpokenCocoImageRetrieval()
  task._dataset = spoken_coco.SpokenCocoDataset(
      base_path=testdata_dir, split='val'
  )
  return task


class SpokenCocoImageRetrievalTest(absltest.TestCase):
  """Tests for the SpokenCocoImageRetrieval task class."""

  def setUp(self):
    super().setUp()
    self.testdata_dir = self.create_tempdir().full_path
    _create_mock_dataset(self.testdata_dir)
    self.task = _make_task_with_mock_data(self.testdata_dir)

  def test_metadata(self):
    meta = self.task.metadata
    self.assertEqual(meta.name, 'SpokenCocoEnImageRetrieval')
    self.assertEqual(meta.type, 'ImageRetrieval')
    self.assertEqual(meta.category, 'speech')
    self.assertEqual(meta.main_score, 'MRR')
    self.assertIn('en', meta.eval_langs)
    self.assertIn('val', meta.eval_splits)

  def test_sub_tasks(self):
    self.assertEqual(self.task.sub_tasks, ['image_retrieval'])

  def test_multimodal_inputs(self):
    sounds = list(self.task.multimodal_inputs())
    self.assertLen(sounds, 3)

    # First caption of first image.
    s0 = sounds[0]
    self.assertIsInstance(s0, types.Sound)
    self.assertEqual(s0.context.id, 'speaker1-utt1_1_100')
    self.assertEqual(s0.context.language, 'en')
    self.assertEqual(s0.context.text, 'A DOG SITTING ON A COUCH')

    # Second caption of first image.
    s1 = sounds[1]
    self.assertEqual(s1.context.id, 'speaker2-utt2_1_200')
    self.assertEqual(s1.context.text, 'A BROWN DOG ON A RED SOFA')

    # Only caption of second image.
    s2 = sounds[2]
    self.assertEqual(s2.context.id, 'speaker3-utt3_2_300')
    self.assertEqual(s2.context.text, 'A CAT SLEEPING ON A BED')

  def test_documents(self):
    documents = list(self.task.documents())
    self.assertLen(documents, 2)

    # Both documents should be Image objects.
    for doc in documents:
      self.assertIsInstance(doc, types.Image)

    self.assertEqual(
        documents[0].context.id, 'val2014/COCO_val2014_000000000001.jpg'
    )
    self.assertEqual(
        documents[1].context.id, 'val2014/COCO_val2014_000000000002.jpg'
    )

  def test_examples(self):
    examples = list(self.task.examples('image_retrieval'))
    self.assertLen(examples, 3)

    # Each example maps a caption (sound_id) to its image (reference_id).
    e0 = examples[0]
    self.assertIsInstance(e0, retrieval_evaluator.RetrievalReferenceId)
    self.assertEqual(e0.sound_id, 'speaker1-utt1_1_100')
    self.assertEqual(e0.reference_id, 'val2014/COCO_val2014_000000000001.jpg')

    e1 = examples[1]
    self.assertEqual(e1.sound_id, 'speaker2-utt2_1_200')
    self.assertEqual(e1.reference_id, 'val2014/COCO_val2014_000000000001.jpg')

    e2 = examples[2]
    self.assertEqual(e2.sound_id, 'speaker3-utt3_2_300')
    self.assertEqual(e2.reference_id, 'val2014/COCO_val2014_000000000002.jpg')

  def test_multiple_captions_reference_same_image(self):
    """Verify that multiple captions from the same image share reference_id."""
    examples = list(self.task.examples('image_retrieval'))
    # First two examples should share the same image reference.
    self.assertEqual(examples[0].reference_id, examples[1].reference_id)
    # Third example should reference a different image.
    self.assertNotEqual(examples[0].reference_id, examples[2].reference_id)

  def test_sounds_and_examples_have_matching_ids(self):
    """sound IDs from multimodal_inputs() should match sound_ids in examples()."""
    sound_ids = {s.context.id for s in self.task.multimodal_inputs()}
    example_sound_ids = {
        e.sound_id for e in self.task.examples('image_retrieval')
    }
    self.assertEqual(sound_ids, example_sound_ids)

  def test_documents_and_examples_have_matching_ids(self):
    """document IDs from documents() should cover reference_ids in examples()."""
    doc_ids = {d.context.id for d in self.task.documents()}
    reference_ids = {
        e.reference_id for e in self.task.examples('image_retrieval')
    }
    self.assertTrue(
        reference_ids.issubset(doc_ids),
        f'Missing document IDs: {reference_ids - doc_ids}',
    )

  def test_task_is_registered(self):
    """Verify that the task name is set correctly in metadata."""
    self.assertIsNotNone(spoken_coco_task.SpokenCocoImageRetrieval.metadata)
    self.assertEqual(
        spoken_coco_task.SpokenCocoImageRetrieval.metadata.name,
        'SpokenCocoEnImageRetrieval',
    )


if __name__ == '__main__':
  absltest.main()
