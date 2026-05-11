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

"""Tests for the Flickr8k image retrieval task."""

import os

from absl.testing import absltest
from mseb import types
from mseb.datasets import flickr8k
import numpy as np
from PIL import Image as PILImage
import pytest
from scipy.io import wavfile

retrieval_evaluator = pytest.importorskip('mseb.evaluators.retrieval_evaluator')

flickr8k_task = pytest.importorskip(
    'mseb.tasks.retrievals.speech_to_image.flickr8k'
)


def _create_mock_dataset(base_dir: str) -> None:
  """Creates a minimal Flickr8k file structure in base_dir."""
  # Create images.
  image_dir = os.path.join(base_dir, 'Images')
  os.makedirs(image_dir)
  for filename in ('img_001.jpg', 'img_002.jpg'):
    img = PILImage.fromarray(np.zeros((16, 24, 3), dtype=np.uint8))
    img.save(os.path.join(image_dir, filename))

  # Create WAV files.
  wav_dir = os.path.join(base_dir, 'flickr_audio', 'wavs')
  os.makedirs(wav_dir)
  sample_rate = 16000
  for filename in ('img_001_0.wav', 'img_001_1.wav', 'img_002_0.wav'):
    waveform = np.zeros(sample_rate // 2, dtype=np.float32)
    wavfile.write(os.path.join(wav_dir, filename), sample_rate, waveform)

  # Create wav2capt.txt
  with open(os.path.join(base_dir, 'flickr_audio', 'wav2capt.txt'), 'w') as f:
    f.write('img_001_0.wav img_001.jpg #0\n')
    f.write('img_001_1.wav img_001.jpg #1\n')
    f.write('img_002_0.wav img_002.jpg #0\n')

  # Create wav2spk.txt
  with open(os.path.join(base_dir, 'flickr_audio', 'wav2spk.txt'), 'w') as f:
    f.write('img_001_0.wav speaker_A\n')
    f.write('img_001_1.wav speaker_B\n')
    f.write('img_002_0.wav speaker_A\n')

  # Create captions.txt
  with open(os.path.join(base_dir, 'captions.txt'), 'w') as f:
    f.write('image,caption\n')
    f.write('img_001.jpg,A dog sitting on a couch\n')
    f.write('img_001.jpg,A brown dog on a red sofa\n')
    f.write('img_002.jpg,A cat sleeping on a bed\n')

  # Create test split file.
  with open(os.path.join(base_dir, 'Flickr_8k.testImages.txt'), 'w') as f:
    f.write('img_001.jpg\n')
    f.write('img_002.jpg\n')


def _make_task_with_mock_data(
    testdata_dir: str,
) -> flickr8k_task.Flickr8kImageRetrieval:
  """Creates a task instance backed by mock data."""
  task = flickr8k_task.Flickr8kImageRetrieval()
  task._dataset = flickr8k.Flickr8kDataset(base_path=testdata_dir, split='test')
  return task


@pytest.mark.scann
@pytest.mark.optional
class Flickr8kImageRetrievalTest(absltest.TestCase):
  """Tests for the Flickr8kImageRetrieval task class."""

  def setUp(self):
    super().setUp()
    self.testdata_dir = self.create_tempdir().full_path
    _create_mock_dataset(self.testdata_dir)
    self.task = _make_task_with_mock_data(self.testdata_dir)

  def test_metadata(self):
    meta = self.task.metadata
    self.assertEqual(meta.name, 'Flickr8kEnImageRetrieval')
    self.assertEqual(meta.type, 'ImageRetrieval')
    self.assertEqual(meta.category, 'speech')
    self.assertEqual(meta.main_score, 'MRR')
    self.assertIn('en', meta.eval_langs)
    self.assertIn('test', meta.eval_splits)

  def test_sub_tasks(self):
    self.assertEqual(self.task.sub_tasks, ['image_retrieval'])

  def test_sounds(self):
    sounds = list(self.task.sounds())
    self.assertLen(sounds, 3)
    for s in sounds:
      self.assertIsInstance(s, types.Sound)
    sound_ids = {s.context.id for s in sounds}
    self.assertEqual(sound_ids, {'img_001_0', 'img_001_1', 'img_002_0'})

  def test_documents(self):
    documents = list(self.task.documents())
    self.assertLen(documents, 2)
    for doc in documents:
      self.assertIsInstance(doc, types.Image)
    doc_ids = {d.context.id for d in documents}
    self.assertEqual(doc_ids, {'Images/img_001.jpg', 'Images/img_002.jpg'})

  def test_examples(self):
    examples = list(self.task.examples('image_retrieval'))
    self.assertLen(examples, 3)
    for e in examples:
      self.assertIsInstance(e, retrieval_evaluator.RetrievalReferenceId)

  def test_multiple_captions_reference_same_image(self):
    """Verify that multiple captions from the same image share reference_id."""
    examples = list(self.task.examples('image_retrieval'))
    ref_by_sound = {e.sound_id: e.reference_id for e in examples}
    self.assertEqual(ref_by_sound['img_001_0'], ref_by_sound['img_001_1'])
    self.assertNotEqual(ref_by_sound['img_001_0'], ref_by_sound['img_002_0'])

  def test_sounds_and_examples_have_matching_ids(self):
    """sound IDs from sounds() should match sound_ids in examples()."""
    sound_ids = {s.context.id for s in self.task.sounds()}
    example_sound_ids = {
        e.sound_id for e in self.task.examples('image_retrieval')
    }
    self.assertEqual(sound_ids, example_sound_ids)

  def test_documents_and_examples_have_matching_ids(self):
    """document IDs should cover all reference_ids in examples()."""
    doc_ids = {d.context.id for d in self.task.documents()}
    reference_ids = {
        e.reference_id for e in self.task.examples('image_retrieval')
    }
    self.assertTrue(
        reference_ids.issubset(doc_ids),
        f'Missing document IDs: {reference_ids - doc_ids}',
    )


if __name__ == '__main__':
  absltest.main()
