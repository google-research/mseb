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

"""Tests for the Flickr8k speech retrieval task (image-to-speech)."""

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
    'mseb.tasks.retrievals.image_to_speech.flickr8k'
)


@pytest.mark.scann
@pytest.mark.optional
def _create_mock_dataset(base_dir: str) -> None:
  """Creates a minimal Flickr8k file structure in base_dir."""
  image_dir = os.path.join(base_dir, 'Images')
  os.makedirs(image_dir)
  for filename in ('img_001.jpg', 'img_002.jpg'):
    img = PILImage.fromarray(np.zeros((16, 24, 3), dtype=np.uint8))
    img.save(os.path.join(image_dir, filename))

  wav_dir = os.path.join(base_dir, 'flickr_audio', 'wavs')
  os.makedirs(wav_dir)
  sample_rate = 16000
  for filename in ('img_001_0.wav', 'img_001_1.wav', 'img_002_0.wav'):
    waveform = np.zeros(sample_rate // 2, dtype=np.float32)
    wavfile.write(os.path.join(wav_dir, filename), sample_rate, waveform)

  wav2capt_path = os.path.join(base_dir, 'flickr_audio', 'wav2capt.txt')
  with open(wav2capt_path, 'w') as f:
    f.write('img_001_0.wav img_001.jpg #0\n')
    f.write('img_001_1.wav img_001.jpg #1\n')
    f.write('img_002_0.wav img_002.jpg #0\n')

  wav2spk_path = os.path.join(base_dir, 'flickr_audio', 'wav2spk.txt')
  with open(wav2spk_path, 'w') as f:
    f.write('img_001_0.wav speaker_A\n')
    f.write('img_001_1.wav speaker_B\n')
    f.write('img_002_0.wav speaker_A\n')

  captions_path = os.path.join(base_dir, 'captions.txt')
  with open(captions_path, 'w') as f:
    f.write('image,caption\n')
    f.write('img_001.jpg,A dog sitting on a couch\n')
    f.write('img_001.jpg,A brown dog on a red sofa\n')
    f.write('img_002.jpg,A cat sleeping on a bed\n')

  split_path = os.path.join(base_dir, 'Flickr_8k.testImages.txt')
  with open(split_path, 'w') as f:
    f.write('img_001.jpg\n')
    f.write('img_002.jpg\n')


def _make_task_with_mock_data(
    testdata_dir: str,
) -> flickr8k_task.Flickr8kSpeechRetrieval:
  """Creates a task instance backed by mock data."""
  task = flickr8k_task.Flickr8kSpeechRetrieval()
  task._dataset = flickr8k.Flickr8kDataset(base_path=testdata_dir, split='test')
  return task


class Flickr8kSpeechRetrievalTest(absltest.TestCase):
  """Tests for the Flickr8kSpeechRetrieval task class."""

  def setUp(self):
    super().setUp()
    self.testdata_dir = self.create_tempdir().full_path
    _create_mock_dataset(self.testdata_dir)
    self.task = _make_task_with_mock_data(self.testdata_dir)

  def test_metadata(self):
    meta = self.task.metadata
    self.assertEqual(meta.name, 'Flickr8kEnSpeechRetrieval')
    self.assertEqual(meta.type, 'SpeechRetrieval')
    self.assertEqual(meta.category, 'image')
    self.assertEqual(meta.main_score, 'MRR')
    self.assertIn('en', meta.eval_langs)
    self.assertIn('test', meta.eval_splits)

  def test_sub_tasks(self):
    self.assertEqual(self.task.sub_tasks, ['speech_retrieval'])

  def test_multimodal_inputs(self):
    """multimodal_inputs() should yield one Image per unique image (queries)."""
    images = list(self.task.multimodal_inputs())
    self.assertLen(images, 2)
    for img in images:
      self.assertIsInstance(img, types.Image)
    image_ids = {img.context.id for img in images}
    self.assertEqual(image_ids, {'Images/img_001.jpg', 'Images/img_002.jpg'})

  def test_documents(self):
    """documents() should yield one Sound per spoken caption (index)."""
    documents = list(self.task.documents())
    self.assertLen(documents, 3)
    for doc in documents:
      self.assertIsInstance(doc, types.Sound)
    doc_ids = {d.context.id for d in documents}
    self.assertEqual(doc_ids, {'img_001_0', 'img_001_1', 'img_002_0'})

  def test_examples(self):
    """examples() maps image_id -> caption uttid."""
    examples = list(self.task.examples('speech_retrieval'))
    self.assertLen(examples, 2)
    for e in examples:
      self.assertIsInstance(e, retrieval_evaluator.RetrievalReferenceId)

    # Check that each example maps an image to a caption.
    example_pairs = {(e.sound_id, len(e.reference_id)) for e in examples}
    self.assertIn(('Images/img_001.jpg', 2), example_pairs)
    self.assertIn(('Images/img_002.jpg', 1), example_pairs)

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
