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

import os
import pathlib
import shutil

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from mseb import dataset
import pytest
import tensorflow_datasets as tfds

svq = pytest.importorskip("mseb.tasks.retrievals.document_in_lang.svq")
FLAGS = flags.FLAGS


@pytest.mark.scann
@pytest.mark.optional
class SVQEnUsDocumentInLangRetrievalTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent.parent.parent,
        "testdata",
    )
    # Add a .git marker to prevent SimpleVoiceQuestionsDataset from trying to
    # download the data from Huggingface.
    cache_dir = self.create_tempdir().full_path
    shutil.rmtree(cache_dir)
    shutil.copytree(testdata_path, cache_dir)
    os.chmod(cache_dir, 0o755)
    pathlib.Path.touch(pathlib.Path(os.path.join(cache_dir, ".git")))
    self.enter_context(
        flagsaver.flagsaver((dataset._DATASET_BASEPATH, cache_dir))
    )

  def test_svq_document_in_lang_retrieval_documents(self):
    with tfds.testing.mock_data(num_examples=1):
      task = svq.SVQEnUsDocumentInLangRetrieval()
      self.assertEqual(task.sub_tasks[0], "document_retrieval_in_lang")
      for document in task.documents():
        self.assertEqual(document.context.id, "chg dif hhia i e ce")
        self.assertEqual(document.context.id, document.context.title)
        self.assertEqual(document.text, "gebc   ahgjefjhfef")

  def test_svq_en_us_document_in_lang_retrieval_sounds(self):
    task = svq.SVQEnUsDocumentInLangRetrieval()
    sounds = list(task.sounds())
    self.assertLen(sounds, 2)
    sound = sounds[0]
    self.assertEqual(sound.context.id, "utt_11697423627206642872")
    self.assertEqual(sound.context.speaker_id, "speaker_5452472707103026757")
    self.assertEqual(sound.context.speaker_age, 27)
    self.assertEqual(sound.context.language, "en_us")
    sound = sounds[1]
    self.assertEqual(sound.context.id, "utt_15041124811443622614")
    self.assertEqual(sound.context.speaker_id, "speaker_10322347911861405809")
    self.assertEqual(sound.context.speaker_age, 25)
    self.assertEqual(sound.context.language, "en_us")

  def test_svq_en_us_document_in_lang_retrieval_examples(self):
    task = svq.SVQEnUsDocumentInLangRetrieval()
    examples = list(task.examples("document_retrieval_in_lang"))
    self.assertLen(examples, 2)
    example = examples[0]
    self.assertEqual(example.sound_id, "utt_11697423627206642872")
    self.assertEqual(example.reference_id, "Red-short carbon steel")
    example = examples[1]
    self.assertEqual(example.sound_id, "utt_15041124811443622614")
    self.assertEqual(example.reference_id, "Red-short carbon steel")


if __name__ == "__main__":
  absltest.main()
