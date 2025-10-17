# Copyright 2025 The MSEB Authors.
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

svq = pytest.importorskip("mseb.tasks.retrievals.passage_in_lang.svq")
FLAGS = flags.FLAGS


@pytest.mark.scann
@pytest.mark.optional
class SVQEnUsPassageInLangRetrievalTest(absltest.TestCase):

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

  def test_svq_passage_in_lang_retrieval_documents(self):
    task = svq.SVQEnUsPassageInLangRetrieval()
    self.assertEqual(task.sub_tasks, ["passage_retrieval_in_lang"])
    documents = list(task.documents())
    self.assertLen(documents, 3)
    document = documents[2]
    self.assertEqual(document.context.id, "english-1064747448949054415-7")
    self.assertEqual(document.context.title, "Little Albert experiment")
    self.assertTrue(document.text.startswith("Albert was about one year"))

  def test_svq_en_us_passage_in_lang_retrieval_sounds(self):
    task = svq.SVQEnUsPassageInLangRetrieval()
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

  def test_svq_en_us_passage_in_lang_retrieval_examples(self):
    task = svq.SVQEnUsPassageInLangRetrieval()
    examples = list(task.examples("passage_retrieval_in_lang"))
    self.assertLen(examples, 2)
    example = examples[0]
    self.assertEqual(example.sound_id, "utt_11697423627206642872")
    self.assertEqual(example.reference_id, "english-6037841464917965779-1")
    example = examples[1]
    self.assertEqual(example.sound_id, "utt_15041124811443622614")
    self.assertEqual(example.reference_id, "english-6037841464917965779-1")


if __name__ == "__main__":
  absltest.main()
