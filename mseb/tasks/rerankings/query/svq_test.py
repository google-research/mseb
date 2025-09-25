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

from mseb.datasets import simple_voice_questions
from mseb.tasks.rerankings.query import svq

FLAGS = flags.FLAGS


class SVQEnUsQueryRerankingTest(absltest.TestCase):

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
        flagsaver.flagsaver(
            (simple_voice_questions.SVQ_BASEPATH, cache_dir)
        )
    )

  def test_svq_query_reranking_candidate_lists(self):
    task = svq.SVQEnUsQueryReranking()
    self.assertEqual(task.sub_tasks, ["query_reranking"])
    candidate_lists = list(task.candidate_lists())
    self.assertLen(candidate_lists, 2)
    candidates = candidate_lists[1]
    self.assertLen(candidates, 5)
    self.assertEqual(candidates[0].context.id, candidates[0].text)
    self.assertIsNone(candidates[0].context.title)
    self.assertEqual(candidates[0].text, "At what temperature does steel melt?")
    self.assertEqual(candidates[1].context.id, candidates[1].text)
    self.assertEqual(
        candidates[1].text, "At what temperature does steel melts?"
    )
    self.assertEqual(candidates[2].context.id, candidates[2].text)
    self.assertEqual(candidates[2].text, "At what tempo, does shale melt?")
    self.assertEqual(candidates[3].context.id, candidates[3].text)
    self.assertEqual(candidates[3].text, "At what degree does steel liquify?")
    self.assertEqual(candidates[4].context.id, candidates[4].text)
    self.assertEqual(
        candidates[4].text, "At what heat intensity does steel melt?"
    )

  def test_svq_query_reranking_sounds(self):
    task = svq.SVQEnUsQueryReranking()
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

  def test_svq_query_reranking_examples(self):
    task = svq.SVQEnUsQueryReranking()
    examples = list(task.examples("query_reranking"))
    self.assertLen(examples, 2)
    example = examples[0]
    self.assertEqual(example.sound_id, "utt_11697423627206642872")
    self.assertLen(example.texts, 5)
    self.assertEqual(example.language, "en_us")
    example = examples[1]
    self.assertEqual(example.sound_id, "utt_15041124811443622614")
    self.assertLen(example.texts, 5)
    self.assertEqual(example.language, "en_us")


if __name__ == "__main__":
  absltest.main()
