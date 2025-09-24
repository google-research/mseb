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

from absl.testing import absltest
from absl.testing import flagsaver
from mseb.tasks.reasonings.span_in_lang import svq


class SVQEnUsSpanInLangReasoningTest(absltest.TestCase):

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
        flagsaver.flagsaver((svq.svq_data.SVQ_BASEPATH, cache_dir))
    )

  def test_svq_span_in_lang_reasoning_span_lists(self):
    task = svq.SVQEnUsSpanInLangReasoningGecko()
    self.assertEqual(task.sub_tasks, ["span_reasoning_in_lang"])
    span_lists = list(task.span_lists())
    self.assertLen(span_lists, 2)
    spans = span_lists[1]
    self.assertLen(spans, 5)
    self.assertEqual(spans[0].context.id, spans[0].text)
    self.assertIsNone(spans[0].context.title)
    self.assertEqual(spans[0].text, "At what temperature does steel melt?")
    self.assertEqual(spans[1].context.id, spans[1].text)
    self.assertEqual(
        spans[1].text, "At what temperature does steel melts?"
    )
    self.assertEqual(spans[2].context.id, spans[2].text)
    self.assertEqual(spans[2].text, "At what tempo, does shale melt?")
    self.assertEqual(spans[3].context.id, spans[3].text)
    self.assertEqual(spans[3].text, "At what degree does steel liquify?")
    self.assertEqual(spans[4].context.id, spans[4].text)
    self.assertEqual(
        spans[4].text, "At what heat intensity does steel melt?"
    )

  def test_svq_span_in_lang_reasoning_sounds(self):
    task = svq.SVQEnUsSpanInLangReasoningGecko()
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

  def test_svq_span_in_lang_reasoning_examples(self):
    task = svq.SVQEnUsSpanInLangReasoningGecko()
    examples = list(task.examples("span_reasoning_in_lang"))
    self.assertLen(examples, 2)
    example = examples[0]
    self.assertEqual(example.sound_id, "utt_11697423627206642872")
    self.assertLen(example.texts, 5)
    self.assertEqual(
        example.reference_answer, "above 900 \\u00b0F"
    )
    example = examples[1]
    self.assertEqual(example.sound_id, "utt_15041124811443622614")
    self.assertLen(example.texts, 5)
    self.assertEqual(
        example.reference_answer, "No Answer"
    )


if __name__ == "__main__":
  absltest.main()
