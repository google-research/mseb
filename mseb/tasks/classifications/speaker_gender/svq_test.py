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
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from mseb import dataset
from mseb.tasks.classifications.speaker_gender import svq


FLAGS = flags.FLAGS


class SVQEnUsSpeakerGenderClassificationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent.parent.parent,
        "testdata",
    )
    self.enter_context(
        flagsaver.flagsaver((
            dataset._DATASET_BASEPATH,
            os.path.join(testdata_path, "svq_mini"),
        ))
    )

  @mock.patch("mseb.utils.download_from_hf")
  def test_svq_speaker_gender_classification_sounds(self, _):
    task = svq.SVQEnUsSpeakerGenderClassification()
    sounds = list(task.sounds())
    self.assertLen(sounds, 10)
    sound = sounds[0]
    self.assertEqual(
        sound.context.id, "utt_6844631007344632667"
    )
    self.assertEqual(sound.context.speaker_id, "speaker_14599080134788979042")
    self.assertEqual(sound.context.speaker_age, 21)
    self.assertEqual(sound.context.language, "en_us")
    self.assertLen(sound.waveform, 311040)

  @mock.patch("mseb.utils.download_from_hf")
  def test_svq_speaker_gender_classification_examples(self, _):
    task = svq.SVQEnUsSpeakerGenderClassification()
    examples = list(task.examples("speaker_gender_classification:clean"))
    self.assertLen(examples, 3)
    example = examples[0]
    self.assertEqual(
        example.example_id, "utt_13729869686284260222"
    )
    self.assertEqual(example.label_id, "Female")
    example = examples[1]
    self.assertEqual(
        example.example_id, "utt_2118836283433598088"
    )
    self.assertEqual(example.label_id, "Male")

  @mock.patch("mseb.utils.download_from_hf")
  def test_svq_speaker_gender_classification_class_labels(self, _):
    task = svq.SVQEnUsSpeakerGenderClassification()
    self.assertContainsSubset(["speaker_gender_classification"], task.sub_tasks)
    class_labels = list(task.class_labels())
    self.assertLen(class_labels, 2)
    self.assertEqual(class_labels, ["Female", "Male"])
    self.assertEqual(task.task_type, "multi_class")


if __name__ == "__main__":
  absltest.main()
