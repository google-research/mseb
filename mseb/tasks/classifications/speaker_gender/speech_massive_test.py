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
from mseb.tasks.classifications.speaker_gender import speech_massive


FLAGS = flags.FLAGS


class SpeechMassiveDeDeSpeakerGenderClassificationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent.parent.parent,
        "testdata",
    )
    self.enter_context(
        flagsaver.flagsaver((
            dataset._DATASET_BASEPATH,
            os.path.join(testdata_path, "speech_massive"),
        ))
    )

  @mock.patch("mseb.utils.download_from_hf")
  def test_speech_massive_speaker_gender_classification_sounds(self, _):
    task = speech_massive.SpeechMassiveDeDeSpeakerGenderClassification()
    sounds = list(task.sounds())
    self.assertLen(sounds, 2)
    sound = sounds[0]
    self.assertEqual(
        sound.context.id, "test/c15b5445ba46918a8d678e7b59b80aa6.wav"
    )
    self.assertEqual(sound.context.speaker_id, "5f32d5f107d49607c3f6cf7a")
    self.assertEqual(sound.context.speaker_age, 40)
    self.assertEqual(sound.context.language, "de_de")
    self.assertLen(sound.waveform, 120960)

  @mock.patch("mseb.utils.download_from_hf")
  def test_speech_massive_speaker_gender_classification_examples(self, _):
    task = speech_massive.SpeechMassiveDeDeSpeakerGenderClassification()
    examples = list(task.examples("speaker_gender_classification"))
    self.assertLen(examples, 2)
    example = examples[0]
    self.assertEqual(
        example.example_id, "test/c15b5445ba46918a8d678e7b59b80aa6.wav"
    )
    self.assertEqual(example.label_id, "Female")
    example = examples[1]
    self.assertEqual(
        example.example_id, "test/ef13f68d170a7d5064690bbea059061c.wav"
    )
    self.assertEqual(example.label_id, "Female")

  @mock.patch("mseb.utils.download_from_hf")
  def test_speech_massive_speaker_gender_classification_class_labels(self, _):
    task = speech_massive.SpeechMassiveDeDeSpeakerGenderClassification()
    self.assertEqual(task.sub_tasks, ["speaker_gender_classification"])
    class_labels = list(task.class_labels())
    self.assertLen(class_labels, 2)
    self.assertEqual(class_labels, ["Female", "Male"])
    self.assertEqual(task.task_type, "multi_class")


if __name__ == "__main__":
  absltest.main()
