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

"""Tests for SVQ speaker gender classification tasks."""

import collections
import json
import os
import pathlib
import shutil
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from mseb import dataset
from mseb.tasks.classifications.speaker_gender import svq

FLAGS = flags.FLAGS


class GetEnvironmentTest(absltest.TestCase):
  """Tests for the _get_environment helper."""

  def test_no_colon(self):
    self.assertEqual(svq._get_environment("speaker_gender_classification"), "*")

  def test_with_colon(self):
    self.assertEqual(
        svq._get_environment("speaker_gender_classification:clean"), "clean"
    )
    self.assertEqual(
        svq._get_environment("speaker_gender_classification:traffic_noise"),
        "traffic_noise",
    )

  def test_multiple_colons(self):
    with self.assertRaises(ValueError):
      svq._get_environment("a:b:c")


class SVQSpeakerGenderClassificationBaseTest(absltest.TestCase):
  """Tests for the base class without a locale set."""

  def test_base_locale_is_none(self):
    self.assertIsNone(svq.SVQSpeakerGenderClassification.locale)

  def test_sub_tasks(self):
    task = svq.SVQSpeakerGenderClassification()
    expected = [
        "speaker_gender_classification",
        "speaker_gender_classification:clean",
        "speaker_gender_classification:media_noise",
        "speaker_gender_classification:traffic_noise",
        "speaker_gender_classification:background_speech",
    ]
    self.assertEqual(task.sub_tasks, expected)

  def test_multimodal_inputs_raises_without_locale(self):
    task = svq.SVQSpeakerGenderClassification()
    with self.assertRaisesRegex(ValueError, "locale"):
      list(task.multimodal_inputs())

  def test_examples_raises_without_locale(self):
    task = svq.SVQSpeakerGenderClassification()
    with self.assertRaisesRegex(ValueError, "locale"):
      list(task.examples("speaker_gender_classification"))


class SVQEnUsSpeakerGenderClassificationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent.parent.parent,
        "testdata",
    )
    svq_mini_dir = os.path.join(testdata_path, "svq_mini")
    temp_dir = self.create_tempdir().full_path
    shutil.copytree(svq_mini_dir, temp_dir, dirs_exist_ok=True)
    os.chmod(temp_dir, 0o755)

    utt_index_path = os.path.join(temp_dir, "utt_index.jsonl")
    with open(utt_index_path, "r") as f:
      records = [json.loads(line) for line in f if line.strip()]

    by_loc_env = collections.defaultdict(list)
    for record in records:
      by_loc_env[(record["locale"], record["environment"])].append(record)

    for (loc, env), recs in by_loc_env.items():
      path = os.path.join(temp_dir, f"utts_{loc}_{env}.jsonl")
      with open(path, "w") as f:
        for r in recs:
          f.write(json.dumps(r) + "\n")

    self.enter_context(
        flagsaver.flagsaver((
            dataset._DATASET_BASEPATH,
            temp_dir,
        ))
    )

  @mock.patch("mseb.utils.download_from_hf")
  def test_svq_speaker_gender_classification_sounds(self, _):
    task = svq.SVQEnUsSpeakerGenderClassification()
    sounds = list(task.multimodal_inputs())
    self.assertLen(sounds, 10)
    sound = sounds[0]
    self.assertEqual(sound.context.id, "utt_6844631007344632667")
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
    self.assertEqual(example.example_id, "utt_13729869686284260222")
    self.assertEqual(example.label_id, "Female")
    example = examples[1]
    self.assertEqual(example.example_id, "utt_2118836283433598088")
    self.assertEqual(example.label_id, "Male")

  @mock.patch("mseb.utils.download_from_hf")
  def test_svq_speaker_gender_classification_class_labels(self, _):
    task = svq.SVQEnUsSpeakerGenderClassification()
    self.assertContainsSubset(["speaker_gender_classification"], task.sub_tasks)
    class_labels = list(task.class_labels())
    self.assertLen(class_labels, 2)
    self.assertEqual(class_labels, ["Female", "Male"])
    self.assertEqual(task.task_type, "multi_class")


class DynamicClassGenerationTest(absltest.TestCase):
  """Tests for factory-generated locale-specific task classes."""

  def test_all_locale_classes_exist(self):
    for locale, (suffix, _) in svq._SVQ_LOCALES.items():
      class_name = f"SVQ{suffix}SpeakerGenderClassification"
      self.assertTrue(
          hasattr(svq, class_name),
          f"Missing class {class_name} for locale {locale}",
      )

  def test_locale_class_has_correct_locale(self):
    task_cls = getattr(svq, "SVQEnUsSpeakerGenderClassification")
    self.assertEqual(task_cls.locale, "en_us")

  def test_locale_class_has_correct_metadata_name(self):
    task_cls = getattr(svq, "SVQArEgSpeakerGenderClassification")
    self.assertEqual(
        task_cls.metadata.name, "SVQArEgSpeakerGenderClassification"
    )

  def test_locale_class_has_correct_eval_langs(self):
    task_cls = getattr(svq, "SVQFiFiSpeakerGenderClassification")
    self.assertEqual(task_cls.metadata.eval_langs, ["fi-FI"])

  def test_locale_class_inherits_from_base(self):
    task_cls = getattr(svq, "SVQKoKrSpeakerGenderClassification")
    self.assertTrue(issubclass(task_cls, svq.SVQSpeakerGenderClassification))

  def test_metadata_type_is_speaker_gender_classification(self):
    task_cls = getattr(svq, "SVQHiInSpeakerGenderClassification")
    self.assertEqual(task_cls.metadata.type, "SpeakerGenderClassification")

  def test_metadata_main_score_is_accuracy(self):
    task_cls = getattr(svq, "SVQJaJpSpeakerGenderClassification")
    self.assertEqual(task_cls.metadata.main_score, "Accuracy")


if __name__ == "__main__":
  absltest.main()
