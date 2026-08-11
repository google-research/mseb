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

"""Tests for Speech Massive intent classification tasks."""

import os
import pathlib
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from mseb import dataset
from mseb.tasks.classifications.intent import speech_massive as speech_massive_intent

FLAGS = flags.FLAGS


def _setup_testdata(test_case):
  """Sets up testdata path for Speech Massive."""
  testdata_path = os.path.join(
      pathlib.Path(os.path.abspath(__file__)).parent.parent.parent.parent,
      'testdata',
  )
  test_case.enter_context(
      flagsaver.flagsaver((
          dataset._DATASET_BASEPATH,
          os.path.join(testdata_path, 'speech_massive'),
      ))
  )


class SpeechMassiveDeDeIntentClassificationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    _setup_testdata(self)

  @mock.patch('mseb.utils.download_from_hf')
  def test_speech_massive_intent_classification_sounds(self, _):
    task = speech_massive_intent.SpeechMassiveDeDeIntentClassification()
    sounds = list(task.multimodal_inputs())
    self.assertLen(sounds, 2)
    sound = sounds[0]
    self.assertEqual(
        sound.context.id, 'test/c15b5445ba46918a8d678e7b59b80aa6.wav'
    )
    self.assertEqual(sound.context.speaker_id, '5f32d5f107d49607c3f6cf7a')
    self.assertEqual(sound.context.speaker_age, 40)
    self.assertEqual(sound.context.language, 'de_de')
    self.assertLen(sound.waveform, 120960)

  @mock.patch('mseb.utils.download_from_hf')
  def test_speech_massive_intent_classification_examples(self, _):
    task = speech_massive_intent.SpeechMassiveDeDeIntentClassification()
    examples = list(task.examples('intent_classification'))
    self.assertLen(examples, 2)
    example = examples[0]
    self.assertEqual(
        example.example_id, 'test/c15b5445ba46918a8d678e7b59b80aa6.wav'
    )
    self.assertEqual(example.label_id, 'audio_volume_mute')
    example = examples[1]
    self.assertEqual(
        example.example_id, 'test/ef13f68d170a7d5064690bbea059061c.wav'
    )
    self.assertEqual(example.label_id, 'takeaway_query')

  @mock.patch('mseb.utils.download_from_hf')
  def test_speech_massive_intent_classification_class_labels(self, _):
    task = speech_massive_intent.SpeechMassiveDeDeIntentClassification()
    self.assertEqual(task.sub_tasks, ['intent_classification'])
    class_labels = list(task.class_labels())
    self.assertLen(class_labels, 60)


class DynamicClassGenerationTest(absltest.TestCase):
  """Tests for the factory-generated locale-specific task classes."""

  def test_all_locale_classes_exist(self):
    for locale, (
        suffix,
        _,
    ) in speech_massive_intent._SPEECH_MASSIVE_LOCALES.items():
      class_name = f'SpeechMassive{suffix}IntentClassification'
      self.assertTrue(
          hasattr(speech_massive_intent, class_name),
          f'Missing class {class_name} for locale {locale}',
      )

  def test_locale_class_has_correct_locale(self):
    task_cls = getattr(
        speech_massive_intent, 'SpeechMassiveDeDeIntentClassification'
    )
    self.assertEqual(task_cls.locale, 'de_de')

  def test_locale_class_has_correct_metadata_name(self):
    task_cls = getattr(
        speech_massive_intent, 'SpeechMassiveArSaIntentClassification'
    )
    self.assertEqual(
        task_cls.metadata.name, 'SpeechMassiveArSaIntentClassification'
    )

  def test_locale_class_has_correct_eval_langs(self):
    task_cls = getattr(
        speech_massive_intent, 'SpeechMassiveKoKrIntentClassification'
    )
    self.assertEqual(task_cls.metadata.eval_langs, ['ko-KR'])

  def test_locale_class_inherits_from_base(self):
    task_cls = getattr(
        speech_massive_intent, 'SpeechMassiveFrFrIntentClassification'
    )
    self.assertTrue(
        issubclass(
            task_cls,
            speech_massive_intent.SpeechMassiveIntentClassification,
        )
    )

  def test_metadata_type_is_intent_classification(self):
    task_cls = getattr(
        speech_massive_intent, 'SpeechMassiveViVnIntentClassification'
    )
    self.assertEqual(task_cls.metadata.type, 'IntentClassification')

  def test_metadata_main_score_is_accuracy(self):
    task_cls = getattr(
        speech_massive_intent, 'SpeechMassiveTrTrIntentClassification'
    )
    self.assertEqual(task_cls.metadata.main_score, 'Accuracy')

  def test_total_class_count(self):
    """12 locales (default) = 12 generated classes."""
    num_locales = len(speech_massive_intent._SPEECH_MASSIVE_LOCALES)
    generated = [
        name
        for name in dir(speech_massive_intent)
        if name.startswith('SpeechMassive')
        and name not in ('SpeechMassiveIntentClassification',)
        and isinstance(getattr(speech_massive_intent, name), type)
        and issubclass(
            getattr(speech_massive_intent, name),
            speech_massive_intent.SpeechMassiveIntentClassification,
        )
    ]
    self.assertLen(generated, num_locales)  # 12 default


class BaseClassTest(absltest.TestCase):
  """Tests for SpeechMassiveIntentClassification base class attributes."""

  def test_base_locale_is_none(self):
    self.assertIsNone(
        speech_massive_intent.SpeechMassiveIntentClassification.locale
    )

  def test_base_filename_is_none(self):
    self.assertIsNone(
        speech_massive_intent.SpeechMassiveIntentClassification.filename
    )

  def test_sub_tasks(self):
    task = speech_massive_intent.SpeechMassiveIntentClassification()
    self.assertEqual(task.sub_tasks, ['intent_classification'])

  def test_task_type(self):
    task = speech_massive_intent.SpeechMassiveIntentClassification()
    self.assertEqual(task.task_type, 'multi_class')


if __name__ == '__main__':
  absltest.main()
