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

from absl.testing import absltest
from absl.testing import flagsaver
from mseb import dataset
import pytest

svq = pytest.importorskip("mseb.tasks.transcriptions.speech.svq")


@pytest.mark.whisper
@pytest.mark.optional
class SVQEnUsSpeechTranscriptionTest(absltest.TestCase):

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
    shutil.copyfile(
        os.path.join(cache_dir, "speech_transcription.jsonl"),
        os.path.join(cache_dir, "utts_en_us_clean.jsonl"),
    )
    self.enter_context(
        flagsaver.flagsaver((dataset._DATASET_BASEPATH, cache_dir))
    )

  def test_svq_en_us_speech_transcription_sounds(self):
    task = svq.SVQEnUsSpeechTranscription()
    sounds = list(task.multimodal_inputs())
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

  def test_svq_en_us_speech_transcription_examples(self):
    task = svq.SVQEnUsSpeechTranscription()
    examples = list(task.examples("speech_transcription"))
    self.assertLen(examples, 2)
    example = examples[0]
    self.assertEqual(example.sound_id, "utt_11697423627206642872")
    self.assertEqual(example.text, "At what temperature does steel melt?")
    self.assertEqual(example.language, "en_us")
    example = examples[1]
    self.assertEqual(example.sound_id, "utt_15041124811443622614")
    self.assertEqual(example.text, "At what temperature does steel melt?")
    self.assertEqual(example.language, "en_us")

  def test_svq_en_us_speech_transcription_multimodal_inputs_beam(self):
    task = svq.SVQEnUsSpeechTranscription()
    transform = task.multimodal_inputs_beam()
    self.assertIsNotNone(transform)


class DynamicClassGenerationTest(absltest.TestCase):
  """Tests for the factory-generated locale-specific task classes."""

  def test_all_locale_classes_exist(self):
    for locale, (suffix, _) in svq._SVQ_LOCALES.items():
      class_name = f"SVQ{suffix}SpeechTranscription"
      self.assertTrue(
          hasattr(svq, class_name),
          f"Missing class {class_name} for locale {locale}",
      )

  def test_locale_class_has_correct_locale(self):
    task_cls = getattr(svq, "SVQEnUsSpeechTranscription")
    self.assertEqual(task_cls.locale, "en_us")

  def test_locale_class_has_correct_metadata_name(self):
    task_cls = getattr(svq, "SVQArEgSpeechTranscription")
    self.assertEqual(task_cls.metadata.name, "SVQArEgSpeechTranscription")

  def test_locale_class_has_correct_eval_langs(self):
    task_cls = getattr(svq, "SVQFiFiSpeechTranscription")
    self.assertEqual(task_cls.metadata.eval_langs, ["fi-FI"])

  def test_locale_class_inherits_from_base(self):
    task_cls = getattr(svq, "SVQKoKrSpeechTranscription")
    self.assertTrue(issubclass(task_cls, svq.SVQSpeechTranscription))

  def test_locale_class_default_size_is_none(self):
    task_cls = getattr(svq, "SVQSwSpeechTranscription")
    self.assertIsNone(task_cls.size)

  def test_compact_classes_exist(self):
    for locale, (suffix, _) in svq._SVQ_LOCALES.items():
      class_name = f"SVQ{suffix}SpeechTranscriptionCompact"
      self.assertTrue(
          hasattr(svq, class_name),
          f"Missing compact class {class_name} for locale {locale}",
      )

  def test_compact_class_has_correct_size(self):
    task_cls = getattr(svq, "SVQEnUsSpeechTranscriptionCompact")
    self.assertEqual(task_cls.size, "compact")

  def test_compact_class_has_correct_locale(self):
    task_cls = getattr(svq, "SVQRuRuSpeechTranscriptionCompact")
    self.assertEqual(task_cls.locale, "ru_ru")

  def test_compact_class_inherits_from_base(self):
    task_cls = getattr(svq, "SVQTeInSpeechTranscriptionCompact")
    self.assertTrue(issubclass(task_cls, svq.SVQSpeechTranscription))

  def test_debug_classes_exist(self):
    for suffix in ("EnUs", "FiFi"):
      class_name = f"SVQ{suffix}SpeechTranscriptionDebug"
      self.assertTrue(
          hasattr(svq, class_name),
          f"Missing debug class {class_name}",
      )

  def test_debug_class_has_correct_size(self):
    task_cls = getattr(svq, "SVQEnUsSpeechTranscriptionDebug")
    self.assertEqual(task_cls.size, "debug")

  def test_debug_only_en_us_and_fi_fi(self):
    """Debug classes should only exist for en-US and fi-FI."""
    for locale, (suffix, _) in svq._SVQ_LOCALES.items():
      class_name = f"SVQ{suffix}SpeechTranscriptionDebug"
      if locale in ("en_us", "fi_fi"):
        self.assertTrue(hasattr(svq, class_name))
      else:
        self.assertFalse(
            hasattr(svq, class_name),
            f"Unexpected debug class {class_name}",
        )

  def test_metadata_type_is_speech_transcription(self):
    task_cls = getattr(svq, "SVQHiInSpeechTranscription")
    self.assertEqual(task_cls.metadata.type, "SpeechTranscription")

  def test_metadata_main_score_is_wer(self):
    task_cls = getattr(svq, "SVQJaJpSpeechTranscription")
    self.assertEqual(task_cls.metadata.main_score, "WER")

  def test_total_class_count(self):
    """26 locales * 2 (default + compact) + 2 debug = 54 classes."""
    num_expected = len(svq._SVQ_LOCALES) * 2 + 2
    generated = [
        name
        for name in dir(svq)
        if name.startswith("SVQ")
        and name != "SVQSpeechTranscription"
        and isinstance(getattr(svq, name), type)
        and issubclass(getattr(svq, name), svq.SVQSpeechTranscription)
    ]
    self.assertLen(generated, num_expected)

  def test_ur_pk_locale_exists(self):
    """Verify the new ur_pk locale is included."""
    self.assertIn("ur_pk", svq._SVQ_LOCALES)
    self.assertTrue(hasattr(svq, "SVQUrPkSpeechTranscription"))
    self.assertTrue(hasattr(svq, "SVQUrPkSpeechTranscriptionCompact"))


class BaseClassTest(absltest.TestCase):
  """Tests for SVQSpeechTranscription base class attributes."""

  def test_base_locale_is_none(self):
    self.assertIsNone(svq.SVQSpeechTranscription.locale)

  def test_base_size_is_none(self):
    self.assertIsNone(svq.SVQSpeechTranscription.size)

  def test_sub_tasks(self):
    task = svq.SVQSpeechTranscription()
    self.assertIn("speech_transcription", task.sub_tasks)
    self.assertIn("speech_transcription:clean", task.sub_tasks)
    self.assertIn("speech_transcription:media_noise", task.sub_tasks)
    self.assertIn("speech_transcription:traffic_noise", task.sub_tasks)
    self.assertIn("speech_transcription:background_speech", task.sub_tasks)


if __name__ == "__main__":
  absltest.main()
