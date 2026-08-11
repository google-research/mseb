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

svq = pytest.importorskip('mseb.tasks.retrievals.passage_in_lang.svq')
FLAGS = flags.FLAGS


def _setup_testdata(test_case):
  """Sets up a temp dir with SVQ testdata and configures the dataset flag."""
  testdata_path = os.path.join(
      pathlib.Path(os.path.abspath(__file__)).parent.parent.parent.parent,
      'testdata',
  )
  cache_dir = test_case.create_tempdir().full_path
  shutil.rmtree(cache_dir)
  shutil.copytree(testdata_path, cache_dir)
  os.chmod(cache_dir, 0o755)
  pathlib.Path.touch(pathlib.Path(os.path.join(cache_dir, '.git')))
  test_case.enter_context(
      flagsaver.flagsaver((dataset._DATASET_BASEPATH, cache_dir))
  )


@pytest.mark.scann
@pytest.mark.optional
class SVQEnUsPassageInLangRetrievalTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    _setup_testdata(self)

  def test_svq_passage_in_lang_retrieval_documents(self):
    task = svq.SVQEnUsPassageInLangRetrieval()
    self.assertEqual(task.sub_tasks[0], 'passage_retrieval_in_lang')
    documents = list(task.documents())
    self.assertLen(documents, 3)
    document = documents[2]
    self.assertEqual(document.context.id, 'english-1064747448949054415-7')
    self.assertEqual(document.context.title, 'Little Albert experiment')
    self.assertTrue(document.text.startswith('Albert was about one year'))

  def test_svq_en_us_passage_in_lang_retrieval_sounds(self):
    task = svq.SVQEnUsPassageInLangRetrieval()
    sounds = list(task.multimodal_inputs())
    self.assertLen(sounds, 2)
    sound = sounds[0]
    self.assertEqual(sound.context.id, 'utt_11697423627206642872')
    self.assertEqual(sound.context.speaker_id, 'speaker_5452472707103026757')
    self.assertEqual(sound.context.speaker_age, 27)
    self.assertEqual(sound.context.language, 'en_us')
    sound = sounds[1]
    self.assertEqual(sound.context.id, 'utt_15041124811443622614')
    self.assertEqual(sound.context.speaker_id, 'speaker_10322347911861405809')
    self.assertEqual(sound.context.speaker_age, 25)
    self.assertEqual(sound.context.language, 'en_us')

  def test_svq_en_us_passage_in_lang_retrieval_examples(self):
    task = svq.SVQEnUsPassageInLangRetrieval()
    examples = list(task.examples('passage_retrieval_in_lang'))
    self.assertLen(examples, 2)
    example = examples[0]
    self.assertEqual(example.sound_id, 'utt_11697423627206642872')
    self.assertEqual(example.reference_id, 'english-6037841464917965779-1')
    example = examples[1]
    self.assertEqual(example.sound_id, 'utt_15041124811443622614')
    self.assertEqual(example.reference_id, 'english-6037841464917965779-1')


class DynamicClassGenerationTest(absltest.TestCase):
  """Tests for the factory-generated locale-specific task classes."""

  def test_all_locale_classes_exist(self):
    for locale, (suffix, _) in svq._SVQ_LOCALES.items():
      class_name = f'SVQ{suffix}PassageInLangRetrieval'
      self.assertTrue(
          hasattr(svq, class_name),
          f'Missing class {class_name} for locale {locale}',
      )

  def test_locale_class_has_correct_locale(self):
    task_cls = getattr(svq, 'SVQEnUsPassageInLangRetrieval')
    self.assertEqual(task_cls.locale, 'en_us')

  def test_locale_class_has_correct_metadata_name(self):
    task_cls = getattr(svq, 'SVQArEgPassageInLangRetrieval')
    self.assertEqual(task_cls.metadata.name, 'SVQArEgPassageInLangRetrieval')

  def test_locale_class_has_correct_eval_langs(self):
    task_cls = getattr(svq, 'SVQFiFiPassageInLangRetrieval')
    self.assertEqual(task_cls.metadata.eval_langs, ['fi-FI'])

  def test_locale_class_inherits_from_base(self):
    task_cls = getattr(svq, 'SVQEnUsPassageInLangRetrieval')
    self.assertTrue(issubclass(task_cls, svq.SVQPassageInLangRetrieval))

  def test_metadata_type_is_passage_in_lang_retrieval(self):
    task_cls = getattr(svq, 'SVQKoKrPassageInLangRetrieval')
    self.assertEqual(task_cls.metadata.type, 'PassageInLangRetrieval')

  def test_metadata_main_score_is_mrr(self):
    task_cls = getattr(svq, 'SVQSwPassageInLangRetrieval')
    self.assertEqual(task_cls.metadata.main_score, 'MRR')


class BaseClassTest(absltest.TestCase):
  """Tests for SVQPassageInLangRetrieval base class attributes."""

  def test_base_locale_is_none(self):
    self.assertIsNone(svq.SVQPassageInLangRetrieval.locale)

  def test_sub_tasks(self):
    task = svq.SVQPassageInLangRetrieval()
    expected = [
        'passage_retrieval_in_lang',
        'passage_retrieval_in_lang:clean',
        'passage_retrieval_in_lang:media_noise',
        'passage_retrieval_in_lang:traffic_noise',
        'passage_retrieval_in_lang:background_speech',
    ]
    self.assertEqual(task.sub_tasks, expected)


@pytest.mark.scann
@pytest.mark.optional
class TaskDataFilteringTest(absltest.TestCase):
  """Tests for _task_data locale filtering."""

  def setUp(self):
    super().setUp()
    _setup_testdata(self)

  def test_task_data_filters_by_locale(self):
    task = svq.SVQEnUsPassageInLangRetrieval()
    df = task._task_data(
        'passage_retrieval_in_lang',
        dtype={'locale': str, 'utt_id': str},
    )
    for row in df.to_dict('records'):
      self.assertEqual(row['locale'], 'en_us')

  def test_task_data_no_locale_returns_all(self):
    task = svq.SVQPassageInLangRetrieval()
    df = task._task_data(
        'passage_retrieval_in_lang',
        dtype={'locale': str, 'utt_id': str},
    )
    self.assertNotEmpty(df)


if __name__ == '__main__':
  absltest.main()
