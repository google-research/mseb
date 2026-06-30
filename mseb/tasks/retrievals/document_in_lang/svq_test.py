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
from absl.testing import parameterized
from mseb import dataset
import pytest
import tensorflow_datasets as tfds

retrieval = pytest.importorskip('mseb.tasks.retrieval')
svq = pytest.importorskip('mseb.tasks.retrievals.document_in_lang.svq')
FLAGS = flags.FLAGS


@pytest.mark.scann
@pytest.mark.optional
class SVQEnUsDocumentInLangRetrievalTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent.parent.parent,
        'testdata',
    )
    # Add a .git marker to prevent SimpleVoiceQuestionsDataset from trying to
    # download the data from Huggingface.
    cache_dir = self.create_tempdir().full_path
    shutil.rmtree(cache_dir)
    shutil.copytree(testdata_path, cache_dir)
    os.chmod(cache_dir, 0o755)
    pathlib.Path.touch(pathlib.Path(os.path.join(cache_dir, '.git')))
    self.enter_context(
        flagsaver.flagsaver((dataset._DATASET_BASEPATH, cache_dir))
    )

  def test_svq_document_in_lang_retrieval_documents(self):
    with tfds.testing.mock_data(num_examples=1):
      task = svq.SVQEnUsDocumentInLangRetrieval()
      self.assertEqual(task.sub_tasks[0], 'document_retrieval_in_lang')
      for document in task.documents():
        self.assertEqual(document.context.id, 'chg dif hhia i e ce')
        self.assertEqual(document.context.id, document.context.title)
        self.assertEqual(document.text, 'gebc   ahgjefjhfef')

  def test_svq_en_us_document_in_lang_retrieval_sounds(self):
    task = svq.SVQEnUsDocumentInLangRetrieval()
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

  def test_svq_en_us_document_in_lang_retrieval_examples(self):
    task = svq.SVQEnUsDocumentInLangRetrieval()
    examples = list(task.examples('document_retrieval_in_lang'))
    self.assertLen(examples, 2)
    example = examples[0]
    self.assertEqual(example.sound_id, 'utt_11697423627206642872')
    self.assertEqual(example.reference_id, 'Red-short carbon steel')
    example = examples[1]
    self.assertEqual(example.sound_id, 'utt_15041124811443622614')
    self.assertEqual(example.reference_id, 'Red-short carbon steel')


@pytest.mark.scann
@pytest.mark.optional
class MakeTaskClassTest(absltest.TestCase):
  """Tests for the _make_task_class factory function."""

  def test_class_name_full_index(self):
    cls = svq._make_task_class(
        svq.SVQDocumentInLangRetrieval,
        locale='en_us',
        suffix='EnUs',
        eval_lang='en-US',
        description='test description',
    )
    self.assertEqual(cls.__name__, 'SVQEnUsDocumentInLangRetrieval')

  def test_class_name_small_index(self):
    cls = svq._make_task_class(
        svq.SVQDocumentInLangRetrievalSmallIndex,
        locale='en_us',
        suffix='EnUs',
        eval_lang='en-US',
        description='test description',
    )
    self.assertEqual(cls.__name__, 'SVQEnUsDocumentInLangRetrievalSmallIndex')

  def test_locale_set(self):
    cls = svq._make_task_class(
        svq.SVQDocumentInLangRetrieval,
        locale='fi_fi',
        suffix='FiFi',
        eval_lang='fi-FI',
        description='test description',
    )
    self.assertEqual(cls.locale, 'fi_fi')

  def test_metadata_fields(self):
    cls = svq._make_task_class(
        svq.SVQDocumentInLangRetrieval,
        locale='ko_kr',
        suffix='KoKr',
        eval_lang='ko-KR',
        description='Document in-lang retrieval task.',
    )
    self.assertEqual(cls.metadata.name, 'SVQKoKrDocumentInLangRetrieval')
    self.assertEqual(
        cls.metadata.description, 'Document in-lang retrieval task.'
    )
    self.assertEqual(cls.metadata.eval_langs, ['ko-KR'])
    self.assertEqual(cls.metadata.main_score, 'MRR')
    self.assertEqual(cls.metadata.type, 'DocumentInLangRetrieval')

  def test_inheritance_full_index(self):
    cls = svq._make_task_class(
        svq.SVQDocumentInLangRetrieval,
        locale='en_us',
        suffix='EnUs',
        eval_lang='en-US',
        description='test',
    )
    self.assertTrue(issubclass(cls, svq.SVQDocumentInLangRetrieval))
    self.assertTrue(issubclass(cls, retrieval.RetrievalTask))

  def test_inheritance_small_index(self):
    cls = svq._make_task_class(
        svq.SVQDocumentInLangRetrievalSmallIndex,
        locale='en_us',
        suffix='EnUs',
        eval_lang='en-US',
        description='test',
    )
    self.assertTrue(issubclass(cls, svq.SVQDocumentInLangRetrievalSmallIndex))
    self.assertTrue(issubclass(cls, svq.SVQDocumentInLangRetrieval))


@pytest.mark.scann
@pytest.mark.optional
class GeneratedClassesTest(absltest.TestCase):
  """Tests that all expected classes are generated and registered."""

  def test_all_full_index_classes_exist(self):
    for locale, (suffix, _) in svq._SVQ_LOCALES.items():
      class_name = f'SVQ{suffix}DocumentInLangRetrieval'
      cls = getattr(svq, class_name)
      self.assertEqual(cls.locale, locale)
      self.assertTrue(issubclass(cls, svq.SVQDocumentInLangRetrieval))

  def test_all_small_index_classes_exist(self):
    for locale, (suffix, _) in svq._SVQ_LOCALES.items():
      class_name = f'SVQ{suffix}DocumentInLangRetrievalSmallIndex'
      cls = getattr(svq, class_name)
      self.assertEqual(cls.locale, locale)
      self.assertTrue(issubclass(cls, svq.SVQDocumentInLangRetrievalSmallIndex))

  def test_expected_class_count(self):
    num_locales = len(svq._SVQ_LOCALES)
    generated = [
        name
        for name in dir(svq)
        if name.startswith('SVQ')
        and name != 'SVQDocumentInLangRetrieval'
        and name != 'SVQDocumentInLangRetrievalSmallIndex'
    ]
    self.assertLen(generated, num_locales * 2)


@pytest.mark.scann
@pytest.mark.optional
class GeneratedClassMetadataTest(parameterized.TestCase):
  """Spot-check metadata on a few generated classes."""

  @parameterized.parameters(
      ('SVQArEgDocumentInLangRetrieval', 'ar_eg', ['ar-EG']),
      ('SVQEnUsDocumentInLangRetrieval', 'en_us', ['en-US']),
      ('SVQSwDocumentInLangRetrieval', 'sw', ['sw']),
      ('SVQTeInDocumentInLangRetrieval', 'te_in', ['te-IN']),
  )
  def test_full_index_metadata(self, class_name, locale, eval_langs):
    cls = getattr(svq, class_name)
    self.assertEqual(cls.metadata.name, class_name)
    self.assertEqual(cls.locale, locale)
    self.assertEqual(cls.metadata.eval_langs, eval_langs)
    self.assertEqual(cls.metadata.type, 'DocumentInLangRetrieval')
    self.assertEqual(
        cls.metadata.reference,
        'https://huggingface.co/datasets/google/svq',
    )

  @parameterized.parameters(
      ('SVQEnUsDocumentInLangRetrievalSmallIndex', 'en_us', ['en-US']),
      ('SVQFiFiDocumentInLangRetrievalSmallIndex', 'fi_fi', ['fi-FI']),
  )
  def test_small_index_metadata(self, class_name, locale, eval_langs):
    cls = getattr(svq, class_name)
    self.assertEqual(cls.metadata.name, class_name)
    self.assertEqual(cls.locale, locale)
    self.assertEqual(cls.metadata.eval_langs, eval_langs)


@pytest.mark.scann
@pytest.mark.optional
class IndexDirTest(parameterized.TestCase):
  """Tests that index_dir is correctly computed for generated classes."""

  def setUp(self):
    super().setUp()
    from mseb import task as task_lib  # pylint: disable=g-import-not-at-top

    self.enter_context(
        flagsaver.flagsaver((task_lib.TASK_CACHE_BASEPATH, '/tmp/test_cache'))
    )

  @parameterized.parameters(
      ('SVQEnUsDocumentInLangRetrieval', 'svq_en_document_retrieval_in_lang'),
      ('SVQArEgDocumentInLangRetrieval', 'svq_ar_document_retrieval_in_lang'),
      ('SVQSwDocumentInLangRetrieval', 'svq_sw_document_retrieval_in_lang'),
  )
  def test_full_index_dir_suffix(self, class_name, expected_suffix):
    cls = getattr(svq, class_name)
    task = cls()
    self.assertTrue(task.index_dir.endswith(expected_suffix))

  def test_small_index_dir(self):
    task = svq.SVQEnUsDocumentInLangRetrievalSmallIndex()
    self.assertTrue(
        task.index_dir.endswith('svq_document_retrieval_in_lang_small_index')
    )
    # Small index dir should NOT contain the full index path component.
    self.assertNotIn('svq_en_document_retrieval_in_lang', task.index_dir)


if __name__ == '__main__':
  absltest.main()
