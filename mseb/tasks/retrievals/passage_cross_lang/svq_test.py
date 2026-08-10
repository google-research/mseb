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

"""Tests for SVQ passage cross-lang retrieval tasks."""

from absl.testing import absltest
import pytest

svq = pytest.importorskip('mseb.tasks.retrievals.passage_cross_lang.svq')


class DynamicClassGenerationTest(absltest.TestCase):
  """Tests for the factory-generated locale-specific task classes."""

  def test_all_locale_classes_exist(self):
    for locale, (suffix, _) in svq._SVQ_LOCALES.items():
      class_name = f'SVQ{suffix}PassageCrossLangRetrieval'
      self.assertTrue(
          hasattr(svq, class_name),
          f'Missing class {class_name} for locale {locale}',
      )

  def test_locale_class_has_correct_locale(self):
    task_cls = getattr(svq, 'SVQFiFiPassageCrossLangRetrieval')
    self.assertEqual(task_cls.locale, 'fi_fi')

  def test_locale_class_has_correct_metadata_name(self):
    task_cls = getattr(svq, 'SVQArEgPassageCrossLangRetrieval')
    self.assertEqual(task_cls.metadata.name, 'SVQArEgPassageCrossLangRetrieval')

  def test_locale_class_has_correct_eval_langs(self):
    task_cls = getattr(svq, 'SVQKoKrPassageCrossLangRetrieval')
    self.assertEqual(task_cls.metadata.eval_langs, ['ko-KR'])

  def test_locale_class_inherits_from_base(self):
    task_cls = getattr(svq, 'SVQUrPkPassageCrossLangRetrieval')
    self.assertTrue(issubclass(task_cls, svq.SVQPassageCrossLangRetrieval))

  def test_metadata_type_is_passage_cross_lang_retrieval(self):
    task_cls = getattr(svq, 'SVQHiInPassageCrossLangRetrieval')
    self.assertEqual(task_cls.metadata.type, 'PassageCrossLangRetrieval')

  def test_metadata_main_score_is_mrr(self):
    task_cls = getattr(svq, 'SVQJaJpPassageCrossLangRetrieval')
    self.assertEqual(task_cls.metadata.main_score, 'MRR')

  def test_total_class_count(self):
    """19 locales variants (default) = 19 classes."""
    num_locales = len(svq._SVQ_LOCALES)
    generated = [
        name
        for name in dir(svq)
        if name.startswith('SVQ')
        and name != 'SVQPassageCrossLangRetrieval'
        and isinstance(getattr(svq, name), type)
        and issubclass(getattr(svq, name), svq.SVQPassageCrossLangRetrieval)
    ]
    self.assertLen(generated, num_locales)


class BaseClassTest(absltest.TestCase):
  """Tests for SVQPassageCrossLangRetrieval base class attributes."""

  def test_base_locale_is_none(self):
    self.assertIsNone(svq.SVQPassageCrossLangRetrieval.locale)

  def test_sub_tasks(self):
    task = svq.SVQPassageCrossLangRetrieval()
    expected = [
        'passage_retrieval_cross_lang',
        'passage_retrieval_cross_lang:clean',
        'passage_retrieval_cross_lang:media_noise',
        'passage_retrieval_cross_lang:traffic_noise',
        'passage_retrieval_cross_lang:background_speech',
    ]
    self.assertEqual(task.sub_tasks, expected)


if __name__ == '__main__':
  absltest.main()
