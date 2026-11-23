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

"""Tests for registered MSEB task metadata invariants."""

from collections.abc import Mapping
import types

from absl.testing import absltest
from mseb import task as task_lib
import mseb.tasks  # pylint: disable=unused-import  Registers all MSEB tasks.

_EXPECTED_SVQ_DOC_BY_TYPE: Mapping[str, str | None] = types.MappingProxyType({
    'Clustering': '/svq_clustering.md',
    'DocumentCrossLangRetrieval': '/svq_retrieval.md',
    'DocumentInLangRetrieval': '/svq_retrieval.md',
    'PassageCrossLangRetrieval': '/svq_retrieval.md',
    'PassageInLangRetrieval': '/svq_retrieval.md',
    'QueryReranking': '/svq_reranking.md',
    'SalientTermReranking': '/svq_reranking.md',
    'SalientTermSegmentation': '/svq_segmentation.md',
    'SpanCrossLangReasoning': '/svq_reasoning.md',
    'SpanInLangReasoning': '/svq_reasoning.md',
    'SpeakerGenderClassification': '/svq_classification.md',
    'SpeechTranscription': '/svq_transcription.md',
    'Stability': None,
})


class RegisteredTaskMetadataTest(absltest.TestCase):

  def test_documentation_fields_are_urls(self):
    tasks = task_lib.get_name_to_task()
    self.assertNotEmpty(tasks)
    for name, task_cls in tasks.items():
      metadata = task_cls.metadata
      for field_name, value in (
          ('documentation_file', metadata.documentation_file),
          ('dataset_documentation_file', metadata.dataset_documentation_file),
      ):
        if value is not None:
          with self.subTest(task=name, field=field_name):
            self.assertStartsWith(value, ('http://', 'https://'))

  def test_svq_tasks_reference_matching_task_doc(self):
    tasks = task_lib.get_name_to_task()
    seen_svq_types: set[str] = set()
    for name, task_cls in tasks.items():
      if not name.startswith('SVQ'):
        continue
      task_type = task_cls.metadata.type
      seen_svq_types.add(task_type)
      with self.subTest(task=name):
        self.assertIn(task_type, _EXPECTED_SVQ_DOC_BY_TYPE)
        expected_suffix = _EXPECTED_SVQ_DOC_BY_TYPE[task_type]
        doc_url = task_cls.metadata.documentation_file
        if expected_suffix is None:
          self.assertIsNone(doc_url)
        else:
          self.assertIsNotNone(doc_url)
          self.assertEndsWith(doc_url, expected_suffix)
        dataset_doc_url = task_cls.metadata.dataset_documentation_file
        self.assertIsNotNone(dataset_doc_url)
        self.assertEndsWith(dataset_doc_url, '/dataset_svq.md')

    self.assertEqual(seen_svq_types, set(_EXPECTED_SVQ_DOC_BY_TYPE))


if __name__ == '__main__':
  absltest.main()
