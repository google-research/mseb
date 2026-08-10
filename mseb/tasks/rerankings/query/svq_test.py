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

"""Tests for SVQ query reranking tasks."""

import os
import pathlib
import shutil

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from mseb import dataset
from mseb import types
import pytest

svq = pytest.importorskip('mseb.tasks.rerankings.query.svq')

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


@pytest.mark.whisper
@pytest.mark.optional
class SVQEnUsQueryRerankingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    _setup_testdata(self)
    self.enter_context(flagsaver.flagsaver((svq._RANDOMIZE_CANDIDATES, False)))

  def test_seed_from_candidates(self):
    candidates = ['a', 'b', 'c']
    seed = svq._seed_from_candidates(candidates)
    self.assertEqual(
        seed,
        106066814613367738644872591937025180872126396142919988090859577352361712472268,
    )

  def test_get_context_text(self):
    candidates = ['a', 'b', 'c']
    context_text = svq._get_context_text(candidates, randomize=True)
    self.assertEqual(
        context_text,
        '[{"id": 0, "text": "b"}, {"id": 1, "text": "c"},'
        ' {"id": 2, "text": "a"}]',
    )

  def test_get_texts_and_rank_by_id(self):
    candidates = ['a', 'b', 'c']
    texts, rank_by_id = svq._get_texts_and_rank_by_id(
        candidates, randomize=True
    )
    self.assertEqual(texts, candidates)
    self.assertEqual(rank_by_id, {0: 1, 1: 2, 2: 0})

  def test_get_texts_and_rank_by_id_no_randomize(self):
    candidates = ['a', 'b', 'c']
    texts, rank_by_id = svq._get_texts_and_rank_by_id(
        candidates, randomize=False
    )
    self.assertEqual(texts, candidates)
    self.assertIsNone(rank_by_id)

  def test_svq_query_reranking_candidate_lists(self):
    task = svq.SVQEnUsQueryReranking()
    self.assertEqual(task.sub_tasks[0], 'query_reranking')
    candidate_lists = list(task.candidate_lists())
    self.assertLen(candidate_lists, 2)
    candidates = candidate_lists[1]
    self.assertLen(candidates, 5)
    self.assertEqual(candidates[0].context.id, candidates[0].text)
    self.assertIsNone(candidates[0].context.title)
    self.assertEqual(candidates[0].text, 'At what temperature does steel melt?')
    self.assertEqual(candidates[1].context.id, candidates[1].text)
    self.assertEqual(
        candidates[1].text, 'At what temperature does steel melts?'
    )
    self.assertEqual(candidates[2].context.id, candidates[2].text)
    self.assertEqual(candidates[2].text, 'At what tempo, does shale melt?')
    self.assertEqual(candidates[3].context.id, candidates[3].text)
    self.assertEqual(candidates[3].text, 'At what degree does steel liquify?')
    self.assertEqual(candidates[4].context.id, candidates[4].text)
    self.assertEqual(
        candidates[4].text, 'At what heat intensity does steel melt?'
    )

  def test_svq_query_reranking_sounds(self):
    task = svq.SVQEnUsQueryReranking()
    sounds = list(task.multimodal_inputs())
    self.assertLen(sounds, 2)
    sound = sounds[0]
    self.assertIsInstance(sound, types.SoundWithTitleAndContext)
    self.assertEqual(sound.context.id, 'utt_11697423627206642872')
    self.assertEqual(sound.context.speaker_id, 'speaker_5452472707103026757')
    self.assertEqual(sound.context.speaker_age, 27)
    self.assertEqual(sound.context.language, 'en_us')
    self.assertEqual(
        sound.context_text,
        '[{"id": 0, "text": "At what temperature does steel melt?"}, {"id": 1,'
        ' "text": "At what temperature does steel melts?"}, {"id": 2, "text":'
        ' "At what tempo, does shale melt?"}, {"id": 3, "text": "At what degree'
        ' does steel liquify?"}, {"id": 4, "text": "At what heat intensity does'
        ' steel melt?"}]',
    )
    sound = sounds[1]
    self.assertIsInstance(sound, types.SoundWithTitleAndContext)
    self.assertEqual(sound.context.id, 'utt_15041124811443622614')
    self.assertEqual(sound.context.speaker_id, 'speaker_10322347911861405809')
    self.assertEqual(sound.context.speaker_age, 25)
    self.assertEqual(sound.context.language, 'en_us')
    self.assertEqual(
        sound.context_text,
        '[{"id": 0, "text": "At what temperature does steel melt?"}, {"id": 1,'
        ' "text": "At what temperature does steel melts?"}, {"id": 2, "text":'
        ' "At what tempo, does shale melt?"}, {"id": 3, "text": "At what degree'
        ' does steel liquify?"}, {"id": 4, "text": "At what heat intensity does'
        ' steel melt?"}]',
    )

  def test_svq_query_reranking_examples(self):
    task = svq.SVQEnUsQueryReranking()
    examples = list(task.examples('query_reranking'))
    self.assertLen(examples, 2)
    example = examples[0]
    self.assertEqual(example.sound_id, 'utt_11697423627206642872')
    self.assertLen(example.texts, 5)
    self.assertEqual(example.language, 'en_us')
    example = examples[1]
    self.assertEqual(example.sound_id, 'utt_15041124811443622614')
    self.assertLen(example.texts, 5)
    self.assertEqual(example.language, 'en_us')


class DynamicClassGenerationTest(absltest.TestCase):
  """Tests for the factory-generated locale-specific task classes."""

  def test_all_locale_classes_exist(self):
    for locale, (suffix, _) in svq._SVQ_LOCALES.items():
      class_name = f'SVQ{suffix}QueryReranking'
      self.assertTrue(
          hasattr(svq, class_name),
          f'Missing class {class_name} for locale {locale}',
      )

  def test_locale_class_has_correct_locale(self):
    task_cls = getattr(svq, 'SVQEnUsQueryReranking')
    self.assertEqual(task_cls.locale, 'en_us')

  def test_locale_class_has_correct_metadata_name(self):
    task_cls = getattr(svq, 'SVQArEgQueryReranking')
    self.assertEqual(task_cls.metadata.name, 'SVQArEgQueryReranking')

  def test_locale_class_has_correct_eval_langs(self):
    task_cls = getattr(svq, 'SVQFiFiQueryReranking')
    self.assertEqual(task_cls.metadata.eval_langs, ['fi-FI'])

  def test_locale_class_inherits_from_base(self):
    task_cls = getattr(svq, 'SVQKoKrQueryReranking')
    self.assertTrue(issubclass(task_cls, svq.SVQQueryReranking))

  def test_metadata_type_is_query_reranking(self):
    task_cls = getattr(svq, 'SVQHiInQueryReranking')
    self.assertEqual(task_cls.metadata.type, 'QueryReranking')

  def test_metadata_main_score_is_map(self):
    task_cls = getattr(svq, 'SVQJaJpQueryReranking')
    self.assertEqual(task_cls.metadata.main_score, 'MAP')

  def test_total_class_count(self):
    """26 locales = 26 classes."""
    num_locales = len(svq._SVQ_LOCALES)
    generated = [
        name
        for name in dir(svq)
        if name.startswith('SVQ')
        and name != 'SVQQueryReranking'
        and isinstance(getattr(svq, name), type)
        and issubclass(getattr(svq, name), svq.SVQQueryReranking)
    ]
    self.assertLen(generated, num_locales)


class BaseClassTest(absltest.TestCase):
  """Tests for SVQQueryReranking base class attributes."""

  def test_base_locale_is_none(self):
    self.assertIsNone(svq.SVQQueryReranking.locale)

  def test_sub_tasks(self):
    task = svq.SVQQueryReranking()
    expected = [
        'query_reranking',
        'query_reranking:clean',
        'query_reranking:media_noise',
        'query_reranking:traffic_noise',
        'query_reranking:background_speech',
    ]
    self.assertEqual(task.sub_tasks, expected)


@pytest.mark.whisper
@pytest.mark.optional
class TaskDataFilteringTest(absltest.TestCase):
  """Tests for _task_data locale filtering."""

  def setUp(self):
    super().setUp()
    _setup_testdata(self)

  def test_task_data_filters_by_locale(self):
    task = svq.SVQEnUsQueryReranking()
    df = task._task_data(
        'query_reranking',
        dtype={'locale': str, 'utt_id': str},
    )
    for row in df.to_dict('records'):
      self.assertEqual(row['locale'], 'en_us')

  def test_task_data_no_locale_returns_all(self):
    task = svq.SVQQueryReranking()
    df = task._task_data(
        'query_reranking',
        dtype={'locale': str, 'utt_id': str},
    )
    self.assertNotEmpty(df)


if __name__ == '__main__':
  absltest.main()
