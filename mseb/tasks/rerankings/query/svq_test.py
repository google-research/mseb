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

import collections
import inspect
import json
import os
import pathlib
import shutil

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from mseb import dataset
from mseb import task as task_lib
from mseb import types
from mseb.tasks.rerankings.query import svq
import pytest

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
  for root, dirs, files in os.walk(cache_dir):
    for d in dirs:
      os.chmod(os.path.join(root, d), 0o755)
    for f in files:
      os.chmod(os.path.join(root, f), 0o644)
  pathlib.Path.touch(pathlib.Path(os.path.join(cache_dir, '.git')))

  # Update query_reranking.jsonl with boolean task column 'rerankings/query':
  # True.
  query_reranking_path = os.path.join(cache_dir, 'query_reranking.jsonl')
  with open(query_reranking_path, 'r') as f:
    records = [json.loads(line) for line in f if line.strip()]

  for r in records:
    r['rerankings/query'] = True

  # Add an extra record for another locale to verify locale-specific filtering
  de_record = {
      'text': 'Wie schmilzt Stahl?',
      'speaker_id': 'speaker_de_001',
      'speaker_gender': 'female',
      'speaker_age': 30,
      'environment': 'clean',
      'locale': 'de_de',
      'utt_id': 'utt_de_001',
      'rerankings/query': True,
      'candidates': ['Wie schmilzt Stahl?', 'Wann schmilzt Stahl?'],
  }
  records.append(de_record)

  with open(query_reranking_path, 'w') as f:
    for r in records:
      f.write(json.dumps(r) + '\n')

  by_loc_env = collections.defaultdict(list)
  for record in records:
    by_loc_env[(record['locale'], record['environment'])].append(record)

  for env in ('clean', 'media_noise', 'traffic_noise', 'background_speech'):
    if ('en_us', env) not in by_loc_env or not by_loc_env[('en_us', env)]:
      by_loc_env[('en_us', env)] = [{
          'locale': 'en_us',
          'utt_id': f'dummy_en_us_{env}',
          'environment': env,
          'rerankings/query': False,
          'candidates': [],
          'text': '',
      }]

  for (loc, env), recs in by_loc_env.items():
    path = os.path.join(cache_dir, f'utts_{loc}_{env}.jsonl')
    with open(path, 'w') as f:
      for r in recs:
        f.write(json.dumps(r) + '\n')

  test_case.enter_context(
      flagsaver.flagsaver((dataset._DATASET_BASEPATH, cache_dir))
  )


class GetEnvironmentTest(absltest.TestCase):
  """Tests for the _get_environment helper."""

  def test_no_colon(self):
    self.assertEqual(svq._get_environment('query_reranking'), '*')

  def test_with_colon(self):
    self.assertEqual(svq._get_environment('query_reranking:clean'), 'clean')
    self.assertEqual(
        svq._get_environment('query_reranking:media_noise'),
        'media_noise',
    )
    self.assertEqual(
        svq._get_environment('query_reranking:traffic_noise'),
        'traffic_noise',
    )
    self.assertEqual(
        svq._get_environment('query_reranking:background_speech'),
        'background_speech',
    )

  def test_multiple_colons(self):
    with self.assertRaises(ValueError):
      svq._get_environment('a:b:c')


class QueryRerankingHelpersTest(absltest.TestCase):
  """Tests for helper functions in the svq module."""

  def test_seed_from_candidates(self):
    candidates = ['a', 'b', 'c']
    seed = svq._seed_from_candidates(candidates)
    self.assertEqual(
        seed,
        106066814613367738644872591937025180872126396142919988090859577352361712472268,
    )

  def test_get_context_text_randomize(self):
    candidates = ['a', 'b', 'c']
    context_text = svq._get_context_text(candidates, randomize=True)
    self.assertEqual(
        context_text,
        '[{"id": 0, "text": "b"}, {"id": 1, "text": "c"},'
        ' {"id": 2, "text": "a"}]',
    )

  def test_get_context_text_no_randomize(self):
    candidates = ['a', 'b', 'c']
    context_text = svq._get_context_text(candidates, randomize=False)
    self.assertEqual(
        context_text,
        '[{"id": 0, "text": "a"}, {"id": 1, "text": "b"},'
        ' {"id": 2, "text": "c"}]',
    )

  def test_get_rank_by_id_randomize(self):
    candidates = ['a', 'b', 'c']
    rank_by_id = svq._get_rank_by_id(candidates, randomize=True)
    self.assertEqual(rank_by_id, {0: 1, 1: 2, 2: 0})

  def test_get_rank_by_id_no_randomize(self):
    candidates = ['a', 'b', 'c']
    rank_by_id = svq._get_rank_by_id(candidates, randomize=False)
    self.assertIsNone(rank_by_id)


@pytest.mark.whisper
@pytest.mark.optional
class SVQEnUsQueryRerankingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    _setup_testdata(self)
    self.enter_context(flagsaver.flagsaver((svq._RANDOMIZE_CANDIDATES, False)))

  def test_sub_tasks_property(self):
    task = svq.SVQEnUsQueryReranking()
    expected = [
        'query_reranking',
        'query_reranking:clean',
        'query_reranking:media_noise',
        'query_reranking:traffic_noise',
        'query_reranking:background_speech',
    ]
    self.assertEqual(task.sub_tasks, expected)

  def test_metadata(self):
    task = svq.SVQEnUsQueryReranking()
    self.assertEqual(task.metadata.name, 'SVQEnUsQueryReranking')
    self.assertEqual(task.metadata.main_score, 'MAP')
    self.assertEqual(task.metadata.type, 'QueryReranking')
    self.assertEqual(task.metadata.category, 'speech')

  def test_embeddings_dir(self):
    temp_dir = self.create_tempdir().full_path
    with flagsaver.flagsaver((task_lib.TASK_CACHE_BASEPATH, temp_dir)):
      task = svq.SVQEnUsQueryReranking()
      self.assertTrue(
          task.embeddings_dir.endswith(
              os.path.join('rerankings', 'svq_en_us_query_reranking')
          )
      )

  def test_svq_query_reranking_candidate_lists(self):
    task = svq.SVQEnUsQueryReranking()
    self.assertEqual(task.sub_tasks[0], 'query_reranking')
    candidate_lists = list(task.candidate_lists())
    self.assertLen(candidate_lists, 2)
    utt_id, candidates = candidate_lists[1]
    self.assertEqual(utt_id, 'utt_15041124811443622614')
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

  def test_candidate_lists_filters_by_locale(self):
    task = svq.SVQEnUsQueryReranking()
    candidate_lists = list(task.candidate_lists())
    sound_ids = [utt_id for utt_id, _ in candidate_lists]
    self.assertNotIn('utt_de_001', sound_ids)

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
    self.assertIsNone(example.rank_by_id)
    example = examples[1]
    self.assertEqual(example.sound_id, 'utt_15041124811443622614')
    self.assertLen(example.texts, 5)
    self.assertEqual(example.language, 'en_us')
    self.assertIsNone(example.rank_by_id)

  def test_examples_clean_sub_task(self):
    task = svq.SVQEnUsQueryReranking()
    examples = list(task.examples('query_reranking:clean'))
    self.assertLen(examples, 1)
    self.assertEqual(examples[0].sound_id, 'utt_15041124811443622614')

  def test_examples_background_speech_sub_task(self):
    task = svq.SVQEnUsQueryReranking()
    examples = list(task.examples('query_reranking:background_speech'))
    self.assertLen(examples, 1)
    self.assertEqual(examples[0].sound_id, 'utt_11697423627206642872')

  def test_examples_traffic_noise_sub_task_empty(self):
    task = svq.SVQEnUsQueryReranking()
    examples = list(task.examples('query_reranking:traffic_noise'))
    self.assertEmpty(examples)

  def test_examples_media_noise_sub_task_empty(self):
    task = svq.SVQEnUsQueryReranking()
    examples = list(task.examples('query_reranking:media_noise'))
    self.assertEmpty(examples)

  def test_examples_filters_by_locale(self):
    task = svq.SVQEnUsQueryReranking()
    examples = list(task.examples('query_reranking'))
    sound_ids = [ex.sound_id for ex in examples]
    self.assertNotIn('utt_de_001', sound_ids)

  def test_examples_randomized(self):
    with flagsaver.flagsaver((svq._RANDOMIZE_CANDIDATES, True)):
      task = svq.SVQEnUsQueryReranking()
      examples = list(task.examples('query_reranking'))
      self.assertLen(examples, 2)
      self.assertIsNotNone(examples[0].rank_by_id)
      self.assertIsInstance(examples[0].rank_by_id, dict)


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

  def test_locale_class_default_size_is_none(self):
    task_cls = getattr(svq, 'SVQSwQueryReranking')
    self.assertIsNone(task_cls.size)

  def test_compact_classes_exist(self):
    for locale, (suffix, _) in svq._SVQ_LOCALES.items():
      class_name = f'SVQ{suffix}QueryRerankingCompact'
      self.assertTrue(
          hasattr(svq, class_name),
          f'Missing compact class {class_name} for locale {locale}',
      )

  def test_compact_class_has_correct_size(self):
    task_cls = getattr(svq, 'SVQEnUsQueryRerankingCompact')
    self.assertEqual(task_cls.size, 'compact')

  def test_compact_class_has_correct_locale(self):
    task_cls = getattr(svq, 'SVQRuRuQueryRerankingCompact')
    self.assertEqual(task_cls.locale, 'ru_ru')

  def test_compact_class_inherits_from_base(self):
    task_cls = getattr(svq, 'SVQTeInQueryRerankingCompact')
    self.assertTrue(issubclass(task_cls, svq.SVQQueryReranking))

  def test_debug_classes_exist(self):
    for suffix in ('EnUs', 'FiFi'):
      class_name = f'SVQ{suffix}QueryRerankingDebug'
      self.assertTrue(
          hasattr(svq, class_name),
          f'Missing debug class {class_name}',
      )

  def test_debug_class_has_correct_size(self):
    task_cls = getattr(svq, 'SVQEnUsQueryRerankingDebug')
    self.assertEqual(task_cls.size, 'debug')

  def test_debug_only_en_us_and_fi_fi(self):
    """Debug classes should only exist for en-US and fi-FI."""
    for locale, (suffix, _) in svq._SVQ_LOCALES.items():
      class_name = f'SVQ{suffix}QueryRerankingDebug'
      if locale in ('en_us', 'fi_fi'):
        self.assertTrue(hasattr(svq, class_name))
      else:
        self.assertFalse(
            hasattr(svq, class_name),
            f'Unexpected debug class {class_name}',
        )

  def test_max_candidates_per_example(self):
    self.assertIsNone(svq.SVQEnUsQueryReranking.max_candidates_per_example)
    self.assertEqual(
        svq.SVQEnUsQueryRerankingCompact.max_candidates_per_example, 100
    )
    self.assertEqual(
        svq.SVQEnUsQueryRerankingDebug.max_candidates_per_example, 5
    )

  def test_embeddings_dir_with_size(self):
    temp_dir = self.create_tempdir().full_path
    with flagsaver.flagsaver((task_lib.TASK_CACHE_BASEPATH, temp_dir)):
      task_compact = svq.SVQEnUsQueryRerankingCompact()
      self.assertTrue(
          task_compact.embeddings_dir.endswith(
              os.path.join('rerankings', 'svq_en_us_query_reranking_compact')
          )
      )
      task_debug = svq.SVQEnUsQueryRerankingDebug()
      self.assertTrue(
          task_debug.embeddings_dir.endswith(
              os.path.join('rerankings', 'svq_en_us_query_reranking_debug')
          )
      )

  def test_metadata_type_is_query_reranking(self):
    task_cls = getattr(svq, 'SVQHiInQueryReranking')
    self.assertEqual(task_cls.metadata.type, 'QueryReranking')

  def test_metadata_main_score_is_map(self):
    task_cls = getattr(svq, 'SVQJaJpQueryReranking')
    self.assertEqual(task_cls.metadata.main_score, 'MAP')

  def test_total_class_count(self):
    """26 locales * 2 (default + compact) + 2 debug = 54 classes."""
    num_locales = len(svq._SVQ_LOCALES) * 2 + 2
    generated = [
        name
        for name in dir(svq)
        if name.startswith('SVQ')
        and name != 'SVQQueryReranking'
        and isinstance(getattr(svq, name), type)
        and issubclass(getattr(svq, name), svq.SVQQueryReranking)
    ]
    self.assertLen(generated, num_locales)

  def test_all_task_configurations(self):
    task_classes = [
        obj
        for _, obj in inspect.getmembers(svq, inspect.isclass)
        if issubclass(obj, svq.SVQQueryReranking)
        and obj is not svq.SVQQueryReranking
        and getattr(obj, 'locale', None) is not None
    ]
    self.assertNotEmpty(task_classes)

    for task_class in task_classes:
      with self.subTest(task_name=task_class.__name__):
        task = task_class()
        self.assertIsInstance(task.locale, str)
        self.assertNotEmpty(task.locale)
        self.assertEqual(task.metadata.name, task_class.__name__)
        self.assertEqual(task.metadata.type, 'QueryReranking')
        self.assertEqual(task.metadata.main_score, 'MAP')
        self.assertEqual(task.metadata.category, 'speech')


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

  def test_embeddings_dir_raises_without_locale(self):
    temp_dir = self.create_tempdir().full_path
    with flagsaver.flagsaver((task_lib.TASK_CACHE_BASEPATH, temp_dir)):
      task = svq.SVQQueryReranking()
      with self.assertRaises(AssertionError):
        _ = task.embeddings_dir


@pytest.mark.whisper
@pytest.mark.optional
class TaskDataFilteringTest(absltest.TestCase):
  """Tests for _task_data filtering."""

  def setUp(self):
    super().setUp()
    _setup_testdata(self)

  def test_task_data_loads_matching_rows(self):
    task = svq.SVQEnUsQueryReranking()
    df = task._task_data(
        'query_reranking',
        dtype={'locale': str, 'utt_id': str},
    )
    self.assertNotEmpty(df)
    self.assertTrue(df['rerankings/query'].all())

  def test_task_data_filters_by_task_boolean(self):
    temp_dir = self.create_tempdir().full_path
    with open(os.path.join(temp_dir, 'custom_task.jsonl'), 'w') as f:
      f.write(
          json.dumps({
              'utt_id': 'utt_1',
              'locale': 'en_us',
              'rerankings/query': True,
          })
          + '\n'
      )
      f.write(
          json.dumps({
              'utt_id': 'utt_2',
              'locale': 'en_us',
              'rerankings/query': False,
          })
          + '\n'
      )
      f.write(
          json.dumps({
              'utt_id': 'utt_3',
              'locale': 'en_us',
              'rerankings/query': True,
          })
          + '\n'
      )
    with open(os.path.join(temp_dir, 'utt_index.jsonl'), 'w') as f:
      for i, uid in enumerate(['utt_1', 'utt_2', 'utt_3']):
        f.write(
            json.dumps({'utt_id': uid, 'locale': 'en_us', 'index': i}) + '\n'
        )

    task = svq.SVQEnUsQueryReranking()
    with flagsaver.flagsaver((dataset._DATASET_BASEPATH, temp_dir)):
      task.__dict__.pop('svq_dataset', None)
      filtered_df = task._task_data('custom_task')
      self.assertEqual(filtered_df['utt_id'].tolist(), ['utt_1', 'utt_3'])


if __name__ == '__main__':
  absltest.main()
