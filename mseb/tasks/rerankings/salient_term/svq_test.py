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

import collections
import inspect
import json
import os
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from mseb import dataset
from mseb import task as task_lib
from mseb import types
from mseb.tasks.rerankings.salient_term import svq
import numpy as np

FLAGS = flags.FLAGS


class SVQSalientTermRerankingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.testdata_dir = self.create_tempdir()

    self.mock_records = [
        {
            'utt_id': 'en_us_001',
            'locale': 'en_us',
            'index': 'fake_index:0',
            'topk_salient_terms': ['weather', 'boston'],
            'candidate_salient_terms': ['weather', 'boston', 'forecast'],
            'environment': 'clean',
            'text': 'fake_transcript_001',
            'passage_id': 'passage_001',
        },
        {
            'utt_id': 'en_us_002',
            'locale': 'en_us',
            'index': 'fake_index:1',
            'topk_salient_terms': ['music'],
            'candidate_salient_terms': ['music', 'playlist'],
            'environment': 'media_noise',
            'text': 'fake_transcript_002',
            'passage_id': 'passage_002',
        },
        {
            'utt_id': 'de_de_001',
            'locale': 'de_de',
            'index': 'fake_index:2',
            'topk_salient_terms': ['wetter'],
            'candidate_salient_terms': ['wetter'],
            'environment': 'clean',
            'text': 'fake_transcript_003',
            'passage_id': 'passage_003',
        },
    ]

    for filename in ('utt_index.jsonl', 'salient_term.jsonl'):
      fake_jsonl_path = os.path.join(self.testdata_dir.full_path, filename)
      with open(fake_jsonl_path, 'w') as f:
        for record in self.mock_records:
          f.write(json.dumps(record) + '\n')

    audio_dir = os.path.join(self.testdata_dir.full_path, 'audio')
    os.makedirs(audio_dir, exist_ok=True)
    by_loc_env = collections.defaultdict(list)
    for record in self.mock_records:
      by_loc_env[(record['locale'], record['environment'])].append(record)

    for env in ('clean', 'media_noise', 'traffic_noise', 'background_speech'):
      if ('en_us', env) not in by_loc_env or not by_loc_env[('en_us', env)]:
        by_loc_env[('en_us', env)] = [{
            'locale': 'en_us',
            'utt_id': f'dummy_en_us_{env}',
            'environment': env,
            'rerankings/salient_term': False,
            'candidate_salient_terms': [],
            'topk_salient_terms': [],
            'text': '',
            'passage_id': 'no_passage_id',
        }]

    for (loc, env), records in by_loc_env.items():
      path = os.path.join(audio_dir, f'utts_{loc}_{env}.jsonl')
      with open(path, 'w') as f:
        for r in records:
          f.write(json.dumps(r) + '\n')

    self.enter_context(
        flagsaver.flagsaver(
            (dataset._DATASET_BASEPATH, self.testdata_dir.full_path)
        )
    )
    self.enter_context(
        flagsaver.flagsaver((svq._RANDOMIZE_CANDIDATE_SALIENT_TERMS, False))
    )

    self.mock_get_sound = self.enter_context(
        mock.patch(
            'mseb.datasets.simple_voice_questions.'
            'SimpleVoiceQuestionsDataset.get_sound'
        )
    )
    self.mock_get_sound.return_value = types.Sound(
        waveform=np.zeros(16000),
        context=types.SoundContextParams(
            id='mock_id', sample_rate=16000, length=16000
        ),
    )

  def test_sub_tasks_property(self):
    task = svq.SVQEnUsSalientTermReranking()
    expected = [
        'salient_term_reranking',
        'salient_term_reranking:clean',
        'salient_term_reranking:media_noise',
        'salient_term_reranking:traffic_noise',
        'salient_term_reranking:background_speech',
    ]
    self.assertEqual(task.sub_tasks, expected)

  def test_metadata(self):
    task = svq.SVQEnUsSalientTermReranking()
    self.assertEqual(task.metadata.name, 'SVQEnUsSalientTermReranking')
    self.assertEqual(task.metadata.main_score, 'NDCG')
    self.assertEqual(task.metadata.type, 'SalientTermReranking')

  def test_candidate_lists(self):
    task = svq.SVQEnUsSalientTermReranking()
    candidate_lists = list(task.candidate_lists())
    # Only en_us records: en_us_001 and en_us_002.
    self.assertLen(candidate_lists, 2)
    utt_id_001, candidates_001 = candidate_lists[0]
    self.assertEqual(utt_id_001, 'en_us_001')
    self.assertLen(candidates_001, 3)
    self.assertEqual(candidates_001[0].text, 'weather')
    self.assertEqual(candidates_001[0].context.id, 'weather')
    self.assertEqual(candidates_001[1].text, 'boston')
    self.assertEqual(candidates_001[2].text, 'forecast')
    utt_id_002, candidates_002 = candidate_lists[1]
    self.assertEqual(utt_id_002, 'en_us_002')
    self.assertLen(candidates_002, 2)
    self.assertEqual(candidates_002[0].text, 'music')
    self.assertEqual(candidates_002[1].text, 'playlist')

  def test_candidate_lists_filters_by_locale(self):
    task = svq.SVQEnUsSalientTermReranking()
    candidate_lists = list(task.candidate_lists())
    all_texts = []
    for _, cl in candidate_lists:
      all_texts.extend([c.text for c in cl])
    self.assertNotIn('wetter', all_texts)

  def test_multimodal_inputs(self):
    task = svq.SVQEnUsSalientTermReranking()
    sounds = list(task.multimodal_inputs())
    self.assertLen(sounds, 2)
    for sound in sounds:
      self.assertIsInstance(sound, types.SoundWithTitleAndContext)

  def test_multimodal_inputs_context_text(self):
    task = svq.SVQEnUsSalientTermReranking()
    sounds = list(task.multimodal_inputs())
    # With randomize=False, candidates are in original order.
    context_text = sounds[0].context_text
    self.assertIn('weather', context_text)
    self.assertIn('boston', context_text)
    self.assertIn('forecast', context_text)

  def test_examples(self):
    task = svq.SVQEnUsSalientTermReranking()
    examples = list(task.examples('salient_term_reranking'))
    self.assertLen(examples, 2)
    ex1 = examples[0]
    self.assertEqual(ex1.sound_id, 'en_us_001')
    self.assertLen(ex1.texts, 2)
    self.assertEqual(ex1.language, 'en_us')
    ex2 = examples[1]
    self.assertEqual(ex2.sound_id, 'en_us_002')
    self.assertLen(ex2.texts, 1)

  def test_examples_filters_by_locale(self):
    task = svq.SVQEnUsSalientTermReranking()
    examples = list(task.examples('salient_term_reranking'))
    sound_ids = [ex.sound_id for ex in examples]
    self.assertNotIn('de_de_001', sound_ids)

  def test_examples_for_clean_sub_task(self):
    task = svq.SVQEnUsSalientTermReranking()
    examples = list(task.examples('salient_term_reranking:clean'))
    sound_ids = [ex.sound_id for ex in examples]
    self.assertIn('en_us_001', sound_ids)
    self.assertNotIn('en_us_002', sound_ids)

  def test_examples_for_media_noise_sub_task(self):
    task = svq.SVQEnUsSalientTermReranking()
    examples = list(task.examples('salient_term_reranking:media_noise'))
    sound_ids = [ex.sound_id for ex in examples]
    self.assertIn('en_us_002', sound_ids)
    self.assertNotIn('en_us_001', sound_ids)

  def test_examples_for_traffic_noise_sub_task_empty(self):
    task = svq.SVQEnUsSalientTermReranking()
    examples = list(task.examples('salient_term_reranking:traffic_noise'))
    self.assertEmpty(examples)

  def test_examples_for_background_speech_sub_task_empty(self):
    task = svq.SVQEnUsSalientTermReranking()
    examples = list(task.examples('salient_term_reranking:background_speech'))
    self.assertEmpty(examples)

  def test_all_task_configurations(self):
    task_classes = [
        obj
        for _, obj in inspect.getmembers(svq, inspect.isclass)
        if issubclass(obj, svq.SVQSalientTermReranking)
        and obj is not svq.SVQSalientTermReranking
        and getattr(obj, 'locale', None) is not None
    ]
    self.assertNotEmpty(task_classes)

    for task_class in task_classes:
      with self.subTest(task_name=task_class.__name__):
        task = task_class()
        self.assertIsInstance(task.locale, str)
        self.assertNotEmpty(task.locale)
        self.assertEqual(task.metadata.name, task_class.__name__)
        self.assertEqual(task.metadata.type, 'SalientTermReranking')
        self.assertEqual(task.metadata.main_score, 'NDCG')
        self.assertEqual(task.metadata.category, 'speech')

  def test_seed_from_candidates(self):
    candidates = ['a', 'b', 'c']
    seed = svq._seed_from_candidates(candidates)
    self.assertEqual(
        seed,
        106066814613367738644872591937025180872126396142919988090859577352361712472268,
    )

  def test_get_context_text_no_randomize(self):
    candidates = ['weather', 'boston']
    context_text = svq._get_context_text(candidates, randomize=False)
    self.assertIn('weather', context_text)
    self.assertIn('boston', context_text)

  def test_rank_by_id_no_randomize(self):
    candidates = ['a', 'b', 'c']
    rank_by_id = svq._get_rank_by_id(candidates, randomize=False)
    self.assertIsNone(rank_by_id)

  def test_get_context_text_randomize(self):
    candidates = ['weather', 'boston']
    context_text = svq._get_context_text(candidates, randomize=True)
    self.assertEqual(
        context_text,
        '[{"id": 0, "text": "weather"}, {"id": 1, "text": "boston"}]',
    )

  def test_rank_by_id_randomize(self):
    candidates = ['a', 'b', 'c']
    rank_by_id = svq._get_rank_by_id(candidates, randomize=True)
    self.assertEqual(rank_by_id, {0: 1, 1: 2, 2: 0})


class DynamicClassGenerationTest(absltest.TestCase):
  """Tests for the factory-generated locale-specific task classes."""

  def test_all_locale_classes_exist(self):
    for locale, (suffix, _) in svq._SVQ_LOCALES.items():
      class_name = f'SVQ{suffix}SalientTermReranking'
      self.assertTrue(
          hasattr(svq, class_name),
          f'Missing class {class_name} for locale {locale}',
      )

  def test_locale_class_has_correct_locale(self):
    task_cls = getattr(svq, 'SVQEnUsSalientTermReranking')
    self.assertEqual(task_cls.locale, 'en_us')

  def test_locale_class_has_correct_metadata_name(self):
    task_cls = getattr(svq, 'SVQArEgSalientTermReranking')
    self.assertEqual(task_cls.metadata.name, 'SVQArEgSalientTermReranking')

  def test_locale_class_has_correct_eval_langs(self):
    task_cls = getattr(svq, 'SVQFiFiSalientTermReranking')
    self.assertEqual(task_cls.metadata.eval_langs, ['fi-FI'])

  def test_locale_class_inherits_from_base(self):
    task_cls = getattr(svq, 'SVQKoKrSalientTermReranking')
    self.assertTrue(issubclass(task_cls, svq.SVQSalientTermReranking))

  def test_locale_class_default_size_is_none(self):
    task_cls = getattr(svq, 'SVQSwSalientTermReranking')
    self.assertIsNone(task_cls.size)

  def test_compact_classes_exist(self):
    for locale, (suffix, _) in svq._SVQ_LOCALES.items():
      class_name = f'SVQ{suffix}SalientTermRerankingCompact'
      self.assertTrue(
          hasattr(svq, class_name),
          f'Missing compact class {class_name} for locale {locale}',
      )

  def test_compact_class_has_correct_size(self):
    task_cls = getattr(svq, 'SVQEnUsSalientTermRerankingCompact')
    self.assertEqual(task_cls.size, 'compact')

  def test_compact_class_has_correct_locale(self):
    task_cls = getattr(svq, 'SVQRuRuSalientTermRerankingCompact')
    self.assertEqual(task_cls.locale, 'ru_ru')

  def test_compact_class_inherits_from_base(self):
    task_cls = getattr(svq, 'SVQTeInSalientTermRerankingCompact')
    self.assertTrue(issubclass(task_cls, svq.SVQSalientTermReranking))

  def test_debug_classes_exist(self):
    for suffix in ('EnUs', 'FiFi'):
      class_name = f'SVQ{suffix}SalientTermRerankingDebug'
      self.assertTrue(
          hasattr(svq, class_name),
          f'Missing debug class {class_name}',
      )

  def test_debug_class_has_correct_size(self):
    task_cls = getattr(svq, 'SVQEnUsSalientTermRerankingDebug')
    self.assertEqual(task_cls.size, 'debug')

  def test_debug_only_en_us_and_fi_fi(self):
    """Debug classes should only exist for en-US and fi-FI."""
    for locale, (suffix, _) in svq._SVQ_LOCALES.items():
      class_name = f'SVQ{suffix}SalientTermRerankingDebug'
      if locale in ('en_us', 'fi_fi'):
        self.assertTrue(hasattr(svq, class_name))
      else:
        self.assertFalse(
            hasattr(svq, class_name),
            f'Unexpected debug class {class_name}',
        )

  def test_metadata_type_is_salient_term_reranking(self):
    task_cls = getattr(svq, 'SVQHiInSalientTermReranking')
    self.assertEqual(task_cls.metadata.type, 'SalientTermReranking')

  def test_metadata_main_score_is_ndcg(self):
    task_cls = getattr(svq, 'SVQJaJpSalientTermReranking')
    self.assertEqual(task_cls.metadata.main_score, 'NDCG')

  def test_total_class_count(self):
    """26 locales * 2 (default + compact) + 2 debug = 54 classes."""
    num_expected = len(svq._SVQ_LOCALES) * 2 + 2
    generated = [
        name
        for name in dir(svq)
        if name.startswith('SVQ')
        and name != 'SVQSalientTermReranking'
        and isinstance(getattr(svq, name), type)
        and issubclass(getattr(svq, name), svq.SVQSalientTermReranking)
    ]
    self.assertLen(generated, num_expected)

  def test_ur_pk_locale_exists(self):
    """Verify the new ur_pk locale is included."""
    self.assertIn('ur_pk', svq._SVQ_LOCALES)
    self.assertTrue(hasattr(svq, 'SVQUrPkSalientTermReranking'))
    self.assertTrue(hasattr(svq, 'SVQUrPkSalientTermRerankingCompact'))

  def test_embeddings_dir_with_size(self):
    temp_dir = self.create_tempdir().full_path
    with flagsaver.flagsaver((task_lib.TASK_CACHE_BASEPATH, temp_dir)):
      task_compact = svq.SVQEnUsSalientTermRerankingCompact()
      self.assertTrue(
          task_compact.embeddings_dir.endswith(
              os.path.join(
                  'rerankings', 'svq_en_us_salient_term_reranking_compact'
              )
          )
      )
      task_debug = svq.SVQEnUsSalientTermRerankingDebug()
      self.assertTrue(
          task_debug.embeddings_dir.endswith(
              os.path.join(
                  'rerankings', 'svq_en_us_salient_term_reranking_debug'
              )
          )
      )


class BaseClassTest(absltest.TestCase):
  """Tests for SVQSalientTermReranking base class attributes."""

  def test_base_locale_is_none(self):
    self.assertIsNone(svq.SVQSalientTermReranking.locale)

  def test_base_size_is_none(self):
    self.assertIsNone(svq.SVQSalientTermReranking.size)

  def test_sub_tasks(self):
    task = svq.SVQSalientTermReranking()
    self.assertIn('salient_term_reranking', task.sub_tasks)
    self.assertIn('salient_term_reranking:clean', task.sub_tasks)
    self.assertIn('salient_term_reranking:media_noise', task.sub_tasks)
    self.assertIn('salient_term_reranking:traffic_noise', task.sub_tasks)
    self.assertIn('salient_term_reranking:background_speech', task.sub_tasks)

  def test_embeddings_dir_raises_without_locale(self):
    temp_dir = self.create_tempdir().full_path
    with flagsaver.flagsaver((task_lib.TASK_CACHE_BASEPATH, temp_dir)):
      task = svq.SVQSalientTermReranking()
      with self.assertRaises(AssertionError):
        _ = task.embeddings_dir


class TaskDataFilteringTest(absltest.TestCase):
  """Tests for _task_data filtering."""

  def test_task_data_filters_by_task_boolean(self):
    temp_dir = self.create_tempdir().full_path
    with open(os.path.join(temp_dir, 'custom_task.jsonl'), 'w') as f:
      f.write(
          json.dumps({
              'utt_id': 'utt_1',
              'locale': 'en_us',
          })
          + '\n'
      )
      f.write(
          json.dumps({
              'utt_id': 'utt_2',
              'locale': 'en_us',
          })
          + '\n'
      )
      f.write(
          json.dumps({
              'utt_id': 'utt_3',
              'locale': 'en_us',
          })
          + '\n'
      )
    with open(os.path.join(temp_dir, 'utt_index.jsonl'), 'w') as f:
      for i, uid in enumerate(['utt_1', 'utt_2', 'utt_3']):
        f.write(
            json.dumps({'utt_id': uid, 'locale': 'en_us', 'index': i}) + '\n'
        )

    task = svq.SVQEnUsSalientTermReranking()
    with flagsaver.flagsaver((dataset._DATASET_BASEPATH, temp_dir)):
      task.__dict__.pop('svq_dataset', None)
      filtered_df = task._task_data('custom_task')
      self.assertEqual(
          filtered_df['utt_id'].tolist(), ['utt_1', 'utt_2', 'utt_3']
      )


if __name__ == '__main__':
  absltest.main()
