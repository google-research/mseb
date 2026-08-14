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

import inspect
import json
import os
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from mseb import dataset
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
        },
        {
            'utt_id': 'en_us_002',
            'locale': 'en_us',
            'index': 'fake_index:1',
            'topk_salient_terms': ['music'],
            'candidate_salient_terms': ['music', 'playlist'],
            'environment': 'media_noise',
            'text': 'fake_transcript_002',
        },
        {
            'utt_id': 'de_de_001',
            'locale': 'de_de',
            'index': 'fake_index:2',
            'topk_salient_terms': ['wetter'],
            'candidate_salient_terms': ['wetter'],
            'environment': 'clean',
            'text': 'fake_transcript_003',
        },
    ]

    for filename in ('utt_index.jsonl', 'salient_term.jsonl'):
      fake_jsonl_path = os.path.join(self.testdata_dir.full_path, filename)
      with open(fake_jsonl_path, 'w') as f:
        for record in self.mock_records:
          f.write(json.dumps(record) + '\n')

    self.enter_context(
        flagsaver.flagsaver(
            (dataset._DATASET_BASEPATH, self.testdata_dir.full_path)
        )
    )
    self.enter_context(
        flagsaver.flagsaver(
            (svq._RANDOMIZE_CANDIDATE_SALIENT_TERMS, False)
        )
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
    examples = list(
        task.examples('salient_term_reranking:background_speech')
    )
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


if __name__ == '__main__':
  absltest.main()
