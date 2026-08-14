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

import json
import os
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from mseb import dataset
from mseb import types
from mseb.tasks.segmentations.salient_term import svq
import numpy as np

FLAGS = flags.FLAGS


def _setup_testdata(test_case):
  """Sets up a temp dir with SVQ testdata and configures the dataset flag."""
  testdata_dir = test_case.create_tempdir()

  mock_records = [
      {
          "utt_id": "en_us_001",
          "locale": "en_us",
          "index": "fake_index:0",
          "environment": "clean",
          "topk_salient_terms": ["weather", "boston"],
          "topk_salient_terms_timestamps": [[1.5, 2.0], [2.8, 3.5]],
          "candidate_salient_terms": ["weather", "boston", "forecast"],
          "candidate_salient_terms_timestamps": [
              [1.5, 2.0],
              [2.8, 3.5],
              [4.0, 4.5],
          ],
      },
      {
          "utt_id": "en_us_002",
          "locale": "en_us",
          "index": "fake_index:1",
          "environment": "media_noise",
          "topk_salient_terms": ["music"],
          "topk_salient_terms_timestamps": [[4.0, 4.8]],
          "candidate_salient_terms": ["music", "playlist"],
          "candidate_salient_terms_timestamps": [[4.0, 4.8], [5.0, 5.5]],
      },
      {
          "utt_id": "de_de_001",
          "locale": "de_de",
          "index": "fake_index:2",
          "environment": "clean",
          "topk_salient_terms": ["wetter"],
          "topk_salient_terms_timestamps": [[1.1, 2.2]],
          "candidate_salient_terms": ["wetter"],
          "candidate_salient_terms_timestamps": [[1.1, 2.2]],
      },
      {
          "utt_id": "en_us_003_no_gt",
          "locale": "en_us",
          "index": "fake_index:3",
          "environment": "traffic_noise",
          "topk_salient_terms": [],
          "topk_salient_terms_timestamps": [],
          "candidate_salient_terms": [],
          "candidate_salient_terms_timestamps": [],
      },
  ]

  for filename in ("utt_index.jsonl", "salient_term.jsonl"):
    fake_jsonl_path = os.path.join(testdata_dir.full_path, filename)
    with open(fake_jsonl_path, "w") as f:
      for record in mock_records:
        f.write(json.dumps(record) + "\n")

  test_case.enter_context(
      flagsaver.flagsaver((dataset._DATASET_BASEPATH, testdata_dir.full_path))
  )

  mock_get_sound = test_case.enter_context(
      mock.patch(
          "mseb.datasets.simple_voice_questions."
          "SimpleVoiceQuestionsDataset.get_sound"
      )
  )
  mock_get_sound.return_value = types.Sound(
      waveform=np.zeros(16000),
      context=types.SoundContextParams(
          id="mock_id", sample_rate=16000, length=16000
      ),
  )
  return mock_get_sound


class BaseSubTaskTest(absltest.TestCase):
  """Tests for the _base_sub_task helper."""

  def test_no_colon(self):
    self.assertEqual(svq._base_sub_task("salient_term"), "salient_term")

  def test_with_colon(self):
    self.assertEqual(svq._base_sub_task("salient_term:clean"), "salient_term")

  def test_multiple_colons(self):
    self.assertEqual(svq._base_sub_task("a:b:c"), "a")


class SVQSalientTermSegmentationBaseTest(absltest.TestCase):
  """Tests for the SVQSalientTermSegmentation base class."""

  def test_base_locale_is_none(self):
    self.assertIsNone(svq.SVQSalientTermSegmentation.locale)

  def test_sub_tasks(self):
    task = svq.SVQSalientTermSegmentation()
    expected = [
        "salient_term",
        "salient_term:clean",
        "salient_term:media_noise",
        "salient_term:traffic_noise",
        "salient_term:background_speech",
    ]
    self.assertEqual(task.sub_tasks, expected)

  def test_multimodal_inputs_raises_without_locale(self):
    task = svq.SVQSalientTermSegmentation()
    with self.assertRaises(ValueError):
      list(task.multimodal_inputs())

  def test_examples_raises_without_locale(self):
    task = svq.SVQSalientTermSegmentation()
    with self.assertRaises(ValueError):
      list(task.examples("salient_term"))


class SVQSalientTermSegmentationTest(parameterized.TestCase):
  """Tests for locale-specific segmentation tasks."""

  def setUp(self):
    super().setUp()
    self.mock_get_sound = _setup_testdata(self)

  def test_sounds(self):
    task = svq.SVQEnUsSalientTermSegmentation()
    sounds = list(task.multimodal_inputs())
    # en_us records: en_us_001, en_us_002, en_us_003_no_gt
    self.assertLen(sounds, 3)
    self.mock_get_sound.assert_any_call({"utt_id": "en_us_001"})
    self.mock_get_sound.assert_any_call({"utt_id": "en_us_002"})
    self.mock_get_sound.assert_any_call({"utt_id": "en_us_003_no_gt"})

  def test_sounds_filters_by_locale(self):
    task = svq.SVQEnUsSalientTermSegmentation()
    list(task.multimodal_inputs())
    # de_de_001 should NOT be included.
    for call in self.mock_get_sound.call_args_list:
      self.assertNotEqual(call, mock.call({"utt_id": "de_de_001"}))

  def test_examples_salient_term(self):
    task = svq.SVQEnUsSalientTermSegmentation()
    examples = list(task.examples("salient_term"))
    # en_us_003_no_gt has empty terms, so only 2 examples.
    self.assertLen(examples, 2)
    ex1 = examples[0]
    self.assertEqual(ex1.example_id, "en_us_001")
    self.assertLen(ex1.segments, 2)
    self.assertEqual(ex1.segments[0].embedding, "weather")
    self.assertAlmostEqual(ex1.segments[0].start_time, 1.5)
    self.assertAlmostEqual(ex1.segments[0].end_time, 2.0)
    self.assertEqual(ex1.segments[1].embedding, "boston")
    self.assertAlmostEqual(ex1.segments[1].start_time, 2.8)
    self.assertAlmostEqual(ex1.segments[1].end_time, 3.5)
    ex2 = examples[1]
    self.assertEqual(ex2.example_id, "en_us_002")
    self.assertLen(ex2.segments, 1)
    self.assertEqual(ex2.segments[0].embedding, "music")
    self.assertAlmostEqual(ex2.segments[0].start_time, 4.0)
    self.assertAlmostEqual(ex2.segments[0].end_time, 4.8)

  def test_examples_filters_by_locale(self):
    task = svq.SVQEnUsSalientTermSegmentation()
    examples = list(task.examples("salient_term"))
    example_ids = [ex.example_id for ex in examples]
    self.assertNotIn("de_de_001", example_ids)

  def test_examples_skips_empty_terms(self):
    task = svq.SVQEnUsSalientTermSegmentation()
    examples = list(task.examples("salient_term"))
    example_ids = [ex.example_id for ex in examples]
    # en_us_003_no_gt has empty terms list.
    self.assertNotIn("en_us_003_no_gt", example_ids)

  def test_examples_clean_subtask(self):
    task = svq.SVQEnUsSalientTermSegmentation()
    examples = list(task.examples("salient_term:clean"))
    example_ids = [ex.example_id for ex in examples]
    # Only en_us_001 has environment=clean.
    self.assertEqual(example_ids, ["en_us_001"])

  def test_examples_media_noise_subtask(self):
    task = svq.SVQEnUsSalientTermSegmentation()
    examples = list(task.examples("salient_term:media_noise"))
    example_ids = [ex.example_id for ex in examples]
    # Only en_us_002 has environment=media_noise.
    self.assertEqual(example_ids, ["en_us_002"])

  def test_examples_traffic_noise_subtask(self):
    task = svq.SVQEnUsSalientTermSegmentation()
    examples = list(task.examples("salient_term:traffic_noise"))
    example_ids = [ex.example_id for ex in examples]
    # en_us_003_no_gt has environment=traffic_noise but empty terms.
    self.assertEmpty(example_ids)

  def test_metadata(self):
    task = svq.SVQEnUsSalientTermSegmentation()
    self.assertEqual(task.sub_tasks[0], "salient_term")
    self.assertEqual(task.locale, "en_us")
    self.assertEqual(task.metadata.name, "SVQEnUsSalientTermSegmentation")
    self.assertEqual(task.metadata.main_score, "NDCG")
    self.assertEqual(task.metadata.type, "SalientTermSegmentation")
    self.assertEqual(task.metadata.eval_langs, ["en-US"])

  def test_task_data_filters_by_locale(self):
    task = svq.SVQEnUsSalientTermSegmentation()
    df = task._task_data("salient_term", dtype={"locale": str, "utt_id": str})
    for row in df.to_dict("records"):
      self.assertEqual(row["locale"], "en_us")

  def test_task_data_base_returns_all(self):
    task = svq.SVQSalientTermSegmentation()
    df = task._task_data("salient_term", dtype={"locale": str, "utt_id": str})
    locales = set(df["locale"].tolist())
    self.assertIn("en_us", locales)
    self.assertIn("de_de", locales)


class DynamicClassGenerationTest(parameterized.TestCase):
  """Tests for factory-generated locale-specific task classes."""

  def test_all_locale_classes_exist(self):
    for locale, (suffix, _) in svq._SVQ_LOCALES.items():
      class_name = f"SVQ{suffix}SalientTermSegmentation"
      self.assertTrue(
          hasattr(svq, class_name),
          f"Missing class {class_name} for locale {locale}",
      )

  def test_locale_class_has_correct_locale(self):
    task_cls = getattr(svq, "SVQEnUsSalientTermSegmentation")
    self.assertEqual(task_cls.locale, "en_us")

  def test_locale_class_has_correct_metadata_name(self):
    task_cls = getattr(svq, "SVQArEgSalientTermSegmentation")
    self.assertEqual(task_cls.metadata.name, "SVQArEgSalientTermSegmentation")

  def test_locale_class_has_correct_eval_langs(self):
    task_cls = getattr(svq, "SVQFiFiSalientTermSegmentation")
    self.assertEqual(task_cls.metadata.eval_langs, ["fi-FI"])

  def test_locale_class_inherits_from_base(self):
    task_cls = getattr(svq, "SVQKoKrSalientTermSegmentation")
    self.assertTrue(issubclass(task_cls, svq.SVQSalientTermSegmentation))

  def test_metadata_type_is_salient_term_segmentation(self):
    task_cls = getattr(svq, "SVQHiInSalientTermSegmentation")
    self.assertEqual(task_cls.metadata.type, "SalientTermSegmentation")

  def test_metadata_main_score_is_ndcg(self):
    task_cls = getattr(svq, "SVQJaJpSalientTermSegmentation")
    self.assertEqual(task_cls.metadata.main_score, "NDCG")

  def test_total_class_count(self):
    """26 locales = 26 classes."""
    num_locales = len(svq._SVQ_LOCALES)
    generated = [
        name
        for name in dir(svq)
        if name.startswith("SVQ")
        and name != "SVQSalientTermSegmentation"
        and isinstance(getattr(svq, name), type)
        and issubclass(getattr(svq, name), svq.SVQSalientTermSegmentation)
    ]
    self.assertLen(generated, num_locales)

  @parameterized.parameters(
      ("ar_eg", "ArEg", "ar-EG"),
      ("en_us", "EnUs", "en-US"),
      ("ja_jp", "JaJp", "ja-JP"),
      ("sw", "Sw", "sw"),
      ("ar_x_gulf", "ArXGulf", "ar-x-gulf"),
  )
  def test_locale_metadata(self, locale, suffix, eval_lang):
    class_name = f"SVQ{suffix}SalientTermSegmentation"
    task_cls = getattr(svq, class_name)
    task = task_cls()
    self.assertEqual(task.locale, locale)
    self.assertEqual(task.metadata.eval_langs, [eval_lang])
    self.assertEqual(task.metadata.name, class_name)
    self.assertEqual(task.metadata.type, "SalientTermSegmentation")
    self.assertEqual(task.metadata.main_score, "NDCG")


if __name__ == "__main__":
  absltest.main()
