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
from mseb import task as mseb_task
from mseb import types
from mseb.tasks.segmentation_selections.salient_term import svq
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
  return testdata_dir, mock_get_sound


class BaseSubTaskTest(absltest.TestCase):
  """Tests for the _base_sub_task helper."""

  def test_no_colon(self):
    self.assertEqual(svq._base_sub_task("salient_term"), "salient_term")

  def test_with_colon(self):
    self.assertEqual(svq._base_sub_task("salient_term:clean"), "salient_term")

  def test_multiple_colons(self):
    self.assertEqual(svq._base_sub_task("a:b:c"), "a")


class SVQSalientTermSegmentationSelectionBaseTest(absltest.TestCase):
  """Tests for the base class without a locale set."""

  def test_base_locale_is_none(self):
    self.assertIsNone(svq.SVQSalientTermSegmentationSelectionTask.locale)

  def test_sub_tasks(self):
    task = svq.SVQSalientTermSegmentationSelectionTask()
    expected = [
        "salient_term",
        "salient_term:clean",
        "salient_term:media_noise",
        "salient_term:traffic_noise",
        "salient_term:background_speech",
    ]
    self.assertEqual(task.sub_tasks, expected)

  def test_multimodal_inputs_raises_without_locale(self):
    task = svq.SVQSalientTermSegmentationSelectionTask()
    with self.assertRaisesRegex(ValueError, "locale"):
      list(task.multimodal_inputs())

  def test_examples_raises_without_locale(self):
    task = svq.SVQSalientTermSegmentationSelectionTask()
    with self.assertRaisesRegex(ValueError, "locale"):
      list(task.examples("salient_term_selection"))

  def test_salient_term_lists_raises_without_locale(self):
    task = svq.SVQSalientTermSegmentationSelectionTask()
    with self.assertRaisesRegex(ValueError, "locale"):
      list(task.salient_term_lists())


class SVQSalientTermSegmentationSelectionTest(parameterized.TestCase):
  """Tests for locale-specific salient term segmentation selection tasks."""

  def setUp(self):
    super().setUp()
    self.testdata_dir, self.mock_get_sound = _setup_testdata(self)

  # --- multimodal_inputs() tests ---

  def test_sounds(self):
    task = svq.SVQEnUsSalientTermSegmentationSelectionTask()
    sounds = list(task.multimodal_inputs())
    # en_us records: en_us_001, en_us_002, en_us_003_no_gt
    self.assertLen(sounds, 3)
    self.mock_get_sound.assert_any_call({"utt_id": "en_us_001"})
    self.mock_get_sound.assert_any_call({"utt_id": "en_us_002"})
    self.mock_get_sound.assert_any_call({"utt_id": "en_us_003_no_gt"})

  def test_sounds_filters_by_locale(self):
    task = svq.SVQEnUsSalientTermSegmentationSelectionTask()
    list(task.multimodal_inputs())
    # de_de_001 should NOT be included.
    for call in self.mock_get_sound.call_args_list:
      self.assertNotEqual(call, mock.call({"utt_id": "de_de_001"}))

  # --- examples() tests ---

  def test_examples(self):
    task = svq.SVQEnUsSalientTermSegmentationSelectionTask()
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

  def test_examples_filters_by_locale(self):
    task = svq.SVQEnUsSalientTermSegmentationSelectionTask()
    examples = list(task.examples("salient_term"))
    example_ids = [ex.example_id for ex in examples]
    self.assertNotIn("de_de_001", example_ids)

  def test_examples_skips_empty_terms(self):
    task = svq.SVQEnUsSalientTermSegmentationSelectionTask()
    examples = list(task.examples("salient_term"))
    example_ids = [ex.example_id for ex in examples]
    self.assertNotIn("en_us_003_no_gt", example_ids)

  def test_examples_clean_subtask(self):
    task = svq.SVQEnUsSalientTermSegmentationSelectionTask()
    examples = list(task.examples("salient_term:clean"))
    example_ids = [ex.example_id for ex in examples]
    # Only en_us_001 has environment=clean and non-empty terms.
    self.assertEqual(example_ids, ["en_us_001"])

  def test_examples_media_noise_subtask(self):
    task = svq.SVQEnUsSalientTermSegmentationSelectionTask()
    examples = list(task.examples("salient_term:media_noise"))
    example_ids = [ex.example_id for ex in examples]
    self.assertEqual(example_ids, ["en_us_002"])

  def test_examples_traffic_noise_subtask(self):
    task = svq.SVQEnUsSalientTermSegmentationSelectionTask()
    examples = list(task.examples("salient_term:traffic_noise"))
    # en_us_003_no_gt has environment=traffic_noise but empty terms.
    self.assertEmpty(examples)

  def test_examples_background_speech_subtask(self):
    task = svq.SVQEnUsSalientTermSegmentationSelectionTask()
    examples = list(task.examples("salient_term:background_speech"))
    self.assertEmpty(examples)

  def test_examples_skips_mismatched_lengths(self):
    mismatched_record = {
        "utt_id": "en_us_004_bad",
        "locale": "en_us",
        "index": "fake_index:4",
        "environment": "clean",
        "topk_salient_terms": ["dog", "cat"],
        "topk_salient_terms_timestamps": [[1.0, 2.0]],  # only 1 timestamp
        "candidate_salient_terms": ["dog"],
        "candidate_salient_terms_timestamps": [[1.0, 2.0]],
    }
    for filename in ("utt_index.jsonl", "salient_term.jsonl"):
      path = os.path.join(self.testdata_dir.full_path, filename)
      with open(path, "a") as f:
        f.write(json.dumps(mismatched_record) + "\n")

    task = svq.SVQEnUsSalientTermSegmentationSelectionTask()
    examples = list(task.examples("salient_term"))
    example_ids = [ex.example_id for ex in examples]
    self.assertNotIn("en_us_004_bad", example_ids)

  # --- salient_term_lists() tests ---

  def test_salient_term_lists_returns_tuples(self):
    task = svq.SVQEnUsSalientTermSegmentationSelectionTask()
    term_lists = list(task.salient_term_lists())
    # en_us_001 and en_us_002 have candidate terms; en_us_003_no_gt is empty.
    self.assertLen(term_lists, 2)
    for utt_id, segments in term_lists:
      self.assertIsInstance(utt_id, str)
      self.assertIsInstance(segments, list)

  def test_salient_term_lists_content(self):
    task = svq.SVQEnUsSalientTermSegmentationSelectionTask()
    term_lists = list(task.salient_term_lists())
    tl_001 = [tl for tl in term_lists if tl[0] == "en_us_001"][0]
    segments = tl_001[1]
    # en_us_001 has candidate_salient_terms: ["weather", "boston", "forecast"]
    self.assertLen(segments, 3)
    term_embeddings = {seg.embedding for seg in segments}
    self.assertEqual(term_embeddings, {"weather", "boston", "forecast"})

  def test_salient_term_lists_timestamps(self):
    task = svq.SVQEnUsSalientTermSegmentationSelectionTask()
    term_lists = list(task.salient_term_lists())
    tl_001 = [tl for tl in term_lists if tl[0] == "en_us_001"][0]
    segments = tl_001[1]
    # Check that timestamps are preserved from
    # candidate_salient_terms_timestamps.
    weather_seg = [s for s in segments if s.embedding == "weather"][0]
    self.assertAlmostEqual(weather_seg.start_time, 1.5)
    self.assertAlmostEqual(weather_seg.end_time, 2.0)

  def test_salient_term_lists_filters_by_locale(self):
    task = svq.SVQEnUsSalientTermSegmentationSelectionTask()
    term_lists = list(task.salient_term_lists())
    context_ids = [tl[0] for tl in term_lists]
    self.assertNotIn("de_de_001", context_ids)

  def test_salient_term_lists_skips_empty_terms(self):
    task = svq.SVQEnUsSalientTermSegmentationSelectionTask()
    term_lists = list(task.salient_term_lists())
    context_ids = [tl[0] for tl in term_lists]
    self.assertNotIn("en_us_003_no_gt", context_ids)

  # --- embeddings_dir tests ---

  def test_embeddings_dir_includes_locale(self):
    with flagsaver.flagsaver(
        (mseb_task.TASK_CACHE_BASEPATH, "/fake/cache/path")
    ):
      task = svq.SVQEnUsSalientTermSegmentationSelectionTask()
      self.assertIn("svq_en_us_salient_terms", task.embeddings_dir)

  # --- metadata tests ---

  def test_metadata(self):
    task = svq.SVQEnUsSalientTermSegmentationSelectionTask()
    self.assertEqual(
        task.metadata.name, "SVQEnUsSalientTermSegmentationSelectionTask"
    )
    self.assertEqual(task.metadata.main_score, "NDCG")
    self.assertEqual(task.metadata.type, "SalientTermSegmentation")
    self.assertEqual(task.metadata.eval_langs, ["en-US"])
    self.assertEqual(task.metadata.category, "speech")

  def test_task_data_filters_by_locale(self):
    task = svq.SVQEnUsSalientTermSegmentationSelectionTask()
    df = task._task_data("salient_term", dtype={"locale": str, "utt_id": str})
    for row in df.to_dict("records"):
      self.assertEqual(row["locale"], "en_us")

  def test_task_data_base_returns_all(self):
    task = svq.SVQSalientTermSegmentationSelectionTask()
    df = task._task_data("salient_term", dtype={"locale": str, "utt_id": str})
    locales = set(df["locale"].tolist())
    self.assertIn("en_us", locales)
    self.assertIn("de_de", locales)


class DynamicClassGenerationTest(parameterized.TestCase):
  """Tests for factory-generated locale-specific task classes."""

  def test_all_locale_classes_exist(self):
    for locale, (suffix, _) in svq._SVQ_LOCALES.items():
      class_name = f"SVQ{suffix}SalientTermSegmentationSelectionTask"
      self.assertTrue(
          hasattr(svq, class_name),
          f"Missing class {class_name} for locale {locale}",
      )

  def test_locale_class_inherits_from_base(self):
    task_cls = getattr(svq, "SVQKoKrSalientTermSegmentationSelectionTask")
    self.assertTrue(
        issubclass(task_cls, svq.SVQSalientTermSegmentationSelectionTask)
    )

  def test_total_class_count(self):
    num_locales = len(svq._SVQ_LOCALES)
    generated = [
        name
        for name in dir(svq)
        if name.startswith("SVQ")
        and name != "SVQSalientTermSegmentationSelectionTask"
        and isinstance(getattr(svq, name), type)
        and issubclass(
            getattr(svq, name), svq.SVQSalientTermSegmentationSelectionTask
        )
    ]
    self.assertLen(generated, num_locales)

  def test_metadata_scores(self):
    task_cls = getattr(svq, "SVQEnUsSalientTermSegmentationSelectionTask")
    expected_metric_names = {
        "mAP",
        "NDCG",
        "WordErrorRate",
        "TimestampsAccuracy",
        "EmbeddingsAccuracy",
        "TimestampsAndEmbeddingsAccuracy",
    }
    actual_metrics = {s.metric for s in task_cls.metadata.scores}
    self.assertEqual(actual_metrics, expected_metric_names)

  @parameterized.parameters(
      ("ar_eg", "ArEg", "ar-EG"),
      ("en_us", "EnUs", "en-US"),
      ("ja_jp", "JaJp", "ja-JP"),
      ("sw", "Sw", "sw"),
      ("ar_x_gulf", "ArXGulf", "ar-x-gulf"),
  )
  def test_locale_metadata(self, locale, suffix, eval_lang):
    class_name = f"SVQ{suffix}SalientTermSegmentationSelectionTask"
    task_cls = getattr(svq, class_name)
    task = task_cls()
    self.assertEqual(task.locale, locale)
    self.assertEqual(task.metadata.eval_langs, [eval_lang])
    self.assertEqual(task.metadata.name, class_name)
    self.assertEqual(task.metadata.type, "SalientTermSegmentation")
    self.assertEqual(task.metadata.main_score, "NDCG")


if __name__ == "__main__":
  absltest.main()
