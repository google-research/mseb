# Copyright 2025 The MSEB Authors.
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
from absl.testing import parameterized
from mseb import dataset
from mseb import types
from mseb.tasks.segmentations.salient_term import svq
import numpy as np


FLAGS = flags.FLAGS


class SVQSalientTermSegmentationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.testdata_dir = self.create_tempdir()

    self.mock_records = [
        {
            "utt_id": "en_us_001",
            "locale": "en_us",
            "index": "fake_index:0",
            "topk_salient_terms": ["weather", "boston"],
            "topk_salient_terms_timestamps": [[1.5, 2.0], [2.8, 3.5]],
        },
        {
            "utt_id": "en_us_002",
            "locale": "en_us",
            "index": "fake_index:1",
            "topk_salient_terms": ["music"],
            "topk_salient_terms_timestamps": [[4.0, 4.8]],
        },
        {
            "utt_id": "de_de_001",
            "locale": "de_de",
            "index": "fake_index:2",
            "topk_salient_terms": ["wetter"],
            "topk_salient_terms_timestamps": [[1.1, 2.2]],
        },
        {
            "utt_id": "en_us_003_no_gt",
            "locale": "en_us",
            "index": "fake_index:3",
            "topk_salient_terms": [],
            "topk_salient_terms_timestamps": [],
        },
    ]

    fake_jsonl_path = os.path.join(
        self.testdata_dir.full_path,
        "utt_index.jsonl"
    )
    with open(fake_jsonl_path, "w") as f:
      for record in self.mock_records:
        f.write(json.dumps(record) + "\n")

    self.enter_context(
        flagsaver.flagsaver(
            (dataset._DATASET_BASEPATH, self.testdata_dir.full_path)
        )
    )

    self.mock_get_sound = self.enter_context(
        mock.patch(
            "mseb.datasets.simple_voice_questions."
            "SimpleVoiceQuestionsDataset.get_sound"
        )
    )
    self.mock_get_sound.return_value = types.Sound(
        waveform=np.zeros(16000),
        context=types.SoundContextParams(
            id="mock_id",
            sample_rate=16000,
            length=16000
        ),
    )

  def test_svq_salient_term_segmentation_sounds(self):
    task = svq.SVQEnUsSalientTermSegmentation()
    sounds = list(task.sounds())

    self.assertLen(sounds, 3)
    self.mock_get_sound.assert_any_call({"utt_id": "en_us_001"})
    self.mock_get_sound.assert_any_call({"utt_id": "en_us_002"})
    self.mock_get_sound.assert_any_call({"utt_id": "en_us_003_no_gt"})

  def test_svq_salient_term_segmentation_examples(self):
    task = svq.SVQEnUsSalientTermSegmentation()
    examples = list(task.examples("salient_term"))
    self.assertLen(examples, 2)
    ex1 = examples[0]
    self.assertEqual(ex1.example_id, "en_us_001")
    self.assertLen(ex1.segments, 2)
    self.assertEqual(ex1.segments[0].embedding, "weather")
    self.assertEqual(ex1.segments[0].start_time, 1.5)
    self.assertEqual(ex1.segments[0].end_time, 2.0)
    self.assertEqual(ex1.segments[1].embedding, "boston")
    self.assertEqual(ex1.segments[1].start_time, 2.8)
    self.assertEqual(ex1.segments[1].end_time, 3.5)
    ex2 = examples[1]
    self.assertEqual(ex2.example_id, "en_us_002")
    self.assertLen(ex2.segments, 1)
    self.assertEqual(ex2.segments[0].embedding, "music")
    self.assertEqual(ex2.segments[0].start_time, 4.0)
    self.assertEqual(ex2.segments[0].end_time, 4.8)

  def test_svq_salient_term_segmentation_metadata(self):
    task = svq.SVQEnUsSalientTermSegmentation()
    self.assertEqual(task.sub_tasks[0], "salient_term")
    self.assertEqual(task.locale, "en_us")
    self.assertEqual(task.metadata.name, "SVQEnUsSalientTermSegmentation")
    self.assertEqual(task.metadata.main_score, "NDCG")

  def test_all_svq_task_configurations(self):
    task_classes = [
        obj for _, obj in inspect.getmembers(svq, inspect.isclass)
        if issubclass(obj, svq.SVQSalientTermSegmentation) and
        obj is not svq.SVQSalientTermSegmentation
    ]
    self.assertNotEmpty(task_classes)

    for task_class in task_classes:
      with self.subTest(task_name=task_class.__name__):
        task = task_class()
        self.assertIsInstance(task.locale, str)
        self.assertNotEmpty(task.locale)
        self.assertEqual(task.metadata.name, task_class.__name__)
        expected_eval_lang = task.locale.replace("_", "-")
        parts = expected_eval_lang.split("-")
        if len(parts) == 2:
          expected_eval_lang = f"{parts[0]}-{parts[1].upper()}"
        self.assertEqual(task.metadata.eval_langs, [expected_eval_lang])


if __name__ == "__main__":
  absltest.main()
