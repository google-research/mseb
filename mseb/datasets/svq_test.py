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

import os
import sys
from unittest import mock

from absl.testing import absltest
from mseb.datasets import svq
import numpy as np
import pandas as pd

# Mock the array_record dependency
sys.modules["array_record"] = mock.MagicMock()


class SVQTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.testdata_dir = self.create_tempdir()

    utt_index_data = [
        {
            "utt_id": "utt_1",
            "locale": "en_us",
            "speaker_id": "spk_a",
            "speaker_age": 38,
            "speaker_gender": "male",
            "text": "question one",
            "index": "audio/1:0",
        },
        {
            "utt_id": "utt_2",
            "locale": "ja_jp",
            "speaker_id": "spk_b",
            "speaker_age": 76,
            "speaker_gender": "female",
            "text": "question two",
            "index": "audio/1:1",
        },
        {
            "utt_id": "utt_3",
            "locale": "ko_kr",
            "speaker_id": "spk_c",
            "speaker_age": 76,
            "speaker_gender": "male",
            "text": (
                "1945\ub144 \uc77c\ubcf8 \ucc9c\ud669\uc740 "
                "\ub204\uad6c\uc778\uac00\uc694?"
            ),
            "index": "audio/1:2",
        }
    ]
    pd.DataFrame(utt_index_data).to_json(
        os.path.join(self.testdata_dir.full_path, "utt_index.jsonl"),
        orient="records", lines=True
    )

    task_data = [
        {
            "text": (
                "  \u0662\u0660\u0661\u0668 \u0645\u06cc\u06ba \u0628\u0646"
                "\u06af\u0644\u0627\u062f\u06cc\u0634 \u06a9\u0648\u0679"
                "\u06c1  \u0627\u0635\u0644\u0627\u062d\u0627\u062a \u0645"
                "\u06cc\u06ba \u06a9\u062a\u0646\u06d2 \u0644\u0648\u06af "
                "\u0632\u062e\u0645\u06cc \u06c1\u0648\u0626\u06d2?"
            ),
            "speaker_id": "speaker_8669843530557324340",
            "speaker_gender": "female",
            "speaker_age": 21,
            "environment": "background_speech",
            "locale": "ur_in",
            "passage_id": "-7615928064691233550",
            "utt_id": "utt_7260261290471960889",
            "page_title": "2018 Bangladesh quota reform movement",
            "passage_text": (
                "On 8 April 2018, hundreds of students began protests..."
            ),
            "span": "More than 160",
            "page_id": "2018 Bangladesh quota reform movement",
            "task": "qa_cross_lang"
        }
    ]
    pd.DataFrame(task_data).to_json(
        os.path.join(
            self.testdata_dir.full_path,
            "span_retrieval_cross_lang.jsonl"
        ),
        orient="records", lines=True, force_ascii=False
    )

    audio_dir = os.path.join(self.testdata_dir.full_path, "audio")
    os.makedirs(audio_dir)
    # Create an empty placeholder file
    with open(os.path.join(audio_dir, "1.array_record"), "wb") as f:
      f.write(b"\0" * (64 * 1024))

  @mock.patch("mseb.utils.download_from_hf")
  @mock.patch("mseb.datasets.svq._read_wav_bytes")
  @mock.patch("mseb.datasets.svq.array_record.ArrayRecordReader")
  def test_corpus_and_task_loading(
      self,
      mock_array_record_reader,
      mock_read_wav,
      _
  ):
    mock_reader_instance = mock.MagicMock()
    mock_reader_instance.read.return_value = [b"dummy_wav_bytes"]
    mock_array_record_reader.return_value = mock_reader_instance

    mock_read_wav.return_value = (np.zeros(16000, dtype=np.float32), 16000)
    # Initialize the dataset (no split needed here)
    dataset = svq.SVQDataset(base_path=self.testdata_dir.full_path)

    self.assertLen(dataset, 3)

    sound1 = dataset.get_sound_by_id("utt_1")
    self.assertEqual(sound1.context.id, "utt_1")
    self.assertEqual(sound1.context.speaker_id, "spk_a")
    self.assertEqual(sound1.context.speaker_age, 38)
    self.assertEqual(sound1.context.language, "en_us")

    task_df = dataset.get_task_data("span_retrieval_cross_lang")
    self.assertIsInstance(task_df, pd.DataFrame)
    self.assertLen(task_df, 1)

    task_record = task_df.iloc[0]
    self.assertEqual(task_record["utt_id"], "utt_7260261290471960889")
    self.assertEqual(task_record["span"], "More than 160")
    self.assertEqual(task_record["task"], "qa_cross_lang")
    self.assertIn("On 8 April 2018", task_record["passage_text"])

if __name__ == "__main__":
  absltest.main()

