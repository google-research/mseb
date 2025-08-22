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
from unittest import mock

from absl.testing import absltest
from mseb.datasets import speech_massive
import numpy as np
import pandas as pd


class SpeechMassiveTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.testdata_dir = self.create_tempdir()

    lang_dir = os.path.join(self.testdata_dir.full_path, "en-US")
    os.makedirs(lang_dir)

    mock_data = [
        {
            "id": "1",
            "path": "en-US/test_1.wav",
            "utt": "hello world",
            "speaker_id": "spk_1",
            "speaker_age": "35",
            "speaker_sex": "Male"
        },
        {
            "id": "2",
            "path": "en-US/test_2.wav",
            "utt": "goodbye world",
            "speaker_id": "spk_2",
            "speaker_age": 42,
            "speaker_sex": "Female"
        },
        {
            "id": "3",
            "path": "en-US/test_3.wav",
            "utt": "another test",
            "speaker_id": "spk_3",
            "speaker_age": None,
            "speaker_sex": "Unidentified"
        }
    ]
    pd.DataFrame(mock_data).to_json(
        os.path.join(lang_dir, "test.jsonl"),
        orient="records", lines=True
    )

  @mock.patch("mseb.utils.download_from_hf")
  @mock.patch("mseb.utils.read_audio")
  def test_loading_and_parsing(self, mock_read_audio, _):
    mock_read_audio.return_value = (np.zeros(16000, dtype=np.float32), 16000)

    dataset = speech_massive.SpeechMassiveDataset(
        base_path=self.testdata_dir.full_path,
        language="en-US",
        split="test"
    )

    self.assertLen(dataset, 3)

    sound1 = dataset[0]
    self.assertEqual(sound1.context.sound_id, "1")
    self.assertEqual(sound1.context.text, "hello world")
    self.assertEqual(sound1.context.speaker_id, "spk_1")
    self.assertEqual(sound1.context.speaker_age, 35)
    self.assertEqual(sound1.context.speaker_gender, "Male")
    self.assertAlmostEqual(sound1.context.waveform_end_second, 1.0)

    sound2 = dataset[1]
    self.assertEqual(sound2.context.speaker_id, "spk_2")
    self.assertEqual(sound2.context.speaker_age, 42)
    self.assertEqual(sound2.context.speaker_gender, "Female")

    sound3 = dataset[2]
    self.assertEqual(sound3.context.speaker_id, "spk_3")
    self.assertIsNone(sound3.context.speaker_age)
    self.assertEqual(sound3.context.speaker_gender, "Unidentified")


if __name__ == "__main__":
  absltest.main()

