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

    lang_dir = os.path.join(self.testdata_dir.full_path, "de-DE")
    os.makedirs(lang_dir)

    mock_data = dict(
        id=["2205"],
        locale=["de-DE"],
        partition=["test"],
        scenario=[10],
        scenario_str=["audio"],
        intent_idx=[46],
        intent_str=["audio_volume_mute"],
        utt=["stille für zwei stunden"],
        annot_utt=["stille für [time : zwei stunden]"],
        worker_id=["8"],
        slot_method=[{
            "slot": np.array(["time"], dtype=object),
            "method": np.array(["translation"], dtype=object),
        }],
        judgments=[{
            "worker_id": np.array(["27", "28", "8"], dtype=object),
            "intent_score": np.array([1, 1, 1], dtype=np.int8),
            "slots_score": np.array([1, 1, 1], dtype=np.int8),
            "grammar_score": np.array([3, 4, 4], dtype=np.int8),
            "spelling_score": np.array([2, 2, 2], dtype=np.int8),
            "language_identification": np.array(
                ["target", "target", "target"], dtype=object
            ),
        }],
        tokens=[np.array(["stille", "für", "zwei", "stunden"], dtype=object)],
        labels=[np.array(["Other", "Other", "time", "time"], dtype=object)],
        audio=[{
            "bytes": (
                b"RIFF$\x00\x00\x00WAVEfmt"
                b" \x10\x00\x00\x00\x01\x00\x01\x00@\x1f\x00\x00\x80>\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
            ),
            "path": "c15b5445ba46918a8d678e7b59b80aa6.wav",
        }],
        path=["test/c15b5445ba46918a8d678e7b59b80aa6.wav"],
        is_transcript_reported=[False],
        is_validated=[True],
        speaker_id=["5f32d5f107d49607c3f6cf7a"],
        speaker_sex=["Female"],
        speaker_age=["40"],
        speaker_ethnicity_simple=["White"],
        speaker_country_of_birth=["Germany"],
        speaker_country_of_residence=["Germany"],
        speaker_nationality=["Germany"],
        speaker_first_language=["German"],
    )
    pd.DataFrame(mock_data).to_parquet(
        os.path.join(lang_dir, "test-00000-of-00001.parquet")
    )

  @mock.patch("mseb.utils.download_from_hf")
  def test_loading_and_parsing(self, _):
    dataset = speech_massive.SpeechMassiveDataset(
        base_path=self.testdata_dir.full_path, language="de-DE", split="test"
    )

    self.assertLen(dataset, 1)

    task_df = dataset.get_task_data()
    self.assertIsInstance(task_df, pd.DataFrame)
    self.assertLen(task_df, 1)

    record = task_df.iloc[0]
    self.assertEqual(record.locale, "de-DE")
    self.assertEqual(record.partition, "test")
    self.assertEqual(record.speaker_id, "5f32d5f107d49607c3f6cf7a")
    self.assertEqual(record.speaker_sex, "Female")
    self.assertEqual(record.speaker_age, "40")
    self.assertEqual(record.intent_str, "audio_volume_mute")
    self.assertEqual(record.intent_idx, 46)

  @mock.patch("mseb.utils.download_from_hf")
  def test_get_sound(self, _):
    dataset = speech_massive.SpeechMassiveDataset(
        base_path=self.testdata_dir.full_path, language="de-DE", split="test"
    )
    record = dataset.get_task_data().iloc[0]
    sound = dataset.get_sound(record)
    self.assertEqual(
        sound.context.id, "test/c15b5445ba46918a8d678e7b59b80aa6.wav"
    )
    self.assertEqual(sound.context.text, "stille für zwei stunden")
    self.assertEqual(sound.context.speaker_id, "5f32d5f107d49607c3f6cf7a")
    self.assertEqual(sound.context.speaker_age, 40)
    self.assertEqual(sound.context.speaker_gender, "Female")
    self.assertAlmostEqual(sound.context.waveform_end_second, 0.0)


if __name__ == "__main__":
  absltest.main()
