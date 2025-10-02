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
import pathlib
import sys
from unittest import mock

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import test_pipeline
from apache_beam.testing import util as beam_testing_util
from mseb.datasets import simple_voice_questions as svq
import pandas as pd


# Mock the array_record dependency
sys.modules["array_record"] = mock.MagicMock()


class SimpleVoiceQuestionsTest(absltest.TestCase):

  def test_corpus_and_task_loading(self):
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, "testdata"
    )
    dataset = svq.SimpleVoiceQuestionsDataset(base_path=testdata_path)

    self.assertLen(dataset, 4)

    sound1 = dataset.get_sound_by_id("utt_14868079180393484423")
    self.assertEqual(sound1.context.id, "utt_14868079180393484423")
    self.assertEqual(sound1.context.speaker_id, "speaker_14224269439222776736")
    self.assertEqual(sound1.context.speaker_age, 40)
    self.assertEqual(sound1.context.language, "en_us")

    task_df = dataset.get_task_data("test_task")
    self.assertIsInstance(task_df, pd.DataFrame)
    self.assertLen(task_df, 1)

    task_record = task_df.iloc[0]
    self.assertEqual(task_record["utt_id"], "utt_14868079180393484423")
    self.assertEqual(task_record["task"], "retrieval_in_lang")
    self.assertIn("1480 and 1481", task_record["passage_text"])

  def test_get_task_data_for_text_index(self):
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, "testdata"
    )
    dataset = svq.SimpleVoiceQuestionsDataset(base_path=testdata_path)
    examples = list(
        dataset.get_task_data("passage_retrieval_in_lang_index").itertuples()
    )
    self.assertLen(examples, 3)
    example = examples[0]
    self.assertEqual(example.id, "english-5046980532842052129-41")
    self.assertEqual(example.title, "American football")
    self.assertTrue(example.context.startswith("Football games last for a"))

    example = examples[1]
    self.assertEqual(example.id, "english-688112198501873749-3")
    self.assertEqual(
        example.title, "Cannabis classification in the United Kingdom"
    )
    self.assertTrue(example.context.startswith("Early in January 2006 Charles"))

    example = examples[2]
    self.assertEqual(example.id, "english-1064747448949054415-7")
    self.assertEqual(example.title, "Little Albert experiment")
    self.assertTrue(example.context.startswith("Albert was about one year old"))


class SimpleVoiceQuestionsBeamTest(absltest.TestCase):

  def test_get_task_data_beam(self):
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, "testdata"
    )
    dataset = svq.SimpleVoiceQuestionsDataset(base_path=testdata_path)
    with test_pipeline.TestPipeline() as p:
      examples = p | dataset.get_task_data_beam("test_task")
      expected_output = [{
          "text": "When did the Ottoman empire conquer Italy?",
          "utt_id": "utt_14868079180393484423",
          "waveform": (88320,),
      }]
      beam_testing_util.assert_that(
          examples
          | beam.Map(
              lambda x: {
                  "text": x["text"],
                  "utt_id": x["utt_id"],
                  "waveform": x["sound"].waveform.shape,
              }
          ),
          beam_testing_util.equal_to(expected_output),
      )

  def test_get_task_sounds_beam(self):
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, "testdata"
    )
    dataset = svq.SimpleVoiceQuestionsDataset(base_path=testdata_path)
    with test_pipeline.TestPipeline() as p:
      examples = p | dataset.get_task_sounds_beam("test_task")
      expected_output = [{
          "waveform": (88320,),
      }]
      beam_testing_util.assert_that(
          examples
          | beam.Map(
              lambda x: {"waveform": x.waveform.shape},
          ),
          beam_testing_util.equal_to(expected_output),
      )


if __name__ == "__main__":
  absltest.main()
