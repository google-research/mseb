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

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from mseb import dataset
from mseb.encoders import prompt as prompt_lib
from mseb.encoders import prompt_registry
import pandas as pd


class PromptRegistryTest(parameterized.TestCase):

  @parameterized.parameters(
      prompt_registry.reasoning,
      prompt_registry.intent_classification,
      prompt_registry.speaker_gender_classification,
      prompt_registry.segmentation,
      prompt_registry.segmentation_from_alignment,
  )
  def test_load_encoder(self, meta):
    prompt = meta.load()
    self.assertIsNotNone(prompt)
    self.assertIsInstance(prompt.GetPromptTemplate(), str)
    self.assertNotEmpty(prompt.GetPromptTemplate())

  def test_sound_classification_loads_from_vocabulary_csv(self):
    temp_dir = self.create_tempdir()
    labels_dir = os.path.join(temp_dir.full_path, "labels")
    os.makedirs(labels_dir)
    with open(os.path.join(labels_dir, "vocabulary.csv"), "w") as f:
      f.write("0,Alarm,/m/07pp_mv\n1,Bark,/m/05tny_\n")

    with flagsaver.flagsaver((dataset._DATASET_BASEPATH, temp_dir.full_path)):
      prompt = prompt_registry.sound_classification.load()
    self.assertIsInstance(prompt, prompt_lib.SoundClassificationPrompt)
    self.assertEqual(prompt.ProcessResponse('{"answer": "Bark"}'), "Bark")

  def test_birdset_prompt_loads_configuration_labels(self):
    temp_dir = self.create_tempdir()
    with open(os.path.join(temp_dir.full_path, "class_lists.json"), "w") as f:
      json.dump({"nbp_list": ["amerob", "barswa"]}, f)
    with open(
        os.path.join(temp_dir.full_path, "config_to_class_list.json"), "w"
    ) as f:
      json.dump({"NBP": "nbp_list"}, f)
    pd.DataFrame({"ebird_code_multilabel": [[1]]}).to_parquet(
        os.path.join(temp_dir.full_path, "birdset_NBP_test_5s.parquet")
    )

    with flagsaver.flagsaver((dataset._DATASET_BASEPATH, temp_dir.full_path)):
      prompt = prompt_registry.get_prompt_metadata(
          "birdset_nbp_classification"
      ).load()
    self.assertIsInstance(prompt, prompt_lib.SoundClassificationPrompt)
    self.assertIn("Bird Classification", prompt.GetPromptTemplate())
    self.assertEqual(prompt.ProcessResponse('{"answer": "barswa"}'), "barswa")
    self.assertEqual(
        prompt.ProcessResponse('{"answer": "amerob"}'),
        prompt.INVALID_ANSWER_STR,
    )


if __name__ == "__main__":
  absltest.main()
