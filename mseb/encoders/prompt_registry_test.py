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

from absl.testing import absltest
from absl.testing import parameterized
from mseb.encoders import prompt_registry


class PromptRegistryTest(parameterized.TestCase):

  @parameterized.parameters(
      prompt_registry.reasoning,
      prompt_registry.intent_classification,
      prompt_registry.speaker_gender_classification,
      prompt_registry.segmentation,
  )
  def test_load_encoder(self, meta):
    prompt = meta.load()
    self.assertIsNotNone(prompt)
    self.assertIsInstance(prompt.GetPromptTemplate(), str)
    self.assertNotEmpty(prompt.GetPromptTemplate())


if __name__ == "__main__":
  absltest.main()
