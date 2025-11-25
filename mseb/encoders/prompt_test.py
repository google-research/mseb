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

import json

from absl.testing import absltest
from mseb.encoders import prompt as prompt_lib


class PromptTest(absltest.TestCase):

  NO_ANSWER_STR = "No answer"
  INVALID_ANSWER_STR = ""

  def test_process_json_response(self):
    self.assertEqual(
        prompt_lib.ProcessJsonResponse(
            json.dumps({
                "answer": "Paris",
                "rationale": "Paris is the capital of France.",
            }),
            keys=["answer"],
            key_to_value_map=dict({self.NO_ANSWER_STR: self.NO_ANSWER_STR}),
            invalid_response_value=self.INVALID_ANSWER_STR,
        ),
        "Paris",
    )
    self.assertEqual(
        prompt_lib.ProcessJsonResponse(
            json.dumps({
                self.NO_ANSWER_STR: "Paris is the capital of France.",
            }),
            keys=["answer"],
            key_to_value_map=dict({self.NO_ANSWER_STR: self.NO_ANSWER_STR}),
            invalid_response_value=self.INVALID_ANSWER_STR,
        ),
        self.NO_ANSWER_STR,
    )
    self.assertEqual(
        prompt_lib.ProcessJsonResponse(
            json.dumps({
                "invalid_response": "Paris is the capital of France.",
            }),
            keys=["answer"],
            key_to_value_map=dict({self.NO_ANSWER_STR: self.NO_ANSWER_STR}),
            invalid_response_value=self.INVALID_ANSWER_STR,
        ),
        self.INVALID_ANSWER_STR,
    )
    self.assertEqual(
        prompt_lib.ProcessJsonResponse(
            "invalid json",
            keys=["answer"],
            key_to_value_map=dict({self.NO_ANSWER_STR: self.NO_ANSWER_STR}),
            invalid_response_value=self.INVALID_ANSWER_STR,
        ),
        self.INVALID_ANSWER_STR,
    )

  def test_classification_prompt(self):
    class_labels = ["Paris", "London", "New York"]
    prompt = prompt_lib.ClassificationPrompt(class_labels=class_labels)
    self.assertEqual(
        prompt.ProcessResponse(
            json.dumps({
                "answer": "Paris",
            })
        ),
        "Paris",
    )
    self.assertEqual(
        prompt.ProcessResponse(
            json.dumps({
                "answer": "Munich",
            })
        ),
        self.INVALID_ANSWER_STR,
    )
    self.assertEqual(
        prompt.ProcessResponse("invalid json"),
        self.INVALID_ANSWER_STR,
    )


if __name__ == "__main__":
  absltest.main()
