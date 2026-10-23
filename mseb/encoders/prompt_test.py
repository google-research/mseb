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

from absl.testing import absltest
from mseb.encoders import prompt as prompt_lib
from mseb.evaluators import reasoning_evaluator


class PromptTest(absltest.TestCase):

  NO_ANSWER_STR = reasoning_evaluator.NO_ANSWER_STR
  INVALID_ANSWER_STR = reasoning_evaluator.INVALID_ANSWER_STR
  NO_RESPONSE_STR = reasoning_evaluator.NO_RESPONSE_STR

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
    for non_mapping_json in ("null", "123", "true", '"answer"', '["answer"]'):
      self.assertEqual(
          prompt_lib.ProcessJsonResponse(
              non_mapping_json,
              keys=["answer"],
              key_to_value_map=dict({self.NO_ANSWER_STR: self.NO_ANSWER_STR}),
              invalid_response_value=self.INVALID_ANSWER_STR,
          ),
          self.INVALID_ANSWER_STR,
      )
      self.assertFalse(
          prompt_lib.ValidateJson(non_mapping_json, {"answer": str()})
      )
    self.assertEqual(
        prompt_lib.ProcessJsonResponse(
            '```json\n{"answer": "Paris"}\n```',
            keys=["answer"],
            key_to_value_map=None,
            invalid_response_value=self.INVALID_ANSWER_STR,
        ),
        "Paris",
    )
    self.assertEqual(
        prompt_lib.ProcessJsonResponse(
            "```json\n"
            + json.dumps({
                "answer": "```code```",
                "rationale": "Preserves internal fences.",
            })
            + "\n```",
            keys=["answer"],
            key_to_value_map=None,
            invalid_response_value=self.INVALID_ANSWER_STR,
        ),
        "```code```",
    )
    self.assertTrue(
        prompt_lib.ValidateJson(
            '```json\n{"answer": "Paris"}\n```', {"answer": str()}
        )
    )

  def test_classification_prompt(self):
    prompt = prompt_lib.ClassificationPrompt(
        class_labels=["alarm_set", "weather_query"]
    )
    self.assertEqual(
        prompt.ProcessResponse('{"answer": "alarm_set"}'),
        "alarm_set",
    )
    self.assertEqual(
        prompt.ProcessResponse('{"answer": {"nested": "alarm_set"}}'),
        prompt.INVALID_ANSWER_STR,
    )
    self.assertEqual(
        prompt.ProcessResponse('{"answer": 42}'),
        prompt.INVALID_ANSWER_STR,
    )
    self.assertEqual(
        prompt.ProcessResponse("null"),
        prompt.INVALID_ANSWER_STR,
    )
    self.assertEqual(
        prompt.ProcessResponse(prompt.NO_RESPONSE_STR),
        prompt.NO_RESPONSE_STR,
    )

    speaker_prompt = prompt_lib.SpeakerClassificationPrompt(
        class_labels=["male", "female"]
    )
    self.assertEqual(
        speaker_prompt.ProcessResponse('{"answer": "female"}'),
        "female",
    )
    self.assertEqual(
        speaker_prompt.ProcessResponse('{"answer": {"nested": "female"}}'),
        speaker_prompt.INVALID_ANSWER_STR,
    )

  def test_reasoning_prompt(self):
    prompt = prompt_lib.ReasoningPrompt()
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
                self.NO_ANSWER_STR: "Munich",
            })
        ),
        self.NO_ANSWER_STR,
    )
    self.assertEqual(
        prompt.ProcessResponse("invalid json"),
        self.INVALID_ANSWER_STR,
    )
    self.assertEqual(
        prompt.ProcessResponse(self.NO_RESPONSE_STR),
        self.NO_RESPONSE_STR,
    )

  def test_retrieval_prompt(self):
    prompt = prompt_lib.RetrievalPrompt()
    self.assertEqual(
        prompt.ProcessResponse(
            json.dumps([
                "english--7143613423291663770-hardneg-1",
                "english-4979886021869611788-hardneg-4",
                "english-6037841464917965779-hardneg-1",
            ])
        ),
        json.dumps([
            {"id": "english--7143613423291663770-hardneg-1"},
            {"id": "english-4979886021869611788-hardneg-4"},
            {"id": "english-6037841464917965779-hardneg-1"},
        ]),
    )
    self.assertEqual(
        prompt.ProcessResponse(
            json.dumps(
                "english--7143613423291663770-hardneg-1\n"
                "english-4979886021869611788-hardneg-4\n"
                "english-6037841464917965779-hardneg-1"
            )
        ),
        self.INVALID_ANSWER_STR,
    )
    self.assertEqual(
        prompt.ProcessResponse(self.NO_RESPONSE_STR), self.NO_RESPONSE_STR
    )

  def test_segmentation_prompt(self):
    prompt = prompt_lib.SegmentationPrompt()
    self.assertEqual(
        prompt.ProcessResponse(
            json.dumps({
                "term": "Paris",
                "start_time": 0.0,
                "end_time": 1.0,
            })
        ),
        json.dumps([{
            "term": "Paris",
            "start_time": 0.0,
            "end_time": 1.0,
        }]),
    )
    self.assertEqual(
        prompt.ProcessResponse(
            "\n".join([
                json.dumps({
                    "term": "Paris",
                    "start_time": 0.0,
                    "end_time": 1.0,
                }),
                json.dumps({
                    "term": "Munich",
                    "start_time": 2.0,
                    "end_time": 3.0,
                }),
                "invalid json",
                json.dumps({"term": "London "})
            ])
        ),
        json.dumps([
            {
                "term": "Paris",
                "start_time": 0.0,
                "end_time": 1.0,
            },
            {
                "term": "Munich",
                "start_time": 2.0,
                "end_time": 3.0,
            },
        ]),
    )
    self.assertEqual(
        prompt.ProcessResponse(
            json.dumps({
                "term": "Paris",
                "start_time": 0.0,
                "end_time": "1.0",
            })
        ),
        self.INVALID_ANSWER_STR,
    )
    self.assertEqual(
        prompt.ProcessResponse("invalid json"),
        self.INVALID_ANSWER_STR,
    )
    self.assertEqual(
        prompt.ProcessResponse(self.NO_RESPONSE_STR),
        self.NO_RESPONSE_STR,
    )

  def test_segmentation_from_alignment_prompt(self):
    prompt = prompt_lib.SegmentationFromAlignmentPrompt()
    print(prompt.GetPromptTemplate())
    self.assertEqual(
        prompt.ProcessResponse(
            json.dumps({
                "term": "Paris",
                "start_time": 0.0,
                "end_time": 1.0,
            })
        ),
        json.dumps([{
            "term": "Paris",
            "start_time": 0.0,
            "end_time": 1.0,
        }]),
    )
    self.assertEqual(
        prompt.ProcessResponse(
            "\n".join([
                json.dumps({
                    "term": "Paris",
                    "start_time": 0.0,
                    "end_time": 1.0,
                }),
                json.dumps({
                    "term": "Munich",
                    "start_time": 2.0,
                    "end_time": 3.0,
                }),
                "invalid json",
                json.dumps({"term": "London "})
            ])
        ),
        json.dumps([
            {
                "term": "Paris",
                "start_time": 0.0,
                "end_time": 1.0,
            },
            {
                "term": "Munich",
                "start_time": 2.0,
                "end_time": 3.0,
            },
        ]),
    )
    self.assertEqual(
        prompt.ProcessResponse(
            json.dumps({
                "term": "Paris",
                "start_time": 0.0,
                "end_time": "1.0",
            })
        ),
        self.INVALID_ANSWER_STR,
    )
    self.assertEqual(
        prompt.ProcessResponse("invalid json"),
        self.INVALID_ANSWER_STR,
    )
    self.assertEqual(
        prompt.ProcessResponse(self.NO_RESPONSE_STR),
        self.NO_RESPONSE_STR,
    )

  def test_sound_classification_prompt(self):
    prompt = prompt_lib.SoundClassificationPrompt(
        class_labels=["label_1", "label_2"]
    )
    self.assertIsInstance(prompt.GetPromptTemplate(), str)
    self.assertEqual(
        prompt.ProcessResponse(
            json.dumps({
                "answer": "label_1",
            })
        ),
        "label_1",
    )
    self.assertEqual(
        prompt.ProcessResponse(
            "\n".join([
                json.dumps({
                    "answer": "label_1",
                }),
                json.dumps({
                    "answer": "label_2",
                }),
                "invalid json",
            ])
        ),
        "label_1\nlabel_2",
    )
    self.assertEqual(
        prompt.ProcessResponse(
            json.dumps({
                "answer": "label_3",
            })
        ),
        self.INVALID_ANSWER_STR,
    )
    self.assertEqual(
        prompt.ProcessResponse(
            '{\n  "answer": "label_1"\n}\n{\n  "answer": "label_2"\n}'
        ),
        "label_1\nlabel_2",
    )
    self.assertEqual(
        prompt.ProcessResponse('{"answer": ["label_2", "label_1", "label_2"]}'),
        "label_2\nlabel_1",
    )
    self.assertEqual(
        prompt.ProcessResponse(
            '[{"answer": "label_1"}, {"answer": "label_3"}]'
        ),
        "label_1",
    )
    self.assertEqual(
        prompt.ProcessResponse('{"answer": {"invalid": 1}}'),
        self.INVALID_ANSWER_STR,
    )
    self.assertEqual(
        prompt.ProcessResponse('"label_1"'),
        self.INVALID_ANSWER_STR,
    )
    self.assertEqual(
        prompt.ProcessResponse("null"),
        self.INVALID_ANSWER_STR,
    )
    self.assertEqual(
        prompt.ProcessResponse(
            '```json\n{"answer": "label_1"}\n```\n'
            '```json\n{"answer": "label_2"}\n```'
        ),
        "label_1\nlabel_2",
    )
    self.assertEqual(
        prompt.ProcessResponse(
            'Here are the predicted sound classes:\n'
            '```json\n{"answer": "label_1"}\n```\n'
            '```json\n{"answer": "label_2"}\n```'
        ),
        "label_1\nlabel_2",
    )
    self.assertEqual(
        prompt.ProcessResponse(
            self.NO_RESPONSE_STR
        ),
        self.NO_RESPONSE_STR,
    )


if __name__ == "__main__":
  absltest.main()
