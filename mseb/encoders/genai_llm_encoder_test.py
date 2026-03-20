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

"""Tests for the GenaiLLMTextEncoder class."""

from absl.testing import absltest
from absl.testing import parameterized
from mseb import types
from mseb.encoders import genai_llm_encoder as genai_llm_encoder_lib
from mseb.encoders import prompt as prompt_lib
from mseb.evaluators import reasoning_evaluator
import numpy as np


NO_ANSWER_STR = reasoning_evaluator.NO_ANSWER_STR
INVALID_ANSWER_STR = reasoning_evaluator.INVALID_ANSWER_STR


class GenaiLLMTextEncoderTest(absltest.TestCase):

  def test_encode(self):
    genai_llm_encoder = genai_llm_encoder_lib.GenaiLLMTextEncoder(
        model_path='not_used',
        api_key='mock_api_key',
        prompt=prompt_lib.ReasoningPrompt(),
    )
    genai_llm_encoder.prompt_encode_fn = lambda prompts: np.array(
        ['{"answer": "Paris", "rationale": "Paris is the capital of France."}']
        * len(prompts)
    )
    genai_llm_encoder._is_setup = True
    outputs = genai_llm_encoder.encode(
        [
            types.TextWithTitleAndContext(
                text='What is the capital of France?',
                title_text='France',
                context_text='Paris is the capital of France.',
                context=types.TextContextParams(id='1'),
            ),
            types.TextWithTitleAndContext(
                text='How tall is Michael Jordan?',
                title_text='Michael Jordan',
                context_text='Michael Jordan is 6 feet 9 inches tall.',
                context=types.TextContextParams(id='2'),
            ),
        ],
    )
    self.assertLen(outputs, 2)
    self.assertEqual(outputs[0].context.id, '1')
    self.assertEqual(outputs[1].context.id, '2')
    for output in outputs:
      self.assertIsInstance(output, types.TextEmbedding)
      self.assertEqual(output.embedding.shape, (1,))
      self.assertEqual(str(output.embedding[0]), 'Paris')

  class GenaiLLMTranscriptTruthEncoderTest(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(
            testcase_name='answer_and_rationale',
            return_value=(
                '{"answer": "Paris", "rationale": "Paris is the capital of'
                ' France."}'
            ),
            expected_answer='Paris',
        ),
        dict(
            testcase_name='answer_only',
            return_value='{"answer": "Paris"}',
            expected_answer='Paris',
        ),
        dict(
            testcase_name='rationale_only',
            return_value='{"rationale": "Paris is the capital of France."}',
            expected_answer=INVALID_ANSWER_STR,
        ),
        dict(
            testcase_name='invalid_json',
            return_value='invalid json',
            expected_answer=INVALID_ANSWER_STR,
        ),
        dict(
            testcase_name='no_answer',
            return_value=(
                '{{"{no_answer}": "Answer not found in passage."}}'.format(
                    no_answer=NO_ANSWER_STR
                )
            ),
            expected_answer=NO_ANSWER_STR,
        ),
    )
    def test_encode_batch(self, return_value: str, expected_answer: str):
      genai_llm_encoder = genai_llm_encoder_lib.GenaiLLMTranscriptTruthEncoder(
          model_path='mock_model_path', api_key='mock_api_key'
      )
      genai_llm_encoder._encoders[-2].prompt_encode_fn = (
          lambda prompts: np.array([return_value] * len(prompts))
      )
      genai_llm_encoder._encoders[-2]._is_setup = True
      genai_llm_encoder.setup()
      outputs = genai_llm_encoder.encode(
          [
              types.SoundWithTitleAndContext(
                  waveform=np.array([1.0, 2.0, 3.0]),
                  context=types.SoundContextParams(
                      id='1',
                      sample_rate=48000,
                      length=3,
                      text='What is the capital of France?',
                  ),
                  title_text='France',
                  context_text='Paris is the capital of France.',
              ),
          ],
      )
      self.assertLen(outputs, 1)
      output = outputs[0]
      self.assertEqual(output.context.id, '1')
      self.assertIsInstance(output, types.TextPrediction)
      self.assertEqual(str(output.prediction), expected_answer)


if __name__ == '__main__':
  absltest.main()
