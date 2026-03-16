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

"""Tests for the LiteLLM LLM Encoder class."""

from absl.testing import absltest
from mseb import types
from mseb.encoders import prompt as prompt_lib
from mseb.evaluators import reasoning_evaluator
import numpy as np
import pytest

litellm_encoder_lib = pytest.importorskip('mseb.encoders.litellm_encoder')


NO_ANSWER_STR = reasoning_evaluator.NO_ANSWER_STR
INVALID_ANSWER_STR = reasoning_evaluator.INVALID_ANSWER_STR


@pytest.mark.whisper
@pytest.mark.optional
class LiteLLMTextEncoderTest(absltest.TestCase):

  def test_encode(self):
    llm_encoder = litellm_encoder_lib.LiteLLMTextEncoder(
        model_name='dummy_model',
        api_key='dummy_api_key',
        prompt=prompt_lib.ReasoningPrompt(),
    )
    llm_encoder.prompt_encode_fn = lambda prompts: np.array(
        ['{"answer": "Paris", "rationale": "Paris is the capital of France."}']
        * len(prompts)
    )
    llm_encoder._is_setup = True
    outputs = llm_encoder.encode(
        [
            types.Text(
                text='What is the capital of France?',
                context=types.TextContextParams(
                    id='1',
                    title='France',
                    context='Paris is the capital of France.',
                ),
            ),
            types.Text(
                text='How tall is Michael Jordan?',
                context=types.TextContextParams(
                    id='2',
                    title='Michael Jordan',
                    context='Michael Jordan is 6 feet 9 inches tall.',
                ),
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


if __name__ == '__main__':
  absltest.main()
