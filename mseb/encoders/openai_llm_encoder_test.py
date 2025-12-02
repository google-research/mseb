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

"""Tests for the OpenAI API-based encoder."""

import json
from unittest import mock

from absl.testing import absltest
from mseb import types
from mseb.encoders import prompt as prompt_lib
from mseb.evaluators import reasoning_evaluator
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
import pytest

llm_encoder_lib = pytest.importorskip('mseb.encoders.openai_llm_encoder')

NO_ANSWER_STR = reasoning_evaluator.NO_ANSWER_STR


class MockOpenAILLMEncoder(llm_encoder_lib.OpenAILLMEncoder):
  """Mock OpenAILLMEncoder for testing."""

  def __init__(self):
    super().__init__(
        server_url='https://mock_server_url',
        api_key='mock_api_key',
        model_name='mock_model_name',
        prompt=prompt_lib.ReasoningPrompt(),
    )

  def _setup(self):
    self._client = mock.MagicMock()
    self._client.chat.completions.create.return_value = ChatCompletion(
        id='mock_id',
        object='chat.completion',
        created=1741570283,
        model=self._model_name,
        choices=[
            Choice(
                finish_reason='stop',
                index=0,
                message=ChatCompletionMessage(
                    role='assistant',
                    content=json.dumps({
                        'answer': 'Paris',
                        'rationale': 'Paris is the capital of France.',
                    }),
                ),
            )
        ],
    )
    super()._setup()


# @pytest.mark.gecko
# @pytest.mark.optional
class OpenAILLMEncoderTest(absltest.TestCase):

  def test_encode(self):
    llm_encoder = MockOpenAILLMEncoder()
    llm_encoder.setup()
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
