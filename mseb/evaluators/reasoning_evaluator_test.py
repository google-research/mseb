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

"""Tests for ReasoningEvaluator class."""

from typing import Any, Sequence, Tuple, Union

from absl.testing import absltest
from mseb import encoder
from mseb.evaluators import reasoning_evaluator
import numpy as np


class MockReasoningEncoder(encoder.Encoder):

  def encode_batch(
      self,
      sequences: Sequence[Union[str, Sequence[float]]],
      contexts: Sequence[encoder.ContextParams],
      **kwargs: Any,
  ) -> Sequence[Tuple[np.ndarray, np.ndarray]]:
    return [(np.array('Paris'), np.array([])) for _ in contexts]


class ReasoningEvaluatorTest(absltest.TestCase):

  def test_one_example(self):
    prompt = """
        Find the answer given the question, title and context. {question_tuple}
    """
    evaluator = reasoning_evaluator.ReasoningEvaluator(
        sound_encoder=MockReasoningEncoder(),
        encode_kwargs={},
    )
    input_data = """
        question: What is the capital of France?
        title: France
        context: Paris is the capital of France.
    """
    output1 = evaluator(
        sequence=[],
        context=encoder.ContextParams(
            prompt=prompt.format(question_tuple=input_data)
        ),
        reference='Paris')
    self.assertEqual(output1['f1'], 1.0)

    input_data = """
        question: How tall is Michael Jordan?
        title: Michael Jordan
        context: Michael Jordan is 6 feet 9 inches tall.
    """
    output2 = evaluator(
        sequence=[],
        context=encoder.ContextParams(
            prompt=prompt.format(question_tuple=input_data)
        ),
        reference='6 feet 9 inches')
    self.assertEqual(output2['f1'], 0.0)
    combined_scores = evaluator.combine_scores([output1, output2])
    self.assertEqual(combined_scores['f1'], 0.5)
    self.assertEqual(combined_scores['f1_std'], 0.5)


if __name__ == '__main__':
  absltest.main()
