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

import logging
from typing import Callable
from unittest import mock

from absl.testing import absltest
from mseb import types
from mseb.encoders import prompt as prompt_lib
import numpy as np
import pytest

text_encoder_with_prompt = pytest.importorskip(
    "mseb.encoders.text_encoder_with_prompt"
)

logger = logging.getLogger(__name__)


class MockTextEncoderWithPrompt(text_encoder_with_prompt.TextEncoderWithPrompt):
  """A concrete implementation of TextEncoderWithPrompt.

  For testing purposes.
  """

  def _setup(self):
    pass

  def __init__(
      self,
      prompt_encode_fn: Callable[[str], np.ndarray] | None = None,
      normalizer: Callable[[str], str] | None = None,
      prompt_template: str | None = None,
  ):
    super().__init__(
        normalizer, prompt=prompt_lib.DefaultPrompt(prompt_template)
    )
    if prompt_encode_fn is not None:
      self.prompt_encode_fn = prompt_encode_fn
    else:
      self.prompt_encode_fn = mock.MagicMock(return_value=np.zeros((10, 8)))


class TextEncoderWithPromptTest(absltest.TestCase):

  def test_check_input_types_does_not_raise_error_for_text_inputs(self):
    mock_encoder = MockTextEncoderWithPrompt()
    mock_encoder._check_input_types([
        types.Text(
            text="This is a text.", context=types.TextContextParams(id="id1")
        ),
        types.Text(
            text="This is another text.",
            context=types.TextContextParams(id="id2"),
        ),
    ])

  def test_check_input_types_does_not_raise_error_for_text_and_sound_inputs(
      self,
  ):
    mock_encoder = MockTextEncoderWithPrompt()
    mock_encoder._check_input_types([
        types.Text(
            text="This is a text.", context=types.TextContextParams(id="text")
        ),
        types.Sound(
            waveform=np.array([1.0, 2.0, 3.0, 4.0]),
            context=types.SoundContextParams(
                sample_rate=2, length=4, id="sound"
            ),
        ),
    ])

  def test_get_normalized_text_prompt_with_normalizer(self):
    self.assertEqual(
        MockTextEncoderWithPrompt(
            normalizer=lambda x: x.lower()
        )._get_normalized_text_prompt("This is a text."),
        "this is a text.",
    )
    self.assertEqual(
        MockTextEncoderWithPrompt(
            prompt_template="task: search result | query: {text}",
        )._get_normalized_text_prompt("This is a text."),
        "task: search result | query: This is a text.",
    )
    self.assertEqual(
        MockTextEncoderWithPrompt(
            normalizer=lambda x: x.lower(),
            prompt_template="title: {title} | context: {text}",
        )._get_normalized_text_prompt("This is ANOTHER text."),
        "title: None | context: this is another text.",
    )
    self.assertEqual(
        MockTextEncoderWithPrompt(
            normalizer=lambda x: x.lower(),
            prompt_template="title: {title} | context: {text}",
        )._get_normalized_text_prompt("This is ANOTHER text.", title="Title"),
        "title: title | context: this is another text.",
    )

  def test_normalizer_is_applied_to_text(self):
    mock_encoder = MockTextEncoderWithPrompt(
        normalizer=lambda x: x.lower(),
        prompt_template="title: {title} | context: {text}",
    )
    mock_encoder.prompt_encode_fn = mock.MagicMock(
        return_value=np.zeros((10, 8))
    )
    _ = mock_encoder.encode([
        types.TextWithTitleAndContext(
            text="This is a text.",
            context=types.TextContextParams(id="id1"),
        ),
        types.TextWithTitleAndContext(
            text="This is ANOTHER text.",
            title_text="Abc",
            context=types.TextContextParams(id="id2"),
        ),
    ])
    _ = mock_encoder.prompt_encode_fn.assert_called_once_with([
        ("title: None | context: this is a text.", None),
        ("title: abc | context: this is another text.", None),
    ])

  def test_get_normalized_text_prompt_with_task_prompts(self):
    mock_encoder = MockTextEncoderWithPrompt(prompt_template="default: {text}")
    mock_encoder.task_prompts = {
        "Task1": prompt_lib.DefaultPrompt("task1 prompt: {text}"),
        "Task2": prompt_lib.DefaultPrompt("task2 prompt: {text}"),
    }

    mock_task1 = mock.MagicMock()
    mock_task1.metadata.name = "Task1"

    mock_task2 = mock.MagicMock()
    mock_task2.metadata.name = "Task2"

    mock_task3 = mock.MagicMock()
    mock_task3.metadata.name = "Task3"

    mock_encoder.set_task(mock_task1)
    self.assertEqual(
        mock_encoder._get_normalized_text_prompt("hello"),
        "task1 prompt: hello",
    )

    mock_encoder.set_task(mock_task2)
    self.assertEqual(
        mock_encoder._get_normalized_text_prompt("hello"),
        "task2 prompt: hello",
    )

    mock_encoder.set_task(mock_task3)
    self.assertEqual(
        mock_encoder._get_normalized_text_prompt("hello"),
        "default: hello",
    )


if __name__ == "__main__":
  absltest.main()
