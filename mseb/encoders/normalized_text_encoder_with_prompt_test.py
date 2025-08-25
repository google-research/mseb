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

import logging
from typing import Any, Callable
from unittest import mock

from absl.testing import absltest
from mseb import types
from mseb.encoders import normalized_text_encoder_with_prompt
import numpy as np


logger = logging.getLogger(__name__)


class MockNormalizedTextEncoderWithPrompt(
    normalized_text_encoder_with_prompt.NormalizedTextEncoderWithPrompt
):
  """A concrete implementation of NormalizedTextEncoderWithPrompt.

  For testing purposes.
  """

  def setup(self):
    pass

  def __init__(
      self,
      text_encode_fn: Callable[[str], np.ndarray] | None = None,
      normalizer: Callable[[str], str] | None = None,
      prompt_template: str | None = None,
      **kwargs: Any
  ):
    super().__init__(normalizer, prompt_template, **kwargs)
    if text_encode_fn is not None:
      self.text_encode_fn = text_encode_fn
    else:
      self.text_encode_fn = mock.MagicMock(return_value=np.zeros((10, 8)))
    self.setup = mock.MagicMock(side_effect=self._setup_impl)

  def _setup_impl(self):
    """The actual implementation for the mocked setup method."""
    self._model_loaded = True


class FaultySetupNormalizedTextEncoderWithPrompt(
    normalized_text_encoder_with_prompt.NormalizedTextEncoderWithPrompt
):
  """An encoder that "forgets" to set the _model_loaded flag in setup."""

  def setup(self):
    logger.info("Faulty setup was called, but did not set the flag.")


class NormalizedTextEncoderWithPromptTest(absltest.TestCase):

  def test_initialization_is_lazy_and_does_not_call_setup(self):
    mock_encoder = MockNormalizedTextEncoderWithPrompt()
    mock_encoder.setup.assert_not_called()

  def test_encode_triggers_setup_exactly_once(self):
    mock_encoder = MockNormalizedTextEncoderWithPrompt(
        text_encode_fn=lambda x: np.zeros((10, 8)),
    )

    mock_encoder.encode(
        types.Text(
            text="This is a text.", context=types.TextContextParams(id="id")
        )
    )
    mock_encoder.setup.assert_called_once()

    mock_encoder.encode_batch([
        types.Text(
            text="This is another text.",
            context=types.TextContextParams(id="id"),
        )
    ])
    mock_encoder.setup.assert_called_once()

  def test_encode_batch_triggers_setup_exactly_once(self):
    mock_encoder = MockNormalizedTextEncoderWithPrompt()
    batch = [
        types.Text(
            text="This is a text.", context=types.TextContextParams(id="id1")
        ),
        types.Text(
            text="This is another text.",
            context=types.TextContextParams(id="id2"),
        ),
    ]
    mock_encoder.encode_batch(batch)
    mock_encoder.setup.assert_called_once()

  def test_default_encode_calls_encode_batch_with_single_item(self):
    mock_encoder = MockNormalizedTextEncoderWithPrompt()
    mock_encoder.encode_batch = mock.MagicMock()
    text = types.Text(
        text="This is a text.", context=types.TextContextParams(id="id")
    )

    mock_encoder.encode(text)
    mock_encoder.encode_batch.assert_called_once_with([text], **{})

  def test_faulty_setup_raises_runtime_error(self):
    mock_encoder = FaultySetupNormalizedTextEncoderWithPrompt()
    with self.assertRaises(RuntimeError):
      mock_encoder.encode(
          types.Text(
              text="This is a text.", context=types.TextContextParams(id="id")
          )
      )

  def test_init_kwargs_are_stored_for_later_use(self):
    mock_encoder = MockNormalizedTextEncoderWithPrompt(
        special_param=123, other_param="abc"
    )
    self.assertIn("special_param", mock_encoder._kwargs)
    self.assertEqual(mock_encoder._kwargs["special_param"], 123)

  def test_final_decorator_prevents_override_in_static_analysis(self):

    class BadNormalizedTextEncoderWithPrompt(
        normalized_text_encoder_with_prompt.NormalizedTextEncoderWithPrompt
    ):

      def setup(self):
        self._model_loaded = True

    bad_encoder = BadNormalizedTextEncoderWithPrompt()
    self.assertIsNotNone(bad_encoder)

  def test_get_normalized_text_prompt_with_normalizer(self):
    self.assertEqual(
        MockNormalizedTextEncoderWithPrompt(
            normalizer=lambda x: x.lower()
        )._get_normalized_text_prompt(
            "This is a text.", types.TextContextParams(id="id")
        ),
        "this is a text.",
    )
    self.assertEqual(
        MockNormalizedTextEncoderWithPrompt(
            prompt_template="task: search result | query: {text}",
        )._get_normalized_text_prompt(
            "This is a text.", types.TextContextParams(id="id")
        ),
        "task: search result | query: This is a text.",
    )
    self.assertEqual(
        MockNormalizedTextEncoderWithPrompt(
            normalizer=lambda x: x.lower(),
            prompt_template="title: {title} | context: {text}",
        )._get_normalized_text_prompt(
            "This is ANOTHER text.", types.TextContextParams(id="id")
        ),
        "title: None | context: this is another text.",
    )
    self.assertEqual(
        MockNormalizedTextEncoderWithPrompt(
            normalizer=lambda x: x.lower(),
            prompt_template="title: {title} | context: {text}",
        )._get_normalized_text_prompt(
            "This is ANOTHER text.",
            types.TextContextParams(id="id", title="Title"),
        ),
        "title: title | context: this is another text.",
    )

  def test_normalizer_is_applied_to_text(self):
    mock_encoder = MockNormalizedTextEncoderWithPrompt(
        normalizer=lambda x: x.lower(),
        prompt_template="title: {title} | context: {text}",
    )
    mock_encoder.text_encode_fn = mock.MagicMock(return_value=np.zeros((10, 8)))
    _ = mock_encoder.encode_batch([
        types.Text(
            text="This is a text.", context=types.TextContextParams(id="id1")
        ),
        types.Text(
            text="This is ANOTHER text.",
            context=types.TextContextParams(id="id2", title="Abc"),
        ),
    ])
    _ = mock_encoder.text_encode_fn.assert_called_once_with([
        "title: None | context: this is a text.",
        "title: abc | context: this is another text.",
    ])


if __name__ == "__main__":
  absltest.main()
