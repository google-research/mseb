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

"""Normalized text encoders with prompt."""

from typing import Any, Callable, Sequence, final

from mseb import encoder
from mseb import types
import numpy as np


class NormalizedTextEncoderWithPrompt(encoder.TextEncoder):
  """Defines the interface for encoding normalized text prompt into embeddings.

  This abstract class provides a standardized structure for text encoders
  within the MSEB benchmark. It's designed for lazy loading of models, making it
  suitable for large-scale, distributed processing environments.

  Subclasses are responsible for implementing the model loading logic (`setup`).
  """

  def __init__(
      self,
      normalizer: Callable[[str], str] | None = None,
      prompt_template: str | None = None,
      **kwargs: Any,
  ):
    """Initializes the encoder with configuration.

    All subclasses of this class are expected to load the model in `setup` and
    set `text_encode_fn` that takes a sequence of string prompts and returns a
    sequence of embeddings (2d np.ndarray or sequence of 1d np.ndarray).

    Args:
      normalizer: A function that normalizes the text before encoding. This is
        useful for removing special characters or formatting the text for better
        encoding results.
      prompt_template: Format of the prompt to be used for encoding.
      **kwargs: Model-specific initialization arguments that will be stored in
        `self._kwargs` for use in `setup()`.
    """
    super().__init__(**kwargs)
    self.normalizer = normalizer
    self.prompt_template = prompt_template
    self.text_encode_fn: (
        Callable[[Sequence[str]], np.ndarray | Sequence[np.ndarray]] | None
    ) = None

  def _get_normalized_text_prompt(
      self, text: str, params: types.TextContextParams
  ) -> str:
    """Returns the prompt to be used for encoding."""
    if self.normalizer is not None:
      text = self.normalizer(text)

    if self.prompt_template is None:
      return text

    if hasattr(params, "title") and params.title is not None:
      title = (
          params.title
          if self.normalizer is None
          else self.normalizer(params.title)
      )
    else:
      title = "None"
    if params.context is not None:
      context = (
          params.context
          if self.normalizer is None
          else self.normalizer(params.context)
      )
    else:
      context = "None"
    prompt = self.prompt_template.format(
        text=text, title=title, context=context
    )
    return prompt

  @final
  def _encode_batch(
      self, text_batch: Sequence[types.Text], **kwargs: Any
  ) -> Sequence[types.TextEmbeddings]:
    """Encodes a batch of text sources."""
    prompt_batch = [
        self._get_normalized_text_prompt(text.text, text.params)
        for text in text_batch
    ]
    assert self.text_encode_fn is not None
    embeddings_batch = self.text_encode_fn(prompt_batch)

    return [
        types.TextEmbeddings(
            id=text.id,
            embeddings=embeddings[np.newaxis],
            spans=np.array([[0, len(text.text)]]),
        )
        for embeddings, text in zip(embeddings_batch, text_batch)
    ]
