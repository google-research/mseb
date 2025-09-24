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

import re
from typing import Callable, final, Sequence

import jaxtyping
from mseb import encoder
from mseb import types
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub


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
  ):
    """Initializes the encoder with configuration.

    All subclasses of this class are expected to load the model in `_setup` and
    set `text_encode_fn` that takes a sequence of string prompts and returns a
    sequence of embeddings (2d np.ndarray or sequence of 1d np.ndarray).

    Args:
      normalizer: A function that normalizes the text before encoding. This is
        useful for removing special characters or formatting the text for better
        encoding results.
      prompt_template: Format of the prompt to be used for encoding.
    """
    super().__init__()
    self.normalizer = normalizer
    self.prompt_template = prompt_template
    self.text_encode_fn: (
        Callable[
            [Sequence[str]],
            jaxtyping.Float[jaxtyping.Array, 'N D']
            | jaxtyping.Shaped[np.ndarray, 'N']
            | Sequence[
                jaxtyping.Float[jaxtyping.Array, 'D']
                | jaxtyping.Shaped[np.ndarray, '']
            ],
        ]
        | None
    ) = None

  def _get_normalized_text_prompt(
      self, text: str, params: types.TextContextParams
  ) -> str:
    """Returns the prompt to be used for encoding."""
    if self.normalizer is not None:
      text = self.normalizer(text)

    if self.prompt_template is None:
      return text

    if hasattr(params, 'title') and params.title is not None:
      title = (
          params.title
          if self.normalizer is None
          else self.normalizer(params.title)
      )
    else:
      title = 'None'
    if params.context is not None:
      context = (
          params.context
          if self.normalizer is None
          else self.normalizer(params.context)
      )
    else:
      context = 'None'
    prompt = self.prompt_template.format(
        text=text, title=title, context=context
    )
    return prompt

  @final
  def _encode(
      self, text_batch: Sequence[types.Text]
  ) -> Sequence[types.TextEmbeddings]:
    """Encodes a batch of text sources."""
    prompt_batch = [
        self._get_normalized_text_prompt(text.text, text.context)
        for text in text_batch
    ]
    assert self.text_encode_fn is not None
    embeddings_batch = self.text_encode_fn(prompt_batch)

    return [
        types.TextEmbeddings(
            embeddings=np.expand_dims(embeddings, axis=0),
            spans=np.array([[0, len(text.text)]]),
            context=text.context,
        )
        for embeddings, text in zip(embeddings_batch, text_batch)
    ]


class GeckoTextEncoder(NormalizedTextEncoderWithPrompt):
  """Text encoder with Gecko model."""

  def __init__(
      self,
      model_path: str,
      normalizer: Callable[[str], str] | None = lambda x: re.sub(
          r'\[\d+\]', '', x.lower()
      ),
      prompt_template: str | None = 'title: {title} | text: {text}',
  ):
    """Initializes the transcript truth and Gecko models.

    Args:
      model_path: A serializable string (e.g., a GCS path or Hub ID) pointing to
        the model to be loaded in setup().
      normalizer: A function that normalizes the text before encoding. This is
        useful for removing special characters or formatting the text for better
        encoding results.
      prompt_template: Format of the prompt to be used for Gecko. Typically, the
        prompt is of the form: 'task: search result | query: {text}' for queries
        and 'title: {title} | text: {text}' for documents".
    """
    super().__init__(normalizer, prompt_template)
    self.model_path = model_path

  def _setup(self):
    """Loads the Gecko model."""
    gecko_model = tf_hub.load(self.model_path)
    self.text_encode_fn = lambda x: gecko_model.signatures['serving_default'](
        tf.constant(x)
    )['encodings'].numpy()


# For testing only.
class MockTextEncoder(NormalizedTextEncoderWithPrompt):

  def _setup(self):
    self.text_encode_fn = lambda prompts: np.zeros((len(prompts), 3))
