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

"""Multi-modal encoders with prompt."""

import dataclasses
import json
import re
from typing import Callable, final, Optional, Sequence, Tuple

import jaxtyping
from mseb import encoder
from mseb import types
from mseb import utils
from mseb.encoders import prompt as prompt_lib
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub


class TextEncoderWithPrompt(encoder.MultiModalEncoder):
  """Defines the interface for encoding text and sound into embeddings using a prompt.

  This abstract class provides a standardized structure for encoders
  within the MSEB benchmark. It's designed for lazy loading of models, making it
  suitable for large-scale, distributed processing environments.

  Subclasses are responsible for implementing the model loading logic (`setup`).
  """

  # String to use for the `text`` field in the prompt template when encoding
  # audio.
  TEXT_FOR_AUDIO = '(in following audio file)'

  def __init__(
      self,
      normalizer: Callable[[str], str] | None = None,
      prompt: prompt_lib.Prompt = prompt_lib.DefaultPrompt(),
  ):
    """Initializes the encoder with configuration.

    All subclasses of this class are expected to load the model in `_setup` and
    set `prompt_encode_fn` that takes a sequence of multimodal prompts and
    returns a sequence of embeddings (2d np.ndarray or sequence of 1d
    np.ndarray).

    Args:
      normalizer: A function that normalizes the text before encoding. This is
        useful for removing special characters or formatting the text for better
        encoding results.
      prompt: Prompt object definiing the prompt template and reponse format to
        be used for encoding.
    """
    super().__init__()
    self.normalizer = normalizer
    self.prompt = prompt
    self.prompt_encode_fn: (
        Callable[
            [Sequence[Tuple[str, Optional[bytes]]]],
            jaxtyping.Float[jaxtyping.Array, 'N D']
            | jaxtyping.Shaped[np.ndarray, 'N']
            | Sequence[
                jaxtyping.Float[jaxtyping.Array, 'D']
                | jaxtyping.Shaped[np.ndarray, '']
            ],
        ]
        | None
    ) = None

  @final
  def _check_input_types(self, batch: Sequence[types.MultiModalObject]) -> None:
    if not all(
        any(isinstance(x, type) for type in [types.Text, types.Sound])
        for x in batch
    ):
      raise ValueError(
          'TextEncoderWithPrompt only supports a batch of Text or'
          ' Sound inputs.'
      )

  def _get_normalized_text_prompt(
      self, text: str, title: str | None = None, context: str | None = None
  ) -> str:
    """Returns the prompt to be used for encoding."""
    prompt_template = self.prompt.GetPromptTemplate()
    if self.normalizer is not None:
      text = self.normalizer(text)

    if prompt_template is None:
      return text

    if title is not None:
      title = title if self.normalizer is None else self.normalizer(title)
    else:
      title = 'None'
    if context is not None:
      context = context if self.normalizer is None else self.normalizer(context)
    else:
      context = 'None'
    text_prompt = prompt_template.format(
        text=text, title=title, context=context
    )
    return text_prompt

  @final
  def _encode(
      self, batch: Sequence[types.MultiModalObject]
  ) -> Sequence[types.TextEmbedding | types.SoundEmbedding]:
    """Encodes a batch of text or audio sources."""
    prompt_batch = []
    for example in batch:
      if isinstance(example, types.TextWithTitleAndContext):
        prompt_text = self._get_normalized_text_prompt(
            example.text, title=example.title_text, context=example.context_text
        )
        prompt_audio = None
      elif isinstance(example, types.Text):
        prompt_text = self._get_normalized_text_prompt(example.text)
        prompt_audio = None
      elif isinstance(example, types.SoundWithTitleAndContext):
        prompt_text = self._get_normalized_text_prompt(
            self.TEXT_FOR_AUDIO,
            title=example.title_text,
            context=example.context_text,
        )
        prompt_audio = utils.sound_to_wav_bytes(example)
      elif isinstance(example, types.Sound):
        prompt_text = self._get_normalized_text_prompt(self.TEXT_FOR_AUDIO)
        prompt_audio = utils.sound_to_wav_bytes(example)
      else:
        raise ValueError('Unexpected input type.')
      prompt_batch.append((prompt_text, prompt_audio))

    assert self.prompt_encode_fn is not None
    response_batch = self.prompt_encode_fn(prompt_batch)
    embeddings_batch = [
        self.prompt.ProcessResponse(response) for response in response_batch
    ]

    outputs = []
    for embeddings, example, response in zip(
        embeddings_batch, batch, response_batch
    ):
      if isinstance(response, str):
        debug_text = json.dumps({'model_response': response})
      else:
        debug_text = None
      assert isinstance(example, types.Text) or isinstance(example, types.Sound)
      if isinstance(example, types.Text):
        outputs.append(
            types.TextEmbedding(
                embedding=np.expand_dims(embeddings, axis=0),
                spans=np.array([[0, len(example.text)]]),
                context=dataclasses.replace(example.context, text=example.text,
                                            debug_text=debug_text),
            )
        )
      if isinstance(example, types.Sound):
        outputs.append(
            types.SoundEmbedding(
                embedding=np.expand_dims(embeddings, axis=0),
                timestamps=np.array([[0, len(example.waveform)]]),
                context=dataclasses.replace(
                    example.context,
                    debug_text=debug_text,
                ),
            )
        )
    return outputs


class GeckoTextEncoder(TextEncoderWithPrompt):
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
    super().__init__(
        normalizer, prompt=prompt_lib.DefaultPrompt(prompt_template)
    )
    self.model_path = model_path

  def _setup(self):
    """Loads the Gecko model."""
    gecko_model = tf_hub.load(self.model_path)
    self.prompt_encode_fn = lambda batch: gecko_model.signatures[
        'serving_default'
    ](tf.constant([x[0] for x in batch]))['encodings'].numpy()


# For testing only.
class MockTextEncoder(TextEncoderWithPrompt):

  def _setup(self):
    self.prompt_encode_fn = lambda prompts: np.zeros((len(prompts), 3))
