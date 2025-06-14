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

"""Cascaded Whisper and text embedder encoders."""

from typing import Any, Callable, Sequence, Tuple, Union

from mseb import encoder
from mseb.encoders import whisper_encoder
import numpy as np
import tensorflow as tf
import whisper


class EmbedWhisperEncoder(encoder.Encoder):
  """A base class for encoding speech with a text embedder."""

  def __init__(
      self,
      whisper_model: whisper.Whisper,
      transcripts_encode_fn: Callable[[Sequence[str]], np.ndarray],
      prompt_template: str = '{text}',
  ):
    """Initializes the Whisper and text embedder.

    Args:
      whisper_model: An instance of Whisper model.
      transcripts_encode_fn: A function that takes a sequence of strings and
        returns a numpy array of embeddings.
      prompt_template: Prompt template to be used for the text embedder.
    """
    self.whisper_encoder = whisper_encoder.SpeechToTextEncoder(whisper_model)
    self.transcripts_encode_fn = transcripts_encode_fn
    self.prompt_template = prompt_template

  def encode(
      self,
      sequence: Union[str, Sequence[float]],
      context: encoder.ContextParams,
      **kwargs: Any,
  ) -> Tuple[np.ndarray, np.ndarray]:
    timestamps, transcripts = self.whisper_encoder.encode(
        sequence, context, **kwargs
    )
    prompts = [
        self.prompt_template.format(text=transcript)
        for transcript in transcripts
    ]
    embeddings = self.transcripts_encode_fn(prompts)
    return timestamps, embeddings


class GeckoWhisperEncoder(EmbedWhisperEncoder):
  """Cascaded Whisper and Gecko encoder."""

  def __init__(
      self,
      whisper_model: whisper.Whisper,
      gecko_model: tf.keras.Model,
      prompt_template: str = 'task: search result | query: {text}',
  ):
    """Initializes the Whisper and Gecko models.

    Args:
      whisper_model: An instance of Whisper model.
      gecko_model: An instance of Gecko model.
      prompt_template: Format of the prompt to be used for Gecko. Typically, the
        prompt is of the form: 'task: search result | query: {text}' for queries
        and 'title: {title} | text: {text}' for documents"
    """
    super().__init__(
        whisper_model=whisper_model,
        transcripts_encode_fn=lambda x: gecko_model.signatures[
            'serving_default'
        ](tf.constant(x))['encodings'].numpy(),
        prompt_template=prompt_template,
    )
