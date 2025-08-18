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

"""Gecko embeddings encoder on transcript truth."""

from typing import Any, Sequence, Tuple, Union

from mseb import encoder
import numpy as np
import tensorflow as tf


class GeckoTranscriptTruthEncoder(encoder.Encoder):
  """Transcript truth encoder with Gecko model."""

  def __init__(
      self,
      gecko_model: tf.keras.Model,
      prompt_template: str = 'task: search result | query: {text}',
  ):
    """Initializes the transcript truth and Gecko models.

    Args:
      gecko_model: An instance of Gecko model.
      prompt_template: Format of the prompt to be used for Gecko. Typically, the
        prompt is of the form: 'task: search result | query: {text}' for queries
        and 'title: {title} | text: {text}' for documents".
    """
    self.transcript_truths_encode_fn = lambda x: gecko_model.signatures[
        'serving_default'
    ](tf.constant(x))['encodings'].numpy()
    self.prompt_template = prompt_template

  def encode(
      self,
      sequence: Union[str, Sequence[float]],
      context: encoder.ContextParams,
      **kwargs: Any,
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Encodes the transcript truth and Gecko embeddings."""
    timestamps = np.array(
        [[context.audio_start_seconds, context.audio_end_seconds]]
    )
    title = hasattr(context, 'title') and context.title or 'None'
    prompts = [self.prompt_template.format(text=context.text, title=title)]
    embeddings = self.transcript_truths_encode_fn(prompts)
    return timestamps, embeddings
