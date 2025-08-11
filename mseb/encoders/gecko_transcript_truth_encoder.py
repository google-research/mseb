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
from mseb import types
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub


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


class GeckoTranscriptTruthEncoderV2(encoder.SoundEncoder):
  """Transcript truth encoder with Gecko model."""

  def __init__(
      self,
      model_path: str,
      prompt_template: str = 'task: search result | query: {text}',
  ):
    """Initializes the transcript truth and Gecko models.

    Args:
      model_path: A serializable string (e.g., a GCS path or Hub ID) pointing to
        the model to be loaded in setup().
      prompt_template: Format of the prompt to be used for Gecko. Typically, the
        prompt is of the form: 'task: search result | query: {text}' for queries
        and 'title: {title} | text: {text}' for documents".
    """
    super().__init__(model_path)
    self.prompt_template = prompt_template
    self.transcript_truths_encode_fn = None

  def setup(self):
    """Loads the Gecko model."""
    gecko_model = tf_hub.load(self.model_path)
    self.transcript_truths_encode_fn = lambda x: gecko_model.signatures[
        'serving_default'
    ](tf.constant(x))['encodings'].numpy()
    self._model_loaded = True

  def _encode_batch(
      self,
      sound_batch: Sequence[types.Sound],
      **kwargs: Any,
  ) -> Sequence[types.SoundEmbedding]:
    """Encodes the transcript truth and Gecko embeddings.

    Args:
      sound_batch: A sequence of types.Sound objects to encode.
      **kwargs: Any additional parameters required for encoding.

    Returns:
      A list of types.SoundEmbedding objects, one for each input.
    """
    del kwargs  # Unused.

    prompts = []
    for sound in sound_batch:
      params = sound.context
      if hasattr(params, 'title') and params.title is not None:
        title = params.title
      else:
        title = 'None'
      if params.text is None:
        raise ValueError('Text is required for encoding.')
      prompts.append(self.prompt_template.format(text=params.text, title=title))

    assert self.transcript_truths_encode_fn is not None
    embeddings = self.transcript_truths_encode_fn(prompts)

    outputs = []
    for sound, embedding in zip(sound_batch, embeddings):
      params = sound.context
      timestamp = np.array(
          [[params.waveform_start_second, params.waveform_end_second]]
      )
      outputs.append(
          types.SoundEmbedding(
              embedding=embedding[np.newaxis],
              timestamps=timestamp,
              context=params,
          )
      )

    return outputs
