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

"""Perch Encoder."""

from typing import Sequence

import librosa
from mseb import encoder
from mseb import types
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

PERCH_SAMPLE_RATE = 32000
PERCH_WINDOW_LEN_SAMPLES = 5 * PERCH_SAMPLE_RATE
DEFAULT_MODEL_PATH = 'https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1'


class PerchEncoder(encoder.MultiModalEncoder):
  """Embed audio using Perch model."""

  def __init__(
      self,
      model_path: str | None = None,
      embedding_type: str = 'embedding',
  ):
    """Initializes the PerchEncoder.

    Args:
      model_path: The TFHub path to Perch embedding model.
      embedding_type: str, 'embedding' or 'logits'.

    Raises:
      RuntimeError: If model loading fails.
    """
    super().__init__()
    if model_path is None:
      model_path = DEFAULT_MODEL_PATH
    self.model_path = model_path
    self.model = None
    assert embedding_type in ('embedding', 'logits')
    self.embedding_type = embedding_type

  def _check_input_types(
      self, batch: Sequence[types.MultiModalObject]
  ) -> None:
    if not all(isinstance(x, types.Sound) for x in batch):
      raise ValueError(
          'PerchEncoder only supports a batch of all Sound '
          'inputs.'
      )

  def _setup(self):
    """Loads the Perch model."""
    self.model = hub.load(self.model_path)

  def _encode(
      self,
      batch: Sequence[types.MultiModalObject],
  ) -> Sequence[types.SoundEmbedding]:
    assert self.model is not None, 'Call setup() before encode().'
    sound_batch: list[types.Sound] = []
    for example in batch:
      assert isinstance(example, types.Sound)
      sound_batch.append(example)

    embeddings = []
    for sound in sound_batch:
      waveform = np.asarray(sound.waveform, dtype=np.float32)
      params = sound.context

      assert params.sample_rate

      if params.sample_rate != PERCH_SAMPLE_RATE:
        waveform = librosa.resample(
            waveform,
            orig_sr=params.sample_rate,
            target_sr=PERCH_SAMPLE_RATE,
        )

      if waveform.size == 0:
        n_frames = 1
      else:
        n_frames = int(np.ceil(waveform.size / PERCH_WINDOW_LEN_SAMPLES))
      pad_len = n_frames * PERCH_WINDOW_LEN_SAMPLES - waveform.size
      padded_waveform = np.pad(waveform, (0, pad_len))
      segments = np.split(padded_waveform, n_frames)
      # Initially support only native chunk size.
      if len(segments) != 1:
        raise ValueError(
            'PerchEncoder only supports a single segment of 5sec. Got'
            f' {len(segments)}.'
        )
      waveform = segments[0]
      waveform_t = tf.constant(waveform, dtype=tf.float32)[tf.newaxis, :]
      outputs = self.model.signatures['serving_default'](inputs=waveform_t)
      key = 'label' if self.embedding_type == 'logits' else 'embedding'
      embedding = outputs[key][0].numpy()

      timestamps = np.array([[0, params.waveform_end_second]])
      embeddings.append(
          types.SoundEmbedding(
              embedding=embedding, timestamps=timestamps, context=params
          )
      )
    return embeddings
