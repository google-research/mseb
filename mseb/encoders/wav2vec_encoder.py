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

"""Wav2Vec Encoder."""

from typing import Any, Callable, Sequence

import huggingface_hub
from mseb import encoder
from mseb import types
import numpy as np
from requests import exceptions
import torch
import transformers

HTTPError = exceptions.HTTPError


def normalize_embeddings(x: np.ndarray) -> np.ndarray:
  """Normalize embedding vectors to length 1."""
  norms = np.linalg.norm(x, axis=1, keepdims=True)
  normalized_vectors = np.copy(x).astype(np.float32)
  non_zero_norms_mask = norms.flatten() != 0
  if np.any(non_zero_norms_mask):
    normalized_vectors[non_zero_norms_mask] /= norms[non_zero_norms_mask]
  return normalized_vectors


class Wav2VecEncoder(encoder.MultiModalEncoder):
  """A class to embed audio into pooled embedding of Wav2Vec model."""

  def __init__(
      self,
      model_path: str,
      transform_fn: Callable[..., np.ndarray],
      device: str | None,
      pooling: str = 'mean',
      **kwargs: Any,
  ):
    """Initializes the Wav2Vec2FeatureExtractor.

    Args:
      model_path: The name of the pre-trained Wav2Vec2 model from Hugging Face
        Model Hub. Raises ValueError if the model is not found.
      transform_fn: A transformation applied to each embedding.
      device: The device to load the model onto. - cuda: Force GPU (will raise
        error if no CUDA device). - cpu: Force CPU. - None: Automatically
        chooses cuda if available, else cpu.
      pooling: The pooling strategy to apply to the transformed frames. Options
        are 'last', 'mean', 'max', and default is 'mean'.
      **kwargs: Any additional parameters required for encoding.

    Raises:
      RuntimeError: If model loading fails.
    """
    super().__init__()
    self.model_path = model_path
    self._kwargs = kwargs

    if device:
      if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA device requested but not available. '
            'Please check your GPU setup.'
        )
      self.device = torch.device(device)
    else:
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.processor = None
    self.model = None

    self.transform_fn = transform_fn
    self.pool_fn: Callable[[np.ndarray], np.ndarray]
    if pooling == 'last':
      self.pool_fn = lambda x: x[-1][None, :]
    elif pooling == 'mean':
      self.pool_fn = lambda x: np.mean(x, axis=0, keepdims=True)
    elif pooling == 'max':
      self.pool_fn = lambda x: np.max(x, axis=0, keepdims=True)
    else:
      raise ValueError(
          f'Unknown pooling strategy: {pooling}. Supported are '
          'last, mean, max, or None.'
      )
    self.pooling = pooling

  def _check_input_types(
      self, batch: Sequence[types.MultiModalObject]
  ) -> None:
    if not all(isinstance(x, types.Sound) for x in batch):
      raise ValueError(
          'Wav2VecEncoder only supports a batch of all Sound '
          'inputs.'
      )

  def _setup(self):
    """Loads the Wav2Vec2 model.

    Raises:
      RuntimeError: If model loading fails.
    """
    api = huggingface_hub.HfApi()
    try:
      api.model_info(self.model_path)
    except HTTPError as e:
      if isinstance(e, HTTPError) and e.response.status_code == 404:
        raise ValueError(
            f'Model {self.model_path} not found on Hugging Face Hub. '
            'Please check the model name for typos or verify its '
            'existence on huggingface.co/models.'
        ) from e
      else:
        raise RuntimeError(
            'An error occurred while trying to access model '
            f'{self.model_path}. Please check your network '
            'connection or the model name.'
        ) from e
    except Exception as e:
      raise RuntimeError(
          f'An unexpected error occurred during model availability check: {e}'
      ) from e
    try:
      self.processor = transformers.Wav2Vec2Processor.from_pretrained(
          self.model_path
      )
    except OSError as e:
      raise RuntimeError(
          'Failed to load Wav2Vec2Processor for '
          f'{self.model_path}: {e}. This might indicate a '
          'corrupted download or an issue with the model files.'
      ) from e
    try:
      self.model = transformers.Wav2Vec2Model.from_pretrained(self.model_path)
      self.model.eval()  # Set model to evaluation mode for inference
      self.model.to(self.device)
    except OSError as e:
      raise RuntimeError(
          'Failed to load Wav2Vec2Model for '
          f'{self.model_path}: {e}. This might indicate a '
          'corrupted download or an issue with the model files.'
      ) from e
    except RuntimeError as e:
      raise RuntimeError(
          f'Failed to move model to device {self.device}: {e}. '
          'Check your GPU setup.'
      ) from e

  def _encode(
      self,
      batch: Sequence[types.MultiModalObject],
  ) -> Sequence[types.SoundEmbedding]:
    """Encodes speech by Wav2Vec encoder activations and optionally pools them.

    Args:
      batch: A sequence of types.Sound objects to encode.

    Returns:
      A list of tuples, one for each input, each tuple containing:
        - waveform_embeddings (np.ndarray): A 2D array of shape
          (1, embedding_dim).
        - embedding_timestamps (np.ndarray): A 2D array of shape (m, 1),
          where the first row is the [start, end] pair indicating the segment by
          sound waveform index.

    Raises:
      ValueError: If required context parameters are not set or input sequence
                  is invalid.
      FileNotFoundError: If the audio file path is not found.
    """
    sound_batch: list[types.Sound] = []
    for example in batch:
      assert isinstance(example, types.Sound)
      sound_batch.append(example)

    transform_fn_kwargs = self._kwargs.get('transform_fn_kwargs', {})
    outputs = []
    for sound in sound_batch:
      assert self.processor is not None
      sound = encoder.resample_sound(
          sound, self.processor.feature_extractor.sampling_rate
      )
      waveform = np.asarray(sound.waveform, dtype=np.float32)
      params = sound.context

      input_values = self.processor(
          waveform,
          sampling_rate=params.sample_rate,
          return_tensors='pt',
      ).input_values
      input_values = input_values.to(self.device)

      assert self.model is not None
      with torch.no_grad():
        model_outputs = self.model(input_values)
        embeddings = model_outputs.last_hidden_state.squeeze(0)
        embeddings = embeddings.to('cpu').numpy()

      embeddings = self.transform_fn(embeddings, **transform_fn_kwargs)
      embedding = self.pool_fn(embeddings)
      timestamps = np.array([[0, params.length / params.sample_rate]])
      outputs.append(
          types.SoundEmbedding(
              embedding=embedding, timestamps=timestamps, context=params
          )
      )
    return outputs
