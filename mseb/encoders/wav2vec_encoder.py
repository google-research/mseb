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

from typing import Any, Callable, Sequence, Union, Tuple

import huggingface_hub
import librosa
from mseb import encoder
import numpy as np
from requests.exceptions import HTTPError
import torch
import transformers


def normalize_embeddings(x: np.ndarray) -> np.ndarray:
  """Normalize embedding vectors to length 1."""
  norms = np.linalg.norm(x, axis=1, keepdims=True)
  normalized_vectors = np.copy(x).astype(np.float32)
  non_zero_norms_mask = norms.flatten() != 0
  if np.any(non_zero_norms_mask):
    normalized_vectors[non_zero_norms_mask] /= norms[non_zero_norms_mask]
  return normalized_vectors


class Wav2VecEncoder(encoder.Encoder):
  """A class to embed audio into pooled embedding of Wav2Vec model."""

  def __init__(self,
               model_name: str,
               transform_fn: Callable[..., np.ndarray],
               device: str | None,
               pooling: str = 'mean'):
    """Initializes the Wav2Vec2FeatureExtractor.

    Args:
      model_name: The name of the pre-trained Wav2Vec2 model from Hugging Face
                  Model Hub. Raises ValueError if the model is not found.
      transform_fn: A transformation applied to each embedding.
      device: The device to load the model onto.
              - cuda: Force GPU (will raise error if no CUDA device).
              - cpu: Force CPU.
              - None: Automatically chooses cuda if available, else cpu.
      pooling: The pooling strategy to apply to the transformed frames.
               Options are 'last', 'mean', 'max', and default is 'mean'.
    Raises:
      ValueError: If model name or pooling is not supported.
      RuntimeError: If model loading fails.
    """
    if device:
      if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA device requested but not available. '
                           'Please check your GPU setup.')
      self.device = torch.device(device)
    else:
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    api = huggingface_hub.HfApi()
    try:
      api.model_info(model_name)
    except (HTTPError, huggingface_hub.HfHubPyError) as e:
      if isinstance(e, HTTPError) and e.response.status_code == 404:
        raise ValueError(
            f'Model {model_name} not found on Hugging Face Hub. '
            'Please check the model name for typos or verify its '
            'existence on huggingface.co/models.'
        ) from e
      else:
        raise RuntimeError(
            'An error occurred while trying to access model '
            f'{model_name}. Please check your network '
            'connection or the model name.'
        ) from e
    except Exception as e:
      raise RuntimeError('An unexpected error occurred during model '
                         f'availability check: {e}'
                         ) from e
    try:
      self.processor = transformers.Wav2Vec2Processor.from_pretrained(
          model_name)
    except OSError as e:
      raise RuntimeError('Failed to load Wav2Vec2Processor for '
                         f'{model_name}: {e}. This might indicate a '
                         'corrupted download or an issue with the model files.'
                         ) from e
    try:
      self.model = transformers.Wav2Vec2Model.from_pretrained(model_name)
      self.model.eval()  # Set model to evaluation mode for inference
      self.model.to(self.device)
    except OSError as e:
      raise RuntimeError('Failed to load Wav2Vec2Model for '
                         f'{model_name}: {e}. This might indicate a '
                         'corrupted download or an issue with the model files.'
                         ) from e
    except RuntimeError as e:
      raise RuntimeError(f'Failed to move model to device {self.device}: {e}. '
                         'Check your GPU setup.'
                         ) from e
    self.transform_fn = transform_fn
    self.pool_fn: Callable[[np.ndarray], np.ndarray]
    if pooling == 'last':
      self.pool_fn = lambda x: x[-1][None, :]
    elif pooling == 'mean':
      self.pool_fn = lambda x: np.mean(x, axis=0, keepdims=True)
    elif pooling == 'max':
      self.pool_fn = lambda x: np.max(x, axis=0, keepdims=True)
    else:
      raise ValueError(f'Unknown pooling strategy: {pooling}. Supported are '
                       'last, mean, max, or None.')
    self.pooling = pooling

  def encode(self,
             sequence: Union[str, Sequence[float]],
             context: encoder.ContextParams,
             transform_fn_kwargs: dict[str, Any] | None = None,
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Encodes speech by Wav2Vec encoder activations and optionally pools them.

    Args:
      sequence: Input audio, either as a file path (str) or a sequence of audio
                samples (list or np.ndarray).
      context: An object containing context parameters like sample_rate.
      transform_fn_kwargs: A dictionary of keyword arguments to pass to the
                           transform_fn.

    Returns:
      A tuple (timestamp, embedding).
      timestamps: A NumPy array of shape (num_embeddings, 2) where each row is
                  [start_time, end_time]. If pooling is applied, this will be
                  [[0, total_duration]].
      embeddings: A NumPy array of the extracted embeddings.

    Raises:
      ValueError: If required context parameters are not set or input sequence
                  is invalid.
      FileNotFoundError: If the audio file path is not found.
    """
    if transform_fn_kwargs is None:
      transform_fn_kwargs = {}

    if isinstance(sequence, str):
      try:
        audio_sequence, sample_rate = librosa.load(sequence, sr=None)
      except FileNotFoundError as exc:
        raise FileNotFoundError(f'Audio file not found: {sequence}') from exc
      if context.sample_rate is None:
        context.sample_rate = sample_rate
      elif context.sample_rate != sample_rate:
        print(f'Warning: context.sample_rate ({context.sample_rate}) differs '
              'from audio file sample rate ({sample_rate}). Using file sample '
              'rate.')
        context.sample_rate = sample_rate
      sequence = audio_sequence
    elif not isinstance(sequence, np.ndarray):
      sequence = np.asarray(sequence, dtype=np.float32)

    if context.sample_rate is None:
      raise ValueError('Sample rate must be set in context when sequence is '
                       'a raw array or via librosa.load.')
    waveform = librosa.resample(
        sequence,
        orig_sr=context.sample_rate,
        target_sr=self.processor.feature_extractor.sampling_rate)

    input_values = self.processor(
        waveform,
        sampling_rate=self.processor.feature_extractor.sampling_rate,
        return_tensors='pt'
    ).input_values
    input_values = input_values.to(self.device)

    with torch.no_grad():
      outputs = self.model(input_values)
      embeddings = outputs.last_hidden_state.squeeze(0)
      embeddings = embeddings.to('cpu').numpy()

    embeddings = self.transform_fn(embeddings, **transform_fn_kwargs)
    embedding = self.pool_fn(embeddings)
    timestamp = np.array([[0, len(sequence) / context.sample_rate]])
    return timestamp, embedding
