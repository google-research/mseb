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

"""SoundStream Encoder."""

from typing import Any, List, Optional, Sequence, Union

import librosa
from mseb import encoder
from mseb import types
import numpy as np
import requests
import tensorflow as tf


class FetchError(RuntimeError):
  """Custom exception for errors during file fetching."""
  pass


def fetch_lyra_component_bytes(tflite_filename: str) -> bytes:
  """Fetches a specified .tflite component from Lyra github repository.

  Args:
    tflite_filename: The filename of the .tflite component to fetch
                     (e.g., "soundstream_encoder.tflite").

  Returns:
    bytes: The content of the .tflite file as bytes if successful.

  Raises:
    FetchError: If any error occurs during the fetching process (e.g.,
                network issue, HTTP error, or unexpected error).
  """
  base_url = 'https://raw.githubusercontent.com/google/lyra/main/lyra/model_coeffs'
  file_url = f'{base_url}/{tflite_filename}'
  try:
    print(f'Attempting to fetch content from {file_url}...')
    response = requests.get(file_url)
    response.raise_for_status()
    file_content_bytes = response.content
    print(f'Successfully fetched {len(file_content_bytes)} '
          f'bytes from {file_url}')
    return file_content_bytes
  except requests.exceptions.HTTPError as http_err:
    error_message = (
        f'Failed to fetch resource from {file_url} due to HTTP error: '
        f'{http_err.response.status_code} {http_err.response.reason}'
    )
    raise FetchError(error_message) from http_err

  except requests.exceptions.ConnectionError as conn_err:
    raise FetchError(f'Connection error occurred while trying to '
                     f'reach {file_url}: {conn_err}') from conn_err
  except requests.exceptions.Timeout as timeout_err:
    raise FetchError(f'Timeout error occurred while trying to reach '
                     f'{file_url}: {timeout_err}') from timeout_err
  except requests.exceptions.RequestException as req_err:
    raise FetchError(f'An error occurred during the request to '
                     f'{file_url}: {req_err}') from req_err
  except Exception as e:
    raise FetchError(f'An unexpected error occurred while fetching '
                     f'{file_url}: {e}') from e


def pad_first_axis_to_target_multiple_length(arr: np.ndarray,
                                             length_base: int,
                                             pad_value: Any = 0
                                             ) -> np.ndarray:
  """Pads a NumPy array axis so that its length is a multiple of length_base.

  The padding is along first axis. If the array's first axis length is already
  a multiple of length_base, the original array is returned.

  Args:
    arr: The input NumPy array.
    length_base: The number that the first axis length should be a
                multiple of.
    pad_value: The value to use for padding.

  Returns:
    arr or padded_arr
    The padded array, or the original array if no padding was needed.

  Raises:
    ValueError: If length_base is not a positive integer.
  """
  if not isinstance(length_base, int) or length_base <= 0:
    raise ValueError(f'{length_base=} must be a positive integer.')

  current_length = arr.shape[0]
  remainder = current_length % length_base

  if remainder == 0:
    return arr
  else:
    num_to_pad = length_base - remainder
    pad_width = [(0, num_to_pad)] + [(0, 0)] * (arr.ndim - 1)
    padded_arr = np.pad(arr, pad_width, mode='constant',
                        constant_values=pad_value)
    return padded_arr


class SoundStreamEncoder(encoder.MultiModalEncoder):
  """Encodes audio using SoundStream  model from lyra github repository."""

  SOUNDSTREAM_SAMPLING_RATE: int = 16000
  SOUNDSTREAM_NBPQ: int = 4

  def __init__(self,
               model_path: str = 'soundstream_encoder.tflite',
               bits_per_second: Optional[int] = None,
               quantize: bool = False,
               quantizer_model_filename: Optional[str] = 'quantizer.tflite'):
    """Initializes the SoundStreamEncoder.

    Args:
      model_path: Filename of the SoundStream encoder TFLite model.
      bits_per_second: Target bitrate for quantization.
                       Required if quantize is True.
      quantize: Whether to apply quantization to the embeddings.
      quantizer_model_filename: Filename of the quantizer TFLite model.
                                Required if quantize is True.
    Raises:
      RuntimeError: If model components cannot be fetched, the TFLite encoder
                    fails to initialize, or the loaded model is found to be
                    invalid (e.g., missing input details).
      ValueError: If `quantize` is True and `bits_per_second` or
                  `quantizer_model_filename` is not appropriately provided
                  (This part of your init logic is not shown in the snippet,
                  but I'm adding it as it's a common pattern and good to
                  document if present in your full code).
    """
    super().__init__()
    self.encoder_model_filename = model_path
    self.encoder_interpreter = None

    self.bits_per_second = bits_per_second
    self.quantize = quantize
    self.quantizer_model_filename = quantizer_model_filename
    self.quantizer_runner = None
    self.num_desired_quantizers = 0
    self.sample_size: int = -1
    self._encoder_input_tensor_idx: int = -1
    self._encoder_output_tensor_idx: int = -1

  def _check_input_types(
      self, batch: Sequence[types.MultiModalInput]
  ) -> None:
    if not all(isinstance(x, types.Sound) for x in batch):
      raise ValueError(
          'SoundStreamEncoder only supports a batch of all Sound '
          'inputs.'
      )

  def _setup(self):
    try:
      encoder_bytes = fetch_lyra_component_bytes(self.encoder_model_filename)
      self.encoder_interpreter = tf.lite.Interpreter(
          model_content=encoder_bytes)
      self.encoder_interpreter.allocate_tensors()
    except FetchError as e:
      raise RuntimeError(
          'Failed to initialize SoundStream encoder: '
          f'Could not fetch {self.encoder_model_filename}'
      ) from e
    except Exception as e:
      raise RuntimeError(
          'Failed to initialize SoundStream TFLite encoder '
          f'from {self.encoder_model_filename}'
      ) from e
    encoder_input_details = self.encoder_interpreter.get_input_details()
    encoder_output_details = self.encoder_interpreter.get_output_details()

    if not encoder_input_details:
      raise RuntimeError(
          'No input details found for encoder model'
          f' {self.encoder_model_filename}'
      )
    if not encoder_output_details:
      raise RuntimeError(
          'No output details found for encoder model'
          f' {self.encoder_model_filename}'
      )

    self.sample_size = int(encoder_input_details[0]['shape'][1])
    self._encoder_input_tensor_idx = encoder_input_details[0]['index']
    self._encoder_output_tensor_idx = encoder_output_details[0]['index']

    if self.quantize:
      if self.bits_per_second is None:
        raise ValueError(
            'bits_per_second must be provided if quantize is True.')
      if self.quantizer_model_filename is None:
        raise ValueError(
            'quantizer_model_filename must be provided if quantize is True.')
      try:
        quantizer_bytes = fetch_lyra_component_bytes(
            self.quantizer_model_filename
        )
        quantizer_interpreter = tf.lite.Interpreter(
            model_content=quantizer_bytes)
        quantizer_interpreter.allocate_tensors()
        self.quantizer_runner = quantizer_interpreter.get_signature_runner(
            'encode')
      except FetchError as e:
        raise RuntimeError(
            'Failed to initialize quantizer: Could not fetch '
            f'{self.quantizer_model_filename}') from e
      except Exception as e:
        raise RuntimeError(
            'Failed to initialize TFLite quantizer from '
            f'{self.quantizer_model_filename} or get signature') from e

      quantizer_output_sig_details = self.quantizer_runner.get_output_details()
      if 'output_0' not in quantizer_output_sig_details:
        raise RuntimeError(
            'Quantizer signature does not contain expected output_0.')
      # Total available quantizers from the model's output shape
      num_total_quantizers_from_model: int = quantizer_output_sig_details[
          'output_0']['shape'][0]
      if self.sample_size == 0:
        raise ValueError(
            'Encoder sample_size cannot be zero for frame rate calculation.')
      frame_rate: float = self.SOUNDSTREAM_SAMPLING_RATE / self.sample_size
      if frame_rate == 0:
        raise ValueError(
            'Frame rate cannot be zero for bitrate calculation.')
      bits_per_frame: float = self.bits_per_second / frame_rate
      self.num_desired_quantizers = int(np.floor(
          bits_per_frame / self.SOUNDSTREAM_NBPQ))
      self.num_desired_quantizers = np.minimum(
          num_total_quantizers_from_model, self.num_desired_quantizers)
      if self.num_desired_quantizers <= 0:
        # This might indicate a misconfiguration, for example
        # too low bitrate for the NBPQ, frame rate)
        print('Warning: num_desired_quantizers calculated to '
              f'{self.num_desired_quantizers}. '
              'Check bits_per_second, frame_rate, and NBPQ settings.')
        raise ValueError('Calculated num_desired_quantizers is non-positive.')
      self._num_quantizers_tf_tensor_input = tf.constant(
          [self.num_desired_quantizers], dtype=tf.int32)

  def _load_and_preprocess_audio(self,
                                 sequence: Union[str, Sequence[float]],
                                 source_sample_rate: int,
                                 target_sample_rate: int) -> np.ndarray:
    """Loads audio if path is given, then resamples and pads."""
    if isinstance(sequence, str):
      try:
        waveform, sample_rate = librosa.load(sequence, sr=None)
        if sample_rate != source_sample_rate:
          print(f'Warning: Actual audio sampling rate ({sample_rate} Hz) '
                'differs from provided source_sample_rate '
                f'({source_sample_rate} Hz). Using actual SR for resampling '
                'if different from target.')
          if sample_rate != target_sample_rate:
            waveform = librosa.resample(
                waveform,
                orig_sr=sample_rate,
                target_sr=target_sample_rate)
      except Exception as e:
        raise ValueError(
            f'Error loading audio from path: {sequence}'
        ) from e
    elif isinstance(sequence, np.ndarray):
      waveform = sequence
      if source_sample_rate != target_sample_rate:
        waveform = librosa.resample(
            waveform,
            orig_sr=source_sample_rate,
            target_sr=target_sample_rate)
    else:
      raise TypeError('Input sequence must be a file path (str) '
                      'or a NumPy array.')

    if waveform.dtype != np.float32:
      waveform = waveform.astype(np.float32)
    return pad_first_axis_to_target_multiple_length(
        waveform, length_base=self.sample_size, pad_value=0.0
    )

  def _encode(
      self, batch: Sequence[types.MultiModalInput]
  ) -> Sequence[types.SoundEmbedding]:
    """Encodes an audio sequence into embeddings.

    Args:
      batch: A sequence of sound sources to encode.

    Returns:
      A sequence of types.SoundEmbedding objects, one for each input:
        timestamps: Start and end times for each embedding frame.
        embeddings: The resulting (potentially quantized)embeddings.
    """
    sound_batch: list[types.Sound] = []
    for example in batch:
      assert isinstance(example, types.Sound)
      sound_batch.append(example)
    outputs = []
    for sound in sound_batch:
      waveform = self._load_and_preprocess_audio(
          sound.waveform,
          sound.context.sample_rate,
          self.SOUNDSTREAM_SAMPLING_RATE
      )

      timestamps_list: List[List[float]] = []
      embeddings_list: List[np.ndarray] = []

      num_frames = len(waveform) // self.sample_size
      for i in range(num_frames):
        start_sample = i * self.sample_size
        end_sample = start_sample + self.sample_size
        timestamps_list.append([
            start_sample / self.SOUNDSTREAM_SAMPLING_RATE,
            end_sample / self.SOUNDSTREAM_SAMPLING_RATE
        ])
        audio_segment = waveform[start_sample:end_sample]
        encoder_input_data = np.expand_dims(
            audio_segment, axis=0).astype(np.float32)

        assert self.encoder_interpreter is not None
        self.encoder_interpreter.set_tensor(
            self._encoder_input_tensor_idx, encoder_input_data)
        self.encoder_interpreter.invoke()
        raw_embedding = self.encoder_interpreter.get_tensor(
            self._encoder_output_tensor_idx)

        if self.quantize and self.quantizer_runner:
          features_for_quantizer = tf.constant(raw_embedding, dtype=tf.float32)
          quantizer_output_dict = self.quantizer_runner(
              input_frames=features_for_quantizer,
              num_quantizers=self._num_quantizers_tf_tensor_input
          )
          quantized_codes_tensor = quantizer_output_dict['output_0']

          if hasattr(quantized_codes_tensor, 'numpy'):
            quantized_codes = quantized_codes_tensor.numpy()
          else:
            quantized_codes = quantized_codes_tensor
          selected_quantized_embedding = quantized_codes[
              :self.num_desired_quantizers, 0, 0]
          embeddings_list.append(selected_quantized_embedding)
        else:
          embedding_vector = raw_embedding[0, 0, :]
          embeddings_list.append(embedding_vector)
      if not embeddings_list:
        continue
      timestamps = np.array(timestamps_list, dtype=np.float32)
      embeddings = np.array(embeddings_list, dtype=np.float32)
      outputs.append(
          types.SoundEmbedding(
              embedding=embeddings, timestamps=timestamps, context=sound.context
          )
      )
    return outputs
