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

"""Raw Sound Encoder."""

from typing import Callable, Sequence, Union, Any, Tuple

import librosa
from mseb import encoder
import numpy as np


def hanning_window_transform(x: np.ndarray) -> np.ndarray:
  """Applies hanning window to input sequence of frames.

  Args:
    x: Sequence of frames of shape (num_frames, frame_length).

  Returns:
    The input array with the Hanning window applied to each frame.

  Raises:
    If input array is not 2D or first dimension is not 1.
  """
  if x.ndim != 2:
    raise ValueError(
        f'Input must be a 2D array. Received shape: {x.shape}.'
    )
  return x * np.hanning(x.shape[-1])


def spectrogram_transform(x: np.ndarray) -> np.ndarray:
  """Extracts spectrogram from input frame sequence.

  Args:
    x: Sequence of frames of shape (num_frames, frame_length).

  Returns:
    Spectrogram of the input frames, with shape (num_frames, fft_bins).

  Raises:
    If input array is not 2D or first dimension is not 1.
  """
  if x.ndim != 2:
    raise ValueError(
        f'Input must be a 2D array. Received shape: {x.shape}.'
    )
  frame_length = x.shape[-1]
  window = np.hanning(frame_length)
  frames = x * window
  fft_length = int(2 ** np.ceil(np.log2(frame_length)))
  frames = np.fft.rfft(frames, n=fft_length)
  frames = np.abs(frames)
  frames = np.square(frames)
  return frames


def _hertz_to_mel(frequencies_hertz: float | int) -> np.ndarray:
  """Converts hertz to mel."""
  return 1127.0 * np.log(1.0 + (frequencies_hertz / 700.0))


def linear_to_mel_weight_matrix(
    num_mel_bins: int,
    num_spectrogram_bins: int,
    sample_rate: int | float,
    lower_edge_hertz: int | float,
    upper_edge_hertz: int | float,
    dtype: Any = np.float32,
) -> np.ndarray:
  r"""NumPy-port of `tf.signal.linear_to_mel_weight_matrix`.

  Note that this function works purely on numpy because mel-weights are
  shape-dependent constants that usually should not be computed in an
  accelerators.

  Args:
    num_mel_bins: How many bands in the resulting mel spectrum.
    num_spectrogram_bins: How many bins there are in the source spectrogram
      data, which is understood to be `fft_size // 2 + 1`, i.e. the spectrogram
      only contains the nonredundant FFT bins.
    sample_rate: Samples per second of the input signal used to create the
      spectrogram. Used to figure out the frequencies corresponding to each
      spectrogram bin, which dictates how they are mapped into the mel scale.
    lower_edge_hertz: Lower bound on the frequencies to be included in the mel
      spectrum. This corresponds to the lower edge of the lowest triangular
      band.
    upper_edge_hertz: The desired top edge of the highest frequency band.
    dtype: The `DType` of the result matrix.

  Returns:
    An array of shape `[num_spectrogram_bins, num_mel_bins]`.
  Raises:
    ValueError: If `num_mel_bins`/`num_spectrogram_bins`/`sample_rate` are not
      positive, `lower_edge_hertz` is negative, frequency edges are incorrectly
      ordered, `upper_edge_hertz` is larger than the Nyquist frequency.
  [mel]: https://en.wikipedia.org/wiki/Mel_scale
  """

  # Input validator from tensorflow/python/ops/signal/mel_ops.py#L71
  if num_mel_bins <= 0:
    raise ValueError('num_mel_bins must be positive. Got: %s' % num_mel_bins)
  if lower_edge_hertz < 0.0:
    raise ValueError(
        'lower_edge_hertz must be non-negative. Got: %s' % lower_edge_hertz
    )
  if lower_edge_hertz >= upper_edge_hertz:
    raise ValueError(
        'lower_edge_hertz %.1f >= upper_edge_hertz %.1f'
        % (lower_edge_hertz, upper_edge_hertz)
    )
  if sample_rate <= 0.0:
    raise ValueError('sample_rate must be positive. Got: %s' % sample_rate)
  if upper_edge_hertz > sample_rate / 2:
    raise ValueError(
        'upper_edge_hertz must not be larger than the Nyquist '
        'frequency (sample_rate / 2). Got %s for sample_rate: %s'
        % (upper_edge_hertz, sample_rate)
    )

  # For better precision, we internally use float64.  It will not slow down
  # feature extraction because this function is called only once for obtaining
  # a constant matrix.
  internal_dtype = np.float64

  # HTK excludes the spectrogram DC bin.
  bands_to_zero = 1
  nyquist_hertz = sample_rate / 2.0
  linear_frequencies = np.linspace(
      0.0, nyquist_hertz, num_spectrogram_bins, dtype=internal_dtype
  )[bands_to_zero:]
  spectrogram_bins_mel = _hertz_to_mel(linear_frequencies)[:, np.newaxis]

  # Compute num_mel_bins triples of (lower_edge, center, upper_edge). The
  # center of each band is the lower and upper edge of the adjacent bands.
  # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
  # num_mel_bins + 2 pieces.
  edges = np.linspace(
      _hertz_to_mel(lower_edge_hertz),
      _hertz_to_mel(upper_edge_hertz),
      num_mel_bins + 2,
      dtype=internal_dtype,
  )

  # Split the triples up and reshape them into [1, num_mel_bins] tensors.
  lower_edge_mel, center_mel, upper_edge_mel = (
      edges[:-2][np.newaxis, :],
      edges[1:-1][np.newaxis, :],
      edges[2:][np.newaxis, :],
  )

  # Calculate lower and upper slopes for every spectrogram bin.
  # Line segments are linear in the mel domain, not Hertz.
  lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (
      center_mel - lower_edge_mel
  )
  upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (
      upper_edge_mel - center_mel
  )

  # Intersect the line segments with each other and zero.
  mel_weights_matrix = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))

  # Re-add the zeroed lower bins we sliced out above.
  return np.pad(mel_weights_matrix, [[bands_to_zero, 0], [0, 0]]).astype(dtype)


def log_mel_transform(x: np.ndarray,
                      mel_matrix: np.ndarray,
                      power_floor: float = 1e-10
                      ) -> np.ndarray:
  """Applies log mel transformation to spectrograms.

  Args:
    x: Sequence of frames of shape (num_frames, frame_length).
    mel_matrix: Mel weight matrix of shape (fft_bins, num_mel_bins).
    power_floor: Minimum value of the Mel spectrum power before taking the log.

  Returns:
    Log-Mel spectrograms of shape (num_frames, num_mel_bins).

  Raises:
    ValueError: If input shapes are not compatible.
  """
  if x.ndim != 2:
    raise ValueError(
        f'Input must be a 2D array. Received shape: {x.shape}.'
    )
  frame_length = x.shape[-1]
  fft_length = int(2 ** np.ceil(np.log2(frame_length)))
  window = np.hanning(frame_length)
  frames = x.astype(mel_matrix.dtype)
  frames = frames * window
  frames = np.fft.rfft(frames, n=fft_length)
  frames = np.abs(frames)
  frames = np.square(frames)
  if mel_matrix.ndim != 2 or mel_matrix.shape[0] != frames.shape[-1]:
    raise ValueError(f'The mel matrix shape of {mel_matrix.shape} '
                     f'is not compatible with spectrogram shape {frames.shape}.'
                     ' The mel_matrix.shape[0] must match x.shape[-1].')
  mels = np.matmul(frames, mel_matrix)
  mels = np.maximum(mels, power_floor)
  log_mels = np.log(mels)
  return log_mels


class RawEncoder(encoder.Encoder):
  """Minimal sound encoder.

  Encodes sound into sequence of fixed-size frames with a frame-stride.
  Applies an optional transform or pooling function.

  """

  def __init__(self,
               transform_fn: Callable[..., np.ndarray],
               pooling: str | None = None):
    """Initialises raw encoder.

    Args:
      transform_fn: A transformation applied to each frame (e.g.,
                    spectrogram_transform, log_mel_transform). This function
                    should accept a NumPy array of frames and any additional
                    keyword arguments.
      pooling: The pooling strategy to apply to the transformed frames.
               Options are 'last', 'mean', 'max', or None for no pooling.
    """
    self.transform_fn = transform_fn
    self.pool_fn: Callable[[np.ndarray], np.ndarray]
    if pooling == 'last':
      self.pool_fn = lambda x: x[-1][None, :]
    elif pooling == 'mean':
      self.pool_fn = lambda x: np.mean(x, axis=0, keepdims=True)
    elif pooling == 'max':
      self.pool_fn = lambda x: np.max(x, axis=0, keepdims=True)
    elif pooling is None:
      self.pool_fn = lambda x: x
    else:
      raise ValueError(f'Unknown pooling strategy: {pooling}. Supported are '
                       'last, mean, max, or None.')
    self.pooling = pooling

  def encode(self,
             sequence: Union[str, Sequence[float]],
             context: encoder.ContextParams,
             transform_fn_kwargs: dict[str, Any] | None = None,
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Encodes sound to frames of embeddings and optionally pools them into one.

    Args:
      sequence: Input audio, either as a file path (str) or a sequence of audio
                samples (list or np.ndarray).
      context: An object containing context parameters like sample_rate,
               frame_length, and frame_step.
      transform_fn_kwargs: A dictionary of keyword arguments to pass to the
                           transform_fn.

    Returns:
      A tuple containing:
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
    if not context.frame_length:
      raise ValueError('Frame length should be set in context input.')
    if not context.frame_step:
      raise ValueError('Frame step should be set in context input.')

    frame_length = context.frame_length
    frame_step = context.frame_step
    num_frames = np.maximum(0, len(sequence) - frame_length + frame_step)
    num_frames = num_frames  // frame_step
    frames = np.zeros([num_frames, frame_length])
    timestamps = []
    for i in range(num_frames):
      start = i * frame_step
      end = start + frame_length
      frames[i] = sequence[start : end]
      timestamps.append([start / context.sample_rate,
                         end / context.sample_rate])
    embeddings = self.transform_fn(frames, **transform_fn_kwargs)
    embeddings = self.pool_fn(embeddings)
    if self.pooling is not None:
      timestamps = [[0, len(sequence) / context.sample_rate]]
    return np.array(timestamps), embeddings
