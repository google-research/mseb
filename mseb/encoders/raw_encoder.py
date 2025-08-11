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

from typing import Any, Callable, Optional, Sequence

from mseb import encoder
from mseb import types
import numpy as np


def hanning_window_transform(x: np.ndarray) -> np.ndarray:
  """Applies hanning window to input sequence of frames.

  Args:
    x: Sequence of frames of shape (num_frames, frame_length).

  Returns:
    The input array with the Hanning window applied to each frame.

  Raises:
    If input array is not 2D.
  """
  if x.ndim != 2:
    raise ValueError(
        f"Input must be a 2D array. Received shape: {x.shape}."
    )
  return x * np.hanning(x.shape[-1])


def spectrogram_transform(x: np.ndarray) -> np.ndarray:
  """Extracts spectrogram from input frame sequence.

  Args:
    x: Sequence of frames of shape (num_frames, frame_length).

  Returns:
    Spectrogram of the input frames, with shape (num_frames, fft_bins).

  Raises:
    If input array is not 2D.
  """
  if x.ndim != 2:
    raise ValueError(
        f"Input must be a 2D array. Received shape: {x.shape}."
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
  if num_mel_bins <= 0:
    raise ValueError(
        "num_mel_bins must be positive. Got: %s" % num_mel_bins
    )
  if lower_edge_hertz < 0.0:
    raise ValueError(
        "lower_edge_hertz must be non-negative. Got: %s" % lower_edge_hertz
    )
  if lower_edge_hertz >= upper_edge_hertz:
    raise ValueError(
        "lower_edge_hertz %.1f >= upper_edge_hertz %.1f"
        % (lower_edge_hertz, upper_edge_hertz)
    )
  if sample_rate <= 0.0:
    raise ValueError(
        "sample_rate must be positive. Got: %s" % sample_rate
    )
  if upper_edge_hertz > sample_rate / 2:
    raise ValueError(
        "upper_edge_hertz must not be larger than the Nyquist "
        "frequency (sample_rate / 2). Got %s for sample_rate: %s"
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
        f"Input must be a 2D array. Received shape: {x.shape}."
    )
  power_spectrograms = spectrogram_transform(x).astype(mel_matrix.dtype)
  if (mel_matrix.ndim != 2 or
      mel_matrix.shape[0] != power_spectrograms.shape[-1]):
    raise ValueError(
        f"The mel matrix shape of {mel_matrix.shape} "
        f"is not compatible with spectrogram shape {power_spectrograms.shape}."
        " The mel_matrix.shape[0] must match x.shape[-1]."
    )
  mels = np.matmul(power_spectrograms, mel_matrix)
  mels = np.maximum(mels, power_floor)
  log_mels = np.log(mels)
  return log_mels


class RawEncoder(encoder.SoundEncoder):
  """Encodes raw audio into frames using a specified transform.

  This class implements the SoundEncoder interface to create feature frames
  from a raw waveform. It is configured by passing parameters like
  `frame_length`, `frame_step`, `transform_fn`, and `pooling`
  as keyword arguments during initialization.
  """

  def __init__(self, model_path: str = "raw_feature_encoder", **kwargs: Any):
    """Initializes the RawEncoder with its configuration.

    Note: This method is lightweight. All configuration is stored in
    `self._kwargs` and processed later in the `setup()` method.

    Args:
      model_path: A descriptive name for the encoder configuration. Not used
        for loading a model file but required by the base class.
      **kwargs: Configuration arguments. Expected keys include:
        - `frame_length` (int): The number of samples in each frame.
        - `frame_step` (int): The number of samples to advance between frames.
        - `transform_fn` (Callable): A function to apply to the batch of
          frames (e.g., `spectrogram_transform`).
        - `pooling` (Optional[str]): Pooling strategy ('mean', 'max', 'last')
          or None.
        - `transform_fn_kwargs` (Optional[dict]): Keyword arguments to
          pass to the `transform_fn`.
    """
    super().__init__(model_path, **kwargs)

    # Declare all attributes that will be set in setup()
    # This tells the type checker that these attributes will exist.
    self.frame_length: Optional[int] = None
    self.frame_step: Optional[int] = None
    self.transform_fn: Optional[Callable[..., np.ndarray]] = None
    self.pooling: Optional[str] = None
    self.transform_fn_kwargs: dict[str, Any] = {}
    self.pool_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None

  def setup(self):
    """Sets up the encoder by validating configuration and preparing functions.

    This method extracts configuration from `self._kwargs`, validates that
    required parameters are present, and sets up the pooling function.
    """
    self.frame_length = self._kwargs.get("frame_length")
    self.frame_step = self._kwargs.get("frame_step")
    self.transform_fn = self._kwargs.get("transform_fn")
    self.pooling = self._kwargs.get("pooling")
    self.transform_fn_kwargs = self._kwargs.get("transform_fn_kwargs", {})

    if not all([self.frame_length, self.frame_step, self.transform_fn]):
      raise ValueError(
          "`frame_length`, `frame_step`, and `transform_fn` must be"
          " provided in kwargs during initialization."
      )

    if self.pooling == "last":
      self.pool_fn = lambda x: x[-1][None, :]
    elif self.pooling == "mean":
      self.pool_fn = lambda x: np.mean(x, axis=0, keepdims=True)
    elif self.pooling == "max":
      self.pool_fn = lambda x: np.max(x, axis=0, keepdims=True)
    elif self.pooling is None:
      self.pool_fn = lambda x: x
    else:
      raise ValueError(
          f"Unknown pooling strategy: {self.pooling}. Supported are "
          "'last', 'mean', 'max', or None."
      )

    self._model_loaded = True

  def _encode(
      self,
      sound: types.Sound,
      **kwargs: Any,
  ) -> types.SoundEmbedding:
    """Encodes a single sound source."""
    waveform = np.asarray(sound.waveform, dtype=np.float32)
    params = sound.context
    assert len(waveform) == params.length, (
        f"Input waveform length {len(waveform)} does not match "
        f"params.length {params.length}."
    )
    num_frames = (
        (len(waveform) - self.frame_length + self.frame_step) // self.frame_step
    )
    if num_frames <= 0:
      return types.SoundEmbedding(
          embedding=np.array([]), timestamps=np.array([]), context=params
      )

    frames = np.zeros([num_frames, self.frame_length], dtype=np.float32)
    timestamps_list = []
    for i in range(num_frames):
      start = i * self.frame_step
      end = start + self.frame_length
      frames[i] = waveform[start:end]
      timestamps_list.append([start, end])

    final_transform_kwargs = self.transform_fn_kwargs.copy()
    final_transform_kwargs.update(kwargs)

    embeddings = self.transform_fn(frames, **final_transform_kwargs)
    waveform_embeddings = self.pool_fn(embeddings)

    # Adjust timestamps if pooling was applied
    if self.pooling is not None:
      # A single timestamp for the entire utterance
      embedding_timestamps = np.array([[0, len(waveform)]], dtype=int)
    else:
      embedding_timestamps = np.array(timestamps_list, dtype=int)

    return types.SoundEmbedding(
        embedding=waveform_embeddings,
        timestamps=embedding_timestamps,
        context=params,
    )

  def _encode_batch(
      self,
      sound_batch: Sequence[types.Sound],
      **kwargs: Any,
  ) -> Sequence[types.SoundEmbedding]:
    """Encodes a batch of sound sources.

    Args:
      sound_batch: A sequence of sound sources to encode.
      **kwargs: Runtime arguments that are passed to the transform function.
        These will override any `transform_fn_kwargs` set at init.

    Returns:
      A list of tuples, one for each input, each tuple containing:
        - embeddings (np.ndarray): A 2D array of transformed features.
        - timestamps (np.ndarray): A 2D array of [start, end] sample
          indices for each embedding frame.
    """
    outputs = []
    for sound in sound_batch:
      outputs.append(self._encode(sound, **kwargs))
    return outputs
