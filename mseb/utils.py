# Copyright 2026 The MSEB Authors.
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

"""Utilities for MSEB library."""

import dataclasses
import hashlib
import io
import logging
import os
import subprocess
from typing import Optional

from etils import epath
import librosa
from mseb import encoder
from mseb import types
import numpy as np
from scipy.io import wavfile
import soundfile


logger = logging.getLogger(__name__)


def download_from_hf(repo_id: str, target_dir: str, repo_type: str = "dataset"):
  """Clones a repository from Hugging Face if not already present."""
  # Expand the '~' to the user's full home directory path.
  target_dir = os.path.expanduser(target_dir)
  if epath.Path(os.path.join(target_dir, ".git")).exists():
    logger.warning(
        "Repo '%s' already found at %s. Skipping.", repo_id, target_dir
    )
    return

  # For a real library, replacing this with the `huggingface_hub` library would
  # be more robust.
  clone_url = f"https://huggingface.co/{repo_type}s/{repo_id}"
  logging.info("Cloning '%s' to %s...", clone_url, target_dir)
  subprocess.run(["git", "clone", clone_url, target_dir], check=True)


def read_audio(
    file_path: str, target_sr: Optional[int] = None
) -> tuple[np.ndarray, int]:
  """Reads an audio file."""
  waveform, orig_sr = soundfile.read(file_path, dtype="float32")

  # Convert to mono
  if waveform.ndim > 1:
    waveform = np.mean(waveform, axis=1)

  if target_sr and target_sr != orig_sr:
    waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
    return waveform, target_sr

  return waveform, orig_sr


def wav_bytes_to_waveform(wav_bytes: bytes) -> tuple[np.ndarray, int]:
  """Reads WAV bytes and returns a normalized float32 numpy array."""
  rate, data = wavfile.read(io.BytesIO(wav_bytes))

  if data.dtype == np.int16:
    waveform = data.astype(np.float32) / np.iinfo(np.int16).max
  elif data.dtype == np.int32:
    waveform = data.astype(np.float32) / np.iinfo(np.int32).max
  elif data.dtype == np.float32:
    waveform = data.astype(np.float32)
  else:
    raise TypeError(f"Unsupported data type: {data.dtype}")

  # Convert to mono
  if data.ndim > 1:
    waveform = np.mean(waveform, axis=1)

  return waveform, rate


def sound_to_wav_bytes(sound: types.Sound) -> bytes:
  """Converts a Sound object's waveform to a 16-bit WAV byte string.

  This function handles various input waveform data types (integers and floats)
  by first normalizing them to a standard float32 format before encoding.

  Args:
    sound: The input Sound object containing the waveform and context.

  Returns:
    A byte string representing the sound in 16-bit WAV format.
  """
  int16_sound = encoder.resample_sound(
      sound,
      sound.context.sample_rate,
      np.int16
  )

  buffer = io.BytesIO()
  wavfile.write(
      buffer,
      rate=int16_sound.context.sample_rate,
      data=int16_sound.waveform
  )
  return buffer.getvalue()


@dataclasses.dataclass
class SpecAugmentConfig:
  """Configuration for SpecAugment.

  Attributes:
    n_fft: FFT window size.
    hop_length: Number of audio samples between adjacent STFT columns.
    max_freq_mask_proportion: Maximum fraction of frequency bins to mask.
    max_time_mask_proportion: Maximum fraction of time steps to mask.
    max_freq_mask_width: Absolute maximum width of a single frequency mask.
    max_time_mask_width: Absolute maximum width of a single time mask.
    mask_value: Constant value to fill masked regions (0.0 for silence).
  """
  n_fft: int = 2048
  hop_length: int = 512
  max_freq_mask_proportion: float = 0.15
  max_time_mask_proportion: float = 0.20
  max_freq_mask_width: int = 27
  max_time_mask_width: int = 100
  mask_value: float = 0.0


def get_deterministic_seed(utt_id: str, index: int) -> int:
  """Generates a unique, deterministic seed for a sample and index.

  Args:
    utt_id: The unique identifier for the utterance.
    index: The augmentation index (0, 1, ..., N).

  Returns:
    An integer seed suitable for np.random.default_rng.
  """
  identifier = f"{utt_id}_{index}".encode("utf-8")
  return int(hashlib.md5(identifier).hexdigest(), 16) % (2**32)


def apply_specaugment_to_waveform(
    x: np.ndarray,
    config: SpecAugmentConfig,
    rng: np.random.Generator,
) -> np.ndarray:
  """Applies SpecAugment to a waveform using a provided RNG for determinism.

  This function transforms a 1D waveform into a spectrogram via STFT,
  applies random frequency and time masking, and transforms it back to
  the time domain via ISTFT. Unlike the standard SpecAugment approach [1],
  which typically modifies Log-Mel filterbank features (see [2]), this
  function preserves the linear frequency scale to allow for a perfect
  reconstruction via Inverse STFT (iSTFT).

  Args:
    x: Input audio waveform as a 1D float32 numpy array.
    config: SpecAugmentConfig object containing masking parameters.
    rng: A numpy random generator instance (e.g., np.random.default_rng(seed)).

  Returns:
    The augmented waveform as a 1D float32 numpy array of the same length as x.

  References:
    [1] Park et al., "SpecAugment: A Simple Data Augmentation Method
        for Automatic Speech Recognition," 2019.
        https://arxiv.org/abs/1904.08779
    [2] PyPI SpecAugment (Standard Feature-Space Implementation):
        https://pypi.org/project/spec-augment/
  """
  # 1. Compute STFT
  stft_matrix = librosa.stft(
      x,
      n_fft=config.n_fft,
      hop_length=config.hop_length
  )
  magnitude, phase = librosa.magphase(stft_matrix)
  n_freq_bins, n_time_steps = magnitude.shape
  mag_aug = np.copy(magnitude)

  # 2. Determine random total proportions to mask
  freq_mask_proportion = rng.uniform(0.0, config.max_freq_mask_proportion)
  time_mask_proportion = rng.uniform(0.0, config.max_time_mask_proportion)

  total_freq_bins_to_mask = int(n_freq_bins * freq_mask_proportion)
  total_time_steps_to_mask = int(n_time_steps * time_mask_proportion)

  # 3. Apply Frequency Masking
  masked_freq_bins = 0
  while masked_freq_bins < total_freq_bins_to_mask:
    max_width = min(
        config.max_freq_mask_width,
        total_freq_bins_to_mask - masked_freq_bins
    )
    if max_width < 1:
      break
    f_width = rng.integers(1, max_width + 1)
    f0 = rng.integers(0, n_freq_bins - f_width + 1)
    mag_aug[f0:f0 + f_width, :] = config.mask_value
    masked_freq_bins += f_width

  # 4. Apply Time Masking
  masked_time_steps = 0
  while masked_time_steps < total_time_steps_to_mask:
    max_width = min(
        config.max_time_mask_width,
        total_time_steps_to_mask - masked_time_steps
    )
    if max_width < 1:
      break
    t_width = rng.integers(1, max_width + 1)
    t0 = rng.integers(0, n_time_steps - t_width + 1)
    mag_aug[:, t0:t0 + t_width] = config.mask_value
    masked_time_steps += t_width

  # 5. Inverse STFT to return to waveform
  stft_aug_matrix = mag_aug * phase
  return librosa.istft(
      stft_aug_matrix,
      hop_length=config.hop_length,
      length=len(x)
  )
