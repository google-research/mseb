# Copyright 2024 The MSEB Authors.
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

"""SVQ data reading."""

import io
import os
from array_record.python import array_record_module as array_record
import librosa
import numpy as np
import pandas as pd
from scipy.io import wavfile


def read_wav_bytes_to_normalized_float(
    wav_bytes, resample_hz: float | None = None
):
  """Reads WAV bytes object and returns normalized float numpy array."""
  rate, data = wavfile.read(io.BytesIO(wav_bytes))

  if data.ndim > 1 and data.shape[1] > 1:
    raise ValueError("Only mono WAV files are supported.")

  # Convert data to float and normalize
  if data.dtype == np.int16:
    x = data.astype(np.float32) / np.iinfo(np.int16).max
  elif data.dtype == np.int32:
    x = data.astype(np.float32) / np.iinfo(np.int32).max
  elif data.dtype == np.float32:
    x = data
  else:
    raise TypeError(f"Unsupported data type: {data.dtype}")
  if resample_hz is not None and resample_hz != rate:
    x = librosa.resample(x, orig_sr=rate, target_sr=resample_hz)
    rate = resample_hz
  return x, rate


def read_utt_index(basepath):
  """Read utt_index.jsonl file to a dict of {uttid: path:index}."""
  df = pd.read_json(os.path.join(basepath, "utt_index.jsonl"), lines=True)
  return dict(zip(df["utt_id"], df["index"]))


class UttLookup:
  """Lookup utterances by utt_id with optional resampling.

  Usage:
    utt_lookup = UttLookup(basepath)
    waveform = utt_lookup(utt_id)
  """

  def __init__(self, basepath, resample_hz: float | None = None):
    self.basepath = basepath
    self.resample_hz = resample_hz
    self.utt_id_to_path_idx = read_utt_index(basepath)
    self.readers = {}

  def __call__(self, utt_id: str):
    path, idx = self.utt_id_to_path_idx[utt_id].split(":")
    if path not in self.readers:
      array_record_path = os.path.join(self.basepath, f"{path}.array_record")
      self.readers[path] = array_record.ArrayRecordReader(
          array_record_path
      )
    b = self.readers[path].read([int(idx)])
    waveform, _ = read_wav_bytes_to_normalized_float(
        b[0], resample_hz=self.resample_hz
    )
    return waveform
