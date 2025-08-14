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

"""SVQ data reading."""


import io
import json
import os
import subprocess
from absl import flags
import apache_beam as beam
from array_record.python import array_record_module as array_record
import librosa
import numpy as np
import pandas as pd
from scipy.io import wavfile

SVQ_BASEPATH = flags.DEFINE_string(
    "svq_basepath",
    None,
    "Path to the SVQ dataset.",
)


def _get_basepath(basepath: str | None = None) -> str:
  """Return basepath from argument or flag."""
  if basepath is not None:
    return basepath
  if SVQ_BASEPATH.value is not None:
    return SVQ_BASEPATH.value
  raise ValueError(
      "basepath must be provided either as an argument or through the"
      " --svq_basepath flag."
  )


def read_wav_bytes_to_normalized_float(
    wav_bytes, resample_hz: float | None = None
):
  """Reads WAV bytes object and returns normalized float numpy array.

  Args:
    wav_bytes: WAV bytes object.
    resample_hz: Optional resample rate.

  Returns:
    (waveform, original sample rate before any resample)
  """
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
  return x, rate


def maybe_clone_svq_dataset(output_dir: str) -> str:
  """Clones the google/svq dataset from Hugging Face if not already present.

  Args:
    output_dir: The directory where the 'svq' dataset subdirectory should reside
      or be created.

  Returns:
    The path to the local 'svq' dataset directory.

  Raises:
    RuntimeError: If git clone fails.
  """
  dataset_name = "svq"
  repo_url = "https://huggingface.co/datasets/google/svq"
  target_repo_path = os.path.join(output_dir, dataset_name)

  if os.path.exists(os.path.join(target_repo_path, ".git")):
    print(
        f"Dataset '{dataset_name}' already found at {target_repo_path}."
        " Skipping clone."
    )
    return target_repo_path

  os.makedirs(output_dir, exist_ok=True)

  print(
      f"Cloning dataset '{dataset_name}' from {repo_url} to {target_repo_path}"
  )
  cmd = ["git", "clone", repo_url, target_repo_path]
  result = subprocess.run(cmd, check=False, capture_output=True, text=True)

  if result.returncode != 0:
    raise RuntimeError(
        f"Failed to clone repository. Git command output:\n{result.stderr}"
    )
  print(f"Successfully cloned '{dataset_name}' to {target_repo_path}.")
  return target_repo_path


def read_utt_index(basepath: str | None = None):
  """Read utt_index.jsonl file to a dict of {uttid: path:index}."""
  basepath = _get_basepath(basepath)
  df = pd.read_json(os.path.join(basepath, "utt_index.jsonl"), lines=True)
  return dict(zip(df["utt_id"], df["index"]))


class UttLookup:
  """Lookup utterances by utt_id with optional resampling.

  Usage:
    utt_lookup = UttLookup(basepath)
    waveform = utt_lookup(utt_id)
  """

  def __init__(
      self, basepath: str | None = None, resample_hz: float | None = None
  ):
    self.basepath = _get_basepath(basepath)
    self.resample_hz = resample_hz
    self.utt_id_to_path_idx = read_utt_index(self.basepath)
    self.readers = {}
    self.orig_sample_rate_ = None

  @property
  def orig_sample_rate(self):
    if self.orig_sample_rate_ is None:
      utt_id = next(iter(self.utt_id_to_path_idx))
      self(utt_id)
    return self.orig_sample_rate_

  def __call__(self, utt_id: str):
    path, idx = self.utt_id_to_path_idx[utt_id].split(":")
    if path not in self.readers:
      array_record_path = os.path.join(self.basepath, f"{path}.array_record")
      self.readers[path] = array_record.ArrayRecordReader(array_record_path)
    b = self.readers[path].read([int(idx)])
    waveform, sample_rate = read_wav_bytes_to_normalized_float(
        b[0], resample_hz=self.resample_hz
    )
    if self.orig_sample_rate_ is None:
      self.orig_sample_rate_ = sample_rate
    if sample_rate != self.orig_sample_rate_:
      raise ValueError(
          f"Sample rate mismatch: {sample_rate} != {self.orig_sample_rate_}"
      )
    return waveform


def generate_examples(filepath, resample_hz: float | None = None):
  """Generate examples from a jsonl task file."""
  basepath = _get_basepath(os.path.dirname(filepath) or None)
  filepath = os.path.join(basepath, os.path.basename(filepath))
  utt_lookup = UttLookup(basepath, resample_hz=resample_hz)
  task = pd.read_json(filepath, lines=True)
  for ex in task.to_dict(orient="records"):
    utt = utt_lookup(ex["utt_id"])
    ex["waveform"] = utt
    yield ex


def generate_examples_beam(pipeline, filepath: str):
  """Generate examples from a jsonl task file with beam."""
  utt_lookup = UttLookup(os.path.dirname(filepath))
  return (
      pipeline
      | beam.io.ReadFromText(filepath)
      | beam.Map(json.loads)
      | beam.Map(lambda x: x | {"waveform": utt_lookup(x["utt_id"])})
  )
