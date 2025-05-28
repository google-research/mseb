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
import os
from array_record.python import array_record_module as array_record
import datasets
import librosa
import numpy as np
import pandas as pd
from scipy.io import wavfile


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
      self.readers[path] = array_record.ArrayRecordReader(
          array_record_path
      )
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
  basepath = os.path.dirname(filepath)
  utt_lookup = UttLookup(basepath, resample_hz=resample_hz)
  task = pd.read_json(filepath, lines=True)
  for ex in task.to_dict(orient="records"):
    utt = utt_lookup(ex["utt_id"])
    ex["waveform"] = utt
    yield ex


_CITATION = """\
@InProceedings{mseb,
title = {Massive Sound Embedding Benchmark (MSEB)},
author={Georg Heigold, Ehsan Variani, Tom Bagby, Ji Ma, Cyril Allauzen, Shankar Kumar, Michael Riley}
year={2025}
}
"""


class SvqDataset(datasets.GeneratorBasedBuilder):
  """SVQ dataset."""

  VERSION = datasets.Version("1.1.0")

  BUILDER_CONFIGS = [
      datasets.BuilderConfig(
          name="span_reasoning_in_lang",
          description="Span reasoning in language.",
      ),
  ]

  def _info(self):
    return datasets.DatasetInfo(
        description=(
            "Simple Voice Queries (SVQ) dataset, Task: span reasoning in"
            " language."
        ),
        features=datasets.Features({
            "utt_id": datasets.Value("string"),
            "waveform": datasets.Value("float32"),
            "text": datasets.Value("string"),
            "locale": datasets.Value("string"),
            "environment": datasets.Value("string"),
            "speaker_id": datasets.Value("string"),
            "speaker_age": datasets.Value("int32"),
            "speaker_gender": datasets.Value("string"),
            "page_id": datasets.Value("string"),
            "page_title": datasets.Value("string"),
            "passage_id": datasets.Value("string"),
            "passage_text": datasets.Value("string"),
            "span": datasets.Value("string"),
        }),
        homepage="https://huggingface.co/datasets/google/svq",
        license="Apache 2.0",
        citation=_CITATION,
    )
