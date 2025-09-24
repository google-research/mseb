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

"""Simple Voice Questions (SVQ) dataset."""

import io
import os
from typing import Any, Optional

from absl import flags
from array_record.python import array_record_module as array_record
import librosa
from mseb import types
import numpy as np
import pandas as pd
from scipy.io import wavfile
import tensorflow as tf


SVQ_BASEPATH = flags.DEFINE_string(
    "simple_voice_questions_base_path",
    None,
    "Path to the SVQ dataset.",
)


def _get_base_path(basepath: str | None = None) -> str:
  """Return basepath from argument or flag."""
  if basepath is not None:
    return basepath
  if SVQ_BASEPATH.value is not None:
    return SVQ_BASEPATH.value
  raise ValueError(
      "basepath must be provided either as an argument or through the"
      " --simple_voice_questions_base_path flag."
  )


def _read_wav_bytes(
    wav_bytes: bytes, target_sr: Optional[int] = None
) -> tuple[np.ndarray, int]:
  """Reads WAV bytes and returns a normalized float32 numpy array."""
  rate, data = wavfile.read(io.BytesIO(wav_bytes))

  # Convert to mono
  if data.ndim > 1:
    data = np.mean(data, axis=1)
  if data.dtype == np.int16:
    waveform = data.astype(np.float32) / np.iinfo(np.int16).max
  elif data.dtype == np.int32:
    waveform = data.astype(np.float32) / np.iinfo(np.int32).max
  elif data.dtype == np.float32:
    waveform = data.astype(np.float32)
  else:
    raise TypeError(f"Unsupported data type: {data.dtype}")

  if target_sr and target_sr != rate:
    waveform = librosa.resample(waveform, orig_sr=rate, target_sr=target_sr)
    return waveform, target_sr

  return waveform, rate


class _UttLookup:
  """A helper class to efficiently look up utterances from array records."""

  def __init__(self, base_path: str, utt_index_df: pd.DataFrame):
    self.base_path = base_path
    self.utt_id_to_index = dict(
        zip(utt_index_df["utt_id"], utt_index_df["index"])
    )
    self.readers: dict[str, Any] = {}

  def __call__(self, utt_id: str) -> bytes:
    """Retrieves the raw wav bytes for a given utterance ID."""
    if utt_id not in self.utt_id_to_index:
      raise ValueError(f"Utterance ID '{utt_id}' not found in index.")

    path, idx_str = self.utt_id_to_index[utt_id].rsplit(":", 1)
    idx = int(idx_str)

    if path not in self.readers:
      array_record_path = os.path.join(self.base_path, f"{path}.array_record")
      if not tf.io.gfile.exists(array_record_path):
        raise FileNotFoundError(
            f"Array record file not found: {array_record_path}"
        )
      self.readers[path] = array_record.ArrayRecordReader(array_record_path)

    return self.readers[path].read([idx])[0]


class SimpleVoiceQuestionsDataset:
  """Simple Voice Questions (SVQ) dataset.

  This class loads the entire corpus of utterances and provides a method
  to access specific evaluation task files.
  """

  def __init__(
      self,
      base_path: str | None = None,
      split: str = "all",
      target_sr: int | None = None,
  ):
    if split != "all":
      raise ValueError(
          "The 'split' argument is not used for"
          " SimpleVoiceQuestionsDatasetinitialization. Initialize the dataset"
          " without a split to load the main corpus, then use the"
          " `get_task_data('task_name')` method."
      )
    self.base_path = _get_base_path(base_path)
    self.split = split
    self.target_sr = target_sr
    self._index = self._load_index()
    self._utt_lookup = _UttLookup(self.base_path, self._index)
    self.utt_id_to_record = self._index.set_index("utt_id").to_dict("index")

  @property
  def metadata(self) -> types.DatasetMetadata:
    """Returns the structured metadata for SVQ."""
    return types.DatasetMetadata(
        name="Simple Voice Questions (SVQ)",
        description="A dataset for evaluating sound representations.",
        homepage="https://huggingface.co/datasets/google/svq",
        version="1.0.0",
        license="CC BY 4.0",
        mseb_tasks=[
            "classification",
            "clustering",
            "reasoning",
            "reconstruction",
            "reranking",
            "retrieval",
            "segmentation",
        ],
    )

  def __len__(self) -> int:
    return len(self._index)

  def _load_index(self) -> pd.DataFrame:
    """Loads the master index of all unique utterances."""
    utt_index_path = os.path.join(self.base_path, "utt_index.jsonl")
    if not tf.io.gfile.exists(utt_index_path):
      raise FileNotFoundError(f"Master index not found: {utt_index_path}")
    return pd.read_json(utt_index_path, lines=True)

  def get_sound_by_id(self, utt_id: str) -> types.Sound:
    """Retrieves a Sound object by its unique utterance ID."""
    if utt_id not in self.utt_id_to_record:
      raise ValueError(f"Utterance ID '{utt_id}' not found in corpus.")
    record = self.utt_id_to_record[utt_id]
    # We need to manually add utt_id back as it's the index now
    record["utt_id"] = utt_id
    return self._get_sound(record)

  def _get_sound(self, record: dict[str, Any]) -> types.Sound:
    """Loads a single utterance from its record in the utterance index."""
    wav_bytes = self._utt_lookup(record["utt_id"])
    waveform, sr = _read_wav_bytes(wav_bytes, self.target_sr)

    speaker_age_val = record.get("speaker_age")
    context = types.SoundContextParams(
        id=record["utt_id"],
        sample_rate=sr,
        length=len(waveform),
        language=record.get("locale"),
        speaker_id=str(record.get("speaker_id")),
        speaker_age=int(speaker_age_val) if pd.notna(speaker_age_val) else None,
        speaker_gender=record.get("gender"),
        text=record.get("text"),
        waveform_start_second=0.0,
        waveform_end_second=len(waveform) / sr if sr > 0 else 0.0,
    )
    return types.Sound(waveform=waveform, context=context)

  def get_task_data(self, task_name: str) -> pd.DataFrame:
    """Loads the task data for the given task name.

    Args:
      task_name: The name of the task file (e.g., "span_retrieval_cross_lang").

    Returns:
      A pandas DataFrame containing the task data.

    Raises:
      FileNotFoundError: If the task file does not exist.
    """
    task_filename = f"{task_name}.jsonl"
    task_path = os.path.join(self.base_path, task_filename)
    if not tf.io.gfile.exists(task_path):
      raise FileNotFoundError(
          f"Task file not found: {task_path}. "
          f"Ensure the task name '{task_name}' is valid."
      )
    return pd.read_json(task_path, lines=True)
