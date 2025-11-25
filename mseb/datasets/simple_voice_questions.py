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

import json
import os
from typing import Any, Mapping

import apache_beam as beam
from array_record.python import array_record_module as array_record
from etils import epath
from mseb import dataset
from mseb import types
from mseb import utils
from mseb.datasets import base
import pandas as pd


LANGUAGES = [
    "ar",
    "bn",
    "en",
    "fi",
    "gu",
    "hi",
    "id",
    "ja",
    "kn",
    "ko",
    "ml",
    "mr",
    "ru",
    "sw",
    "ta",
    "te",
    "ur"
]


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
      if not epath.Path(array_record_path).exists():
        raise FileNotFoundError(
            f"Array record file not found: {array_record_path}"
        )
      self.readers[path] = array_record.ArrayRecordReader(array_record_path)

    return self.readers[path].read([idx])[0]


class _LoadAudioFn(beam.DoFn):
  """Loads audio for a single utterance."""

  def __init__(self, base_path: str, index_df: pd.DataFrame):
    self._base_path = base_path
    self._index_df = index_df
    self._utt_lookup: _UttLookup | None = None

  def setup(self):
    self._utt_lookup = _UttLookup(self._base_path, self._index_df)

  def process(self, record: dict[str, Any]):
    wav_bytes = self._utt_lookup(record["utt_id"])
    waveform, sr = utils.wav_bytes_to_waveform(wav_bytes)

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
    sound = types.Sound(waveform=waveform, context=context)
    yield record | {"sound": sound}


class ReadTaskData(beam.PTransform):
  """A PTransform that reads task data and loads audio."""

  def __init__(
      self,
      base_path: str,
      index: pd.DataFrame,
      task_path: str,
  ):
    self._base_path = base_path
    self._index = index
    self._task_path = task_path

  def expand(self, pcoll: beam.PCollection) -> beam.PCollection:
    return (
        pcoll
        | "ReadTaskJson" >> beam.io.ReadFromText(self._task_path)
        | "ParseTaskJson" >> beam.Map(json.loads)
        | "LoadAudio" >> beam.ParDo(_LoadAudioFn(self._base_path, self._index))
    )


class SimpleVoiceQuestionsDataset(base.MsebDataset):
  """Simple Voice Questions (SVQ) dataset.

  This class loads the entire corpus of utterances and provides a method
  to access specific evaluation task files.
  """

  def __init__(
      self,
      base_path: str | None = None,
      split: str = "all",
  ):
    super().__init__(base_path=base_path, split=split)
    self.base_path = dataset.get_base_path(self.base_path)
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
    if not epath.Path(utt_index_path).exists():
      raise FileNotFoundError(f"Master index not found: {utt_index_path}")
    return pd.read_json(utt_index_path, lines=True)

  def get_sound(self, record: Mapping[str, Any]) -> types.Sound:
    """Retrieves a Sound object by its unique utterance ID."""
    utt_id = record["utt_id"]
    if utt_id not in self.utt_id_to_record:
      raise ValueError(f"Utterance ID '{utt_id}' not found in corpus.")
    record = self.utt_id_to_record[utt_id]
    # We need to manually add utt_id back as it's the index now
    record["utt_id"] = utt_id
    return self._get_sound(record)

  def _get_sound(self, record: dict[str, Any]) -> types.Sound:
    """Loads a single utterance from its record in the utterance index."""
    wav_bytes = self._utt_lookup(record["utt_id"])
    waveform, sr = utils.wav_bytes_to_waveform(wav_bytes)

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

  def _get_task_path(self, task_name: str) -> str:
    """Returns the path to the task file for the given task name."""
    task_filename = f"{task_name}.jsonl"
    task_path = os.path.join(self.base_path, task_filename)
    if not epath.Path(task_path).exists():
      raise FileNotFoundError(
          f"Task file not found: {task_path}. "
          f"Ensure the task name '{task_name}' is valid."
      )
    return task_path

  def get_task_data(
      self, task_name: str | None = None, dtype: Mapping[str, Any] | None = None
  ) -> pd.DataFrame:
    """Loads the task data for the given task name.

    Args:
      task_name: The name of the task file (e.g., "span_retrieval_cross_lang").
      dtype: The dtype for the columns.

    Returns:
      A pandas DataFrame containing the task data.

    Raises:
      FileNotFoundError: If the task file does not exist.
    """
    if not task_name:
      raise ValueError(
          "task_name must be specified for SimpleVoiceQuestionsDataset"
      )
    return pd.read_json(
        self._get_task_path(task_name), lines=True, dtype=dtype
    )

  def get_task_data_beam(self, task_name: str) -> beam.PTransform:
    """Loads the task data with audio for the given task name with beam."""
    return ReadTaskData(
        self.base_path,
        self._index,
        self._get_task_path(task_name),
    )

  def get_task_sounds_beam(
      self, task_name: str
  ) -> beam.PCollection[types.Sound]:
    """Loads the task data with audio for the given task name with beam."""
    return self.get_task_data_beam(task_name) | "TakeSound" >> beam.Map(
        lambda x: x["sound"]
    )
