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

"""Simple Voice Questions (SVQ) dataset."""

import fnmatch
import glob
import hashlib
import io
import json
import logging
import os
import re
from typing import Any, Mapping

import apache_beam as beam

from array_record.python import array_record_module as array_record
from etils import epath
from mseb import dataset
from mseb import types
from mseb import utils
from mseb.datasets import base
from packaging import version
import pandas as pd
import pyarrow.parquet as pq


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
    "ur",
]


class _UttLookup:
  """A helper class to efficiently look up utterances from array records."""

  def __init__(
      self,
      base_path: str,
      utt_index_df: pd.DataFrame,
      streaming: bool = False,
      repo_id: str = "google/svq",
  ):
    self.base_path = base_path
    self.streaming = streaming
    self.repo_id = repo_id
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
      if self.streaming:
        parquet_path = f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/{path}.parquet"
        self.readers[path] = (
            "parquet_url",
            parquet_path,
        )  # We don't cache read_table for URL to avoid memory bloat
      else:
        parquet_path = os.path.join(self.base_path, f"{path}.parquet")
        array_record_path = os.path.join(self.base_path, f"{path}.array_record")

        if epath.Path(parquet_path).exists():
          with epath.Path(parquet_path).open("rb") as f:
            # Stream bytes natively via epath, restricting memory to 'waveform'
            self.readers[path] = (
                "parquet",
                pq.read_table(io.BytesIO(f.read()), columns=["waveform"]),
            )
        elif epath.Path(array_record_path).exists():
          self.readers[path] = (
              "array_record",
              array_record.ArrayRecordReader(array_record_path),
          )
        else:
          raise FileNotFoundError(
              f"Neither parquet nor array_record found for {path} in"
              f" {self.base_path}"
          )

    reader_type, reader = self.readers[path]
    if reader_type == "parquet":
      row = reader["waveform"][idx].as_py()
      if isinstance(row, dict) and "bytes" in row:
        return row["bytes"]
      elif isinstance(row, bytes):
        return row
      else:
        raise ValueError(f"Unexpected waveform type in parquet: {type(row)}")
    elif reader_type == "parquet_url":
      # Read only the single row from URL using pandas? Or use fsspec
      # Reading single row from parquet URL is hard without ParquetDataset.
      # Let's read the whole file if it's small, or use fsspec
      # For now, let's try to read it using pandas.
      df = pd.read_parquet(reader)  # reader is parquet_path
      row = df.iloc[idx]
      if "bytes" in row["waveform"]:
        return row["waveform"]["bytes"]
      return row["waveform"]
    else:
      return reader.read([idx])[0]


class _LoadAudioFn(beam.DoFn):
  """Loads audio for a single utterance."""

  def __init__(self, base_path: str, index_df: pd.DataFrame):
    self._base_path = base_path
    self._index_df = index_df
    self._utt_lookup: _UttLookup | None = None

  def setup(self):
    self._utt_lookup = _UttLookup(self._base_path, self._index_df)

  def process(self, record: dict[str, Any]):
    wav_bytes = self._utt_lookup(record["utt_id"])  # pyrefly: ignore[not-callable]
    waveform, sr = utils.wav_bytes_to_waveform(wav_bytes)

    speaker_age_val = record.get("speaker_age")
    context = types.SoundContextParams(
        id=record["utt_id"],
        sample_rate=sr,
        length=len(waveform),
        language=record.get("locale"),
        speaker_id=str(record.get("speaker_id")),
        speaker_age=int(speaker_age_val) if pd.notna(speaker_age_val) else None,  # pyrefly: ignore[bad-argument-type]
        speaker_gender=record.get("gender"),
        text=record.get("text"),
        waveform_start_second=0.0,
        waveform_end_second=len(waveform) / sr if sr > 0 else 0.0,
    )
    sound = types.Sound(waveform=waveform, context=context)  # pyrefly: ignore[bad-argument-type]
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
      streaming: bool = False,
      repo_id: str = "google/svq",
  ):
    super().__init__(base_path=base_path, split=split)
    self.base_path = dataset.get_base_path(self.base_path)
    self.streaming = streaming
    self.repo_id = repo_id
    self._index = self._load_index()
    self._utt_lookup = _UttLookup(
        self.base_path, self._index, streaming=streaming, repo_id=repo_id
    )
    self.utt_id_to_record = self._index.set_index("utt_id").to_dict("index")

  @property
  def metadata(self) -> types.DatasetMetadata:
    """Returns the structured metadata for SVQ."""
    return types.DatasetMetadata(
        name="Simple Voice Questions (SVQ)",
        description="A dataset for evaluating sound representations.",
        homepage="https://huggingface.co/datasets/google/svq",
        version="2.0.0",
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
    if self.streaming:
      utt_index_path = f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/utt_index.jsonl"
      try:
        return pd.read_json(utt_index_path, lines=True)
      except Exception as e:  # pylint: disable=broad-except
        logging.warning(
            "Failed to load index from %s, falling back to scanning: %s",
            utt_index_path,
            e,
        )
        # Fallback to scanning *.parquet
        all_files = utils.list_hf_files(self.repo_id, path=".")

        files = [
            f
            for f in all_files
            if fnmatch.fnmatch(os.path.basename(f), "*.parquet")
        ]
        if not files:
          raise FileNotFoundError(
              f"No parquet files found in {self.repo_id}"
          ) from e
    else:
      utt_index_path = os.path.join(self.base_path, "utt_index.jsonl")  # pyrefly: ignore[no-matching-overload]
      if epath.Path(utt_index_path).exists():
        return pd.read_json(utt_index_path, lines=True)

      files = glob.glob(os.path.join(self.base_path, "utts_*.parquet"))  # pyrefly: ignore[no-matching-overload]

    cols = [
        "utt_id",
        "locale",
        "speaker_id",
        "speaker_age",
        "speaker_gender",
        "environment",
        "text",
    ]

    records = []
    for f in files:
      if self.streaming:
        url = f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/{f}"
        table = pd.read_parquet(url, columns=cols)
        rel_name = os.path.splitext(f)[
            0
        ]  # f is relative to repo root if list_hf_files returns relative
      else:
        with epath.Path(f).open("rb") as parquet_file:
          table = pq.read_table(io.BytesIO(parquet_file.read()), columns=cols)
        basename = os.path.basename(f)
        rel_name = os.path.splitext(basename)[0]

      pylist = (
          table.to_pylist() if not self.streaming else table.to_dict("records")
      )  # pandas.to_dict('records') is similar to pylist
      for idx, r in enumerate(pylist):
        r["index"] = f"{rel_name}:{idx}"
        records.append(r)

    if not records:
      raise FileNotFoundError("No utterances found")

    return pd.DataFrame(records)

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
        speaker_age=int(speaker_age_val) if pd.notna(speaker_age_val) else None,  # pyrefly: ignore[bad-argument-type]
        speaker_gender=record.get("gender"),
        text=record.get("text"),
        waveform_start_second=0.0,
        waveform_end_second=len(waveform) / sr if sr > 0 else 0.0,
    )
    return types.Sound(waveform=waveform, context=context)  # pyrefly: ignore[bad-argument-type]

  def _get_task_path(self, task_name: str) -> str:
    """Returns the path to the task file for the given task name.

    Args:
      task_name: The name or wildcard pattern of the task file (e.g.,
        "utts_en_us_clean" or "utts_en_us_*"). Supports wildcards to
        match multiple environment files across a locale. Parquet format is
        preferred; JSONL format is deprecated and will be removed in a future
        release.

    Returns:
      The path to the task file.

    Raises:
      FileNotFoundError: If the task file does not exist.
    """
    full_path_prefix = os.path.join(self.base_path, task_name)  # pyrefly: ignore[no-matching-overload]
    target_dir = os.path.dirname(full_path_prefix)
    file_pattern = os.path.basename(task_name)
    base_target = epath.Path(target_dir)

    parquet_paths = base_target.glob(f"{file_pattern}.parquet")
    if list(parquet_paths):
      return f"{full_path_prefix}.parquet"

    jsonl_paths = base_target.glob(f"{file_pattern}.jsonl")
    if list(jsonl_paths):
      return f"{full_path_prefix}.jsonl"

    raise FileNotFoundError(
        f"Task file not found for task '{task_name}' in {self.base_path}. "
        "Tried .parquet and .jsonl"
    )

  def get_parquet_version(
      self, file_path: epath.Path
  ) -> version.Version | None:
    with file_path.open("rb") as parquet_f:
      schema = pq.read_schema(parquet_f)
    if schema.metadata and b"mseb_version" in schema.metadata:
      return version.parse(schema.metadata[b"mseb_version"].decode("utf-8"))
    return None

  def get_task_data(
      self, task_name: str | None = None, dtype: Mapping[str, Any] | None = None
  ) -> pd.DataFrame:
    """Loads the task data for the given task name.

    Args:
      task_name: The name or wildcard pattern of the task file (e.g.,
        "span_retrieval_cross_lang" or "utts_en_us_*"). Parquet format is
        preferred; JSONL format is deprecated and will be removed in a future
        release.
      dtype: The dtype for the columns (deprecated, only used for JSONL format).

    Returns:
      A pandas DataFrame containing the task data.

    Raises:
      FileNotFoundError: If the task file does not exist.
    """
    path = self._get_task_path(task_name)  # pyrefly: ignore[bad-argument-type]
    matched_files = sorted(
        epath.Path(os.path.dirname(path)).glob(os.path.basename(path))
    )
    if not matched_files:
      raise FileNotFoundError(f"No files matched '{path}'")
    if path.endswith(".parquet"):
      svq_version = version.parse(self.metadata.version)
      for f in matched_files:
        parquet_version = self.get_parquet_version(f)
        if parquet_version is None or parquet_version < svq_version:
          raise ValueError(
              f"Parquet files with version < {svq_version} are not supported."
          )
      dfs = []
      for f in matched_files:
        with f.open("rb") as parquet_f:
          dfs.append(pd.read_parquet(parquet_f))
    else:
      logging.warning(
          "Reading task data from JSONL is deprecated and will be removed in a"
          " future release. Please use Parquet format instead."
      )
      dfs = []
      for f in matched_files:
        with f.open("rb") as jsonl_f:
          dfs.append(pd.read_json(jsonl_f, lines=True, dtype=dtype))  # pyrefly: ignore[no-matching-overload]
    return pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]  # pyrefly: ignore[bad-return]

  def get_task_data_beam(self, task_name: str) -> beam.PTransform:
    """Loads the task data with audio for the given task name with beam."""
    return ReadTaskData(
        self.base_path,  # pyrefly: ignore[bad-argument-type]
        self._index,
        self._get_task_path(task_name),
    )

  def get_task_sounds_beam(
      self, task_name: str, locale: str | None = None
  ) -> beam.PTransform:
    """Loads the task data with audio for the given task name with beam."""
    transform = self.get_task_data_beam(task_name) | "TakeSound" >> beam.Map(
        lambda x: x["sound"]
    )

    if locale:
      transform = transform | f"FilterSoundsByLocale_{locale}" >> beam.Filter(
          lambda x: x.context.language == locale
      )

    return transform


# Matches either:
# 1) Start of string: (?:^)
# 2) A delimiter (- or _) following alphanumeric/prefix characters:
#    (?<=[a-zA-Z0-9_])[-_]
# followed by an optional minus sign and a large run of digits
#    (default >= 6 digits):
_PASSAGE_ID_PATTERN = re.compile(r"(?:^|(?<=[a-zA-Z0-9_])[-_])(-?\d{6,})")


def parse_passage_id(passage_id: str, min_digits: int = 6) -> int:
  """Extracts the large signed integer ID from a passage ID string.

  Handles:
    - Positive IDs with prefixes (e.g. "korean-7766157635581715307-8")
    - Negative IDs with prefixes (e.g. "arabic--3663137242854443418-hardneg-0")
    - Standalone positive/negative IDs (e.g. "-7615928064691233550",
    "8484105171262018122")
    - Trailing suffixes (e.g. "-hardneg", "-hardneg-2", "-unanswerable")

  Args:
    passage_id: The input ID string.
    min_digits: Minimum digits required to distinguish the ID from small suffix
      numbers like `-8` or `-2` (default is 6; SVQ IDs are typically 18-19
      digits).

  Returns:
    The integer value of the large ID.

  Raises:
    ValueError: If no large ID number is found in `passage_id`.
  """
  clean_str = passage_id.strip('\'"“” \t\n')
  pattern = (
      _PASSAGE_ID_PATTERN
      if min_digits == 6
      else re.compile(rf"(?:^|(?<=[a-zA-Z0-9_])[-_])(-?\d{{{min_digits},}})")
  )
  match = pattern.search(clean_str)
  if not match:
    raise ValueError(f"No large ID number found in: {passage_id!r}")
  return int(match.group(1))


def is_member_of_debug(passage_id: str) -> bool:
  _debug_ids: frozenset[str] = frozenset([
      "1023841985531689286",  # english
      "2888513011661240822",  # finnish
  ])
  return str(parse_passage_id(passage_id)) in _debug_ids


def is_member_of_compact(passage_id: str) -> bool:
  digest = hashlib.sha256(
      str(parse_passage_id(passage_id)).encode("utf-8")
  ).digest()
  fp64 = int.from_bytes(digest[:8], byteorder="little")
  return fp64 % 10 == 0
