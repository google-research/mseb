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

"""Speech-MASSIVE dataset."""

from collections.abc import Callable, Mapping
import fnmatch
import glob
import io
import os
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

import apache_beam as beam
from etils import epath
from mseb import dataset
from mseb import types
from mseb import utils
from mseb.datasets import base
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

bcp47_by_locale = {
    "ar_sa": "ar-SA",
    "de_de": "de-DE",
    "es_es": "es-ES",
    "fr_fr": "fr-FR",
    "hu_hu": "hu-HU",
    "ko_kr": "ko-KR",
    "nl_nl": "nl-NL",
    "pl_pl": "pl-PL",
    "pt_pt": "pt-PT",
    "ru_ru": "ru-RU",
    "tr_tr": "tr-TR",
    "vi_vn": "vi-VN",
}
locale_by_bcp47 = {v: k for k, v in bcp47_by_locale.items()}

DEFAULT_BASE_PATH = None


class _HttpRangeFile(io.RawIOBase):
  """Seekable read-only HTTP Range wrapper for selective Parquet reads."""

  def __init__(
      self,
      url: str,
      headers: Mapping[str, str] | None = None,
      repo_id: str | None = None,
      opener: urllib.request.OpenerDirector | None = None,
  ):
    super().__init__()
    self._url = url
    self._origin = urllib.parse.urlsplit(url)[:2]
    self._resolved_url: str | None = None
    self._headers = dict(headers) if headers else {}
    self._repo_id = repo_id
    self._opener = opener or utils.build_safe_url_opener()
    self._pos = 0
    self._size = self._fetch_size()

  def _urlopen(self, byte_range: str):
    """Opens an HTTP Range request and translates 401/403 to PermissionError."""
    target_url = self._resolved_url or self._url
    target_parts = urllib.parse.urlsplit(target_url)
    if self._origin[0] == "https" and target_parts.scheme != "https":
      raise urllib.error.URLError(
          f"Refusing non-HTTPS URL scheme {target_parts.scheme!r}"
      )
    same_origin = (target_parts.scheme, target_parts.netloc) == self._origin
    if same_origin:
      headers = dict(self._headers)
    else:
      headers = {
          header_name: header_value
          for header_name, header_value in self._headers.items()
          if header_name.lower() != "authorization"
      }
    req = urllib.request.Request(target_url, headers=headers)
    req.add_header("Range", byte_range)
    try:
      response = self._opener.open(
          req, timeout=utils.DEFAULT_HTTP_TIMEOUT_SECONDS
      )
    except urllib.error.HTTPError as e:
      if e.code in (401, 403) and target_url != self._url:
        self._resolved_url = None
        return self._urlopen(byte_range)
      if e.code in (401, 403) and self._repo_id:
        raise PermissionError(
            f"Access to Hugging Face dataset '{self._repo_id}' denied "
            f"(HTTP {e.code}). Accept the dataset terms at "
            f"https://huggingface.co/datasets/{self._repo_id} and authenticate "
            "via 'hf auth login' or the HF_TOKEN environment variable."
        ) from e
      raise
    if target_url == self._url:
      resolved_url = response.geturl()
      if resolved_url:
        resolved_scheme = urllib.parse.urlsplit(resolved_url).scheme
        if self._origin[0] != "https" or resolved_scheme == "https":
          self._resolved_url = resolved_url
    return response

  def _fetch_size(self) -> int:
    with self._urlopen("bytes=0-0") as response:
      content_range = response.headers.get("Content-Range", "")
      if "/" in content_range:
        total_size = content_range.rsplit("/", 1)[1]
        if total_size != "*":
          return int(total_size)
      if response.status in (None, 200):
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
          return int(content_length)
      raise OSError(
          "Remote server did not provide total file size in Content-Range or "
          "Content-Length headers."
      )

  def readable(self) -> bool:
    return True

  def seekable(self) -> bool:
    return True

  def tell(self) -> int:
    return self._pos

  def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
    if whence == io.SEEK_SET:
      new_pos = offset
    elif whence == io.SEEK_CUR:
      new_pos = self._pos + offset
    elif whence == io.SEEK_END:
      new_pos = self._size + offset
    else:
      raise ValueError(f"Unsupported whence: {whence}")
    if new_pos < 0:
      raise ValueError(f"Negative seek position: {new_pos}")
    self._pos = new_pos
    return self._pos

  def readinto(self, b: Any) -> int:
    buffer_view = memoryview(b).cast("B")
    if self._pos >= self._size or not buffer_view:
      return 0
    end = min(self._pos + len(buffer_view), self._size) - 1
    expected_bytes = end - self._pos + 1
    with self._urlopen(f"bytes={self._pos}-{end}") as response:
      if response.status is not None and response.status != 206:
        raise OSError(
            "Expected HTTP 206 Partial Content for Range request, got "
            f"{response.status}"
        )
      chunk = response.read(expected_bytes + 1)
    if len(chunk) > expected_bytes:
      raise OSError(
          f"HTTP Range response exceeded requested {expected_bytes} bytes"
      )
    byte_count = len(chunk)
    buffer_view[:byte_count] = chunk
    self._pos += byte_count
    return byte_count


def _open_hf_parquet(
    repo_id: str, filename: str, token: str | None = None
) -> pa.PythonFile:
  """Opens a Parquet file on Hugging Face via buffered HTTP Range requests."""
  url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
  resolved_token = utils.get_hf_token(token)
  headers = {}
  if resolved_token:
    headers["Authorization"] = f"Bearer {resolved_token}"
  raw_file = _HttpRangeFile(url, headers=headers, repo_id=repo_id)
  buffered = io.BufferedReader(raw_file, buffer_size=256 * 1024)
  return pa.PythonFile(buffered, mode="r")


def _read_parquet_file(
    parquet_f: Any,
    with_audio: bool = False,
    filter_fn: Callable[[Any], bool] | None = None,
) -> pd.DataFrame:
  """Reads a Parquet file with optional column and row-group filtering."""
  pf = pq.ParquetFile(parquet_f, pre_buffer=True, buffer_size=8 * 1024 * 1024)
  cols = [field.name for field in pf.schema_arrow]
  if not with_audio and "audio" in cols:
    cols.remove("audio")
  if filter_fn is None:
    return pf.read(columns=cols).to_pandas()

  all_ids = pf.read(columns=["id"])["id"].to_pylist()
  matching_rgs = []
  flat_mask: list[bool] = []
  offset = 0
  for i in range(pf.num_row_groups):
    rg_rows = pf.metadata.row_group(i).num_rows
    rg_ids = all_ids[offset : offset + rg_rows]
    offset += rg_rows
    matches = [bool(filter_fn(val)) for val in rg_ids]
    if any(matches):
      matching_rgs.append(i)
      flat_mask.extend(matches)
  if not matching_rgs:
    return pf.schema_arrow.empty_table().select(cols).to_pandas()
  rg_table = pf.read_row_groups(matching_rgs, columns=cols)
  return rg_table.filter(pa.array(flat_mask, type=pa.bool_())).to_pandas()


class SpeechMassiveDataset(base.MsebDataset):
  """SpeechMassive dataset."""

  def __init__(
      self,
      filename: str,
      base_path: str | None = None,
      repo_id: str = "FBK-MT/Speech-MASSIVE-test",
      streaming: bool = False,
      token: str | None = None,
      filter_fn: Callable[[Any], bool] | None = None,
  ):
    """Initializes the dataset for a specific file pattern.

    Args:
      filename: The file name relative to the base path.
      base_path: The root directory to store/find the dataset. If None, falls
        back to `--dataset_basepath`, then `DEFAULT_BASE_PATH` when `streaming`
        is False.
      repo_id: The Hugging Face repository ID to download from. Defaults to
        'FBK-MT/Speech-MASSIVE-test'.
      streaming: Whether to stream data selectively from Hugging Face when no
        explicit `base_path` or `--dataset_basepath` override is set.
      token: Hugging Face authentication token for private/gated repos.
      filter_fn: Optional predicate over example `id` to filter loaded rows.
    """
    super().__init__(base_path=base_path, split="no_used")
    if streaming:
      try:
        explicit_path = dataset.get_base_path(
            self.base_path, default_basepath=None
        )
      except ValueError:
        explicit_path = None
      if explicit_path:
        self.base_path = explicit_path
        streaming = False
      else:
        self.base_path = None
    else:
      self.base_path = dataset.get_base_path(
          self.base_path, default_basepath=DEFAULT_BASE_PATH
      )
    self.repo_id = repo_id
    self.filename = filename
    self.streaming = streaming
    self.token = token
    self._filter_fn = filter_fn
    self._data_cache = None

  @property
  def _data(self) -> pd.DataFrame:
    if self._data_cache is None:
      self._data_cache = self._load_data(with_audio=False)
    return self._data_cache

  @property
  def metadata(self) -> types.DatasetMetadata:
    """Returns the structured metadata for the MASSIVE dataset."""
    return types.DatasetMetadata(
        name="Speech-MASSIVE",
        description=(
            "A multilingual dataset for intent classification and slot "
            "filling. This loader defaults to the 'speechcolab/massive' "
            "version but can be pointed to other versions like "
            "'FBK-MT/Speech-MASSIVE'."
        ),
        homepage="https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test",
        version="2024.08.08",
        license="cc-by-nc-sa-4.0",
        mseb_tasks=[
            "classification",
            "clustering",
            "reconstruction",
        ],
    )

  def __len__(self) -> int:
    return len(self._data)

  def _download_and_prepare(self) -> None:
    """Downloads the dataset from Hugging Face."""
    utils.download_from_hf(self.repo_id, self.base_path)  # pyrefly: ignore[bad-argument-type]

  def _get_files(self) -> list[str]:
    """Returns the list of files to read."""
    if self.streaming:
      if not glob.has_magic(self.filename):
        return [self.filename]
      else:
        all_files = utils.list_hf_files(
            self.repo_id, path=os.path.dirname(self.filename), token=self.token
        )
        filtered_files = sorted(
            f
            for f in all_files
            if fnmatch.fnmatch(
                os.path.basename(f), os.path.basename(self.filename)
            )
        )
        if not filtered_files:
          raise FileNotFoundError(
              f"No match for {self.filename} in {self.repo_id}"
          )
        return filtered_files
    else:
      parquet_path = os.path.join(self.base_path, self.filename)  # pyrefly: ignore[no-matching-overload]
      parquet_files = tuple(
          epath.Path(os.path.dirname(parquet_path)).glob(
              os.path.basename(parquet_path)
          )
      )

      if not parquet_files and self.base_path != DEFAULT_BASE_PATH:
        self._download_and_prepare()
        parquet_files = tuple(
            epath.Path(os.path.dirname(parquet_path)).glob(
                os.path.basename(parquet_path)
            )
        )

      if not parquet_files:
        raise FileNotFoundError(f"No parquet files found for {parquet_path}")

      return sorted(os.fspath(file) for file in parquet_files)

  def _load_data(self, with_audio: bool = False) -> pd.DataFrame:
    """Loads the task data for the given task name.

    Args:
      with_audio: Whether to include the audio column.

    Returns:
      A pandas DataFrame containing the task data.

    Raises:
      FileNotFoundError: If the task file does not exist.
    """
    files = self._get_files()
    dfs = []
    for f in files:
      if self.streaming:
        with _open_hf_parquet(self.repo_id, f, token=self.token) as parquet_f:
          df = _read_parquet_file(
              parquet_f, with_audio=with_audio, filter_fn=self._filter_fn
          )
      else:
        with epath.Path(f).open("rb") as parquet_f:
          df = _read_parquet_file(
              parquet_f, with_audio=with_audio, filter_fn=self._filter_fn
          )
      dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    def _wav_bytes_to_waveform(x):
      if "bytes" in x:
        samples, sample_rate = utils.wav_bytes_to_waveform(x.get("bytes"))
        return {"samples": samples, "sample_rate": sample_rate}
      else:
        return {"samples": x["waveform"], "sample_rate": x["sample_rate"]}

    if self._data_cache is None:
      self._data_cache = df.drop(columns=["audio"], errors="ignore")

    if "audio" in df.columns:
      df["audio"] = df["audio"].apply(_wav_bytes_to_waveform)

    return df  # pyrefly: ignore[bad-return]

  def get_sound(self, record: dict[str, Any]) -> types.Sound:
    """Converts a single row of the dataset to a Sound object."""
    samples = record["audio"]["samples"]
    sample_rate = record["audio"]["sample_rate"]
    try:
      speaker_age = int(record["speaker_age"])
    except ValueError:
      speaker_age = None
    context = types.SoundContextParams(
        id=record["path"],
        sample_rate=sample_rate,
        length=len(samples),
        language=locale_by_bcp47[record["locale"]],
        speaker_id=record["speaker_id"],
        speaker_age=speaker_age,
        speaker_gender=record["speaker_sex"],
        text=record["utt"],
        waveform_start_second=0.0,
        waveform_end_second=len(samples) / sample_rate
        if sample_rate > 0
        else 0.0,
    )
    return types.Sound(waveform=samples, context=context)

  def get_task_data(
      self,
      task_name: str | None = None,
      dtype: Mapping[str, Any] | None = None,
      with_audio: bool = False,
  ) -> pd.DataFrame:
    r"""Returns the entire dataset as a DataFrame.

    Args:
      task_name: The name of the task.
      dtype: The data types of the columns.
      with_audio: Whether to include the audio column.

    Attributes with example values:
    id                           2205
    locale                       de-DE
    partition                    test
    scenario                     10
    scenario_str                 audio
    intent_idx                   46
    intent_str                   audio_volume_mute
    utt                          stille für zwei stunden
    annot_utt                    stille für [time : zwei stunden]
    worker_id                    8
    slot_method                  {'slot': ['time'], 'method': ['translation']}
    judgments                    {'worker_id': ['27', '28', '8'], 'intent_score.
    tokens                       [stille, für, zwei, stunden]
    labels                       [Other, Other, time, time]
    audio                        {'bytes': b'RIFFF\xb1\x03\x00WAVEfmt \x10\x00\.
    path                         test/c15b5445ba46918a8d678e7b59b80aa6.wav
    is_transcript_reported       False
    is_validated                 True
    speaker_id                   5f32d5f107d49607c3f6cf7a
    speaker_sex                  Female
    speaker_age                  40
    speaker_ethnicity_simple     White
    speaker_country_of_birth     Germany
    speaker_country_of_residence Germany
    speaker_nationality          Germany
    speaker_first_language       German
    """
    if with_audio:
      return self._load_data(with_audio=True)
    return self._data

  def get_task_data_beam(self, task_name: str | None = None) -> beam.PTransform:
    """Loads the task data with audio for the given task name with beam."""
    return ReadTaskData(
        filename=self.filename,
        base_path=self.base_path,
        repo_id=self.repo_id,
        streaming=self.streaming,
        token=self.token,
        files=self._get_files(),
        filter_fn=self._filter_fn,
    )

  def get_task_sounds_beam(
      self, task_name: str | None = None, locale: str | None = None
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


class _LoadFileFn(beam.DoFn):
  """Loads audio and features for a single file."""

  def __init__(self, kwargs: dict[str, Any]):
    self.kwargs = kwargs

  def process(self, filename: str):
    streaming = self.kwargs.get("streaming", False)
    repo_id = self.kwargs.get("repo_id", "FBK-MT/Speech-MASSIVE-test")
    token = self.kwargs.get("token")
    filter_fn = self.kwargs.get("filter_fn")

    if streaming:
      with _open_hf_parquet(repo_id, filename, token=token) as parquet_f:
        df = _read_parquet_file(parquet_f, with_audio=True, filter_fn=filter_fn)
    else:
      with epath.Path(filename).open("rb") as parquet_f:
        df = _read_parquet_file(parquet_f, with_audio=True, filter_fn=filter_fn)

    def _wav_bytes_to_waveform(x):
      if "bytes" in x:
        samples, sample_rate = utils.wav_bytes_to_waveform(x.get("bytes"))
        return {"samples": samples, "sample_rate": sample_rate}
      else:
        return {"samples": x["waveform"], "sample_rate": x["sample_rate"]}

    if not df.empty:
      df["audio"] = df["audio"].apply(_wav_bytes_to_waveform)

    ds = SpeechMassiveDataset(**self.kwargs)

    for record in df.to_dict("records"):
      sound = ds.get_sound(record)
      yield dict(record, sound=sound)


class ReadTaskData(beam.PTransform):
  """A PTransform that reads SpeechMassive task data and loads audio."""

  def __init__(
      self,
      filename: str,
      base_path: str | None,
      repo_id: str,
      streaming: bool,
      token: str | None,
      files: list[str],
      filter_fn: Callable[[Any], bool] | None = None,
  ):
    self._kwargs = {
        "filename": filename,
        "base_path": base_path,
        "repo_id": repo_id,
        "streaming": streaming,
        "token": token,
        "filter_fn": filter_fn,
    }
    self._files = files

  def expand(self, pcoll: beam.PCollection) -> beam.PCollection:
    return (
        pcoll
        | "CreateFiles" >> beam.Create(self._files)
        | "LoadFiles" >> beam.ParDo(_LoadFileFn(self._kwargs))
    )


def is_member_of_debug(example_id: str) -> bool:
  """Returns whether the example ID is in the debug set."""
  debug_ids = frozenset([
      "6902",
      "13734",
      "16603",
      "5757",
      "9460",
      "12722",
      "4043",
      "16455",
      "9054",
      "5037",
  ])
  return example_id in debug_ids


def is_member_of_compact(example_id: str) -> bool:
  """Returns whether the example ID is in the compact set."""

  _compact_ids: frozenset[str] = frozenset([
      "0", "10125", "10132", "10150", "10214", "10216", "10217", "10260",
      "10263", "10286", "10308", "10323", "10338", "10349", "10366", "10427",
      "1043", "10444", "10494", "10513", "10527", "10579", "10600", "10667",
      "10684", "10696", "1071", "10738", "10755", "10790", "10815", "10824",
      "10828", "10832", "1084", "10846", "10848", "10868", "10887", "1090",
      "1093", "10936", "10940", "1098", "11024", "11081", "11107", "11166",
      "11186", "11238", "11261", "11267", "11270", "11272", "11339", "11341",
      "11409", "11427", "11455", "11459", "1151", "11529", "11536", "1156",
      "1158", "11582", "11598", "11628", "11633", "11767", "11839", "11844",
      "11932", "11969", "1205", "12121", "12127", "12160", "12189", "12213",
      "12251", "12254", "12264", "12291", "12306", "12383", "12402", "12419",
      "12468", "12526", "12532", "1256", "12581", "12634", "12717", "12722",
      "12745", "12814", "12839", "12844", "12866", "12869", "12874", "12957",
      "13023", "13031", "13040", "13077", "13078", "13083", "13122", "13221",
      "13255", "13288", "13297", "13341", "13346", "13491", "13516", "13521",
      "13552", "13554", "13693", "1371", "13734", "13754", "13761", "1379",
      "13823", "13896", "13974", "13983", "14001", "14025", "14026", "14033",
      "1404", "14044", "14096", "14117", "14142", "14150", "14170", "14239",
      "14243", "1428", "14323", "14367", "14370", "14378", "14382", "14490",
      "14508", "14509", "1452", "1454", "14599", "14605", "14620", "14653",
      "14661", "14706", "14787", "14838", "14926", "15032", "15039", "15056",
      "15062", "15127", "15133", "15185", "15214", "15229", "15271", "15298",
      "15359", "15409", "15413", "15444", "15508", "1551", "15548", "15554",
      "15557", "15587", "15656", "15703", "15747", "15765", "15789", "15792",
      "15815", "15839", "15849", "15857", "15862", "15880", "15890", "1594",
      "15942", "15948", "15963", "15966", "15982", "16038", "16048", "16050",
      "16228", "16317", "16320", "16373", "16375", "16399", "1640", "16438",
      "16446", "16455", "16559", "16563", "1658", "16603", "16605", "16613",
      "16624", "16694", "16701", "16713", "16719", "16729", "1675", "16754",
      "16763", "16765", "16781", "1684", "16872", "16924", "16961", "16989",
      "16992", "16995", "17023", "17051", "17069", "17087", "17119", "17147",
      "17170", "174", "1750", "1756", "1831", "1833", "1960", "2007", "2026",
      "2082", "2094", "2134", "219", "2198", "2206", "2230", "2264", "2265",
      "2316", "2330", "2354", "2355", "2365", "2441", "2451", "2472", "2480",
      "2512", "2542", "2573", "2610", "263", "2638", "2679", "2711", "2770",
      "2774", "2780", "2798", "281", "2824", "2870", "290", "2903", "2917",
      "2959", "2962", "3029", "3050", "3091", "3096", "3101", "3174", "3207",
      "3264", "3280", "3418", "348", "3516", "3532", "3542", "3562", "3586",
      "360", "3624", "3669", "3674", "3725", "3761", "3762", "3785", "379",
      "3837", "3859", "3894", "391", "3982", "4001", "4026", "4043", "411",
      "4112", "4119", "4168", "4174", "4244", "4253", "4254", "427", "443",
      "4436", "4439", "4442", "4496", "456", "4560", "4667", "4702", "4712",
      "473", "475", "4813", "4881", "4896", "4928", "4985", "4998", "5037",
      "5043", "5050", "5118", "5134", "5157", "5201", "5225", "5286", "5333",
      "5401", "5465", "5524", "5531", "5544", "5550", "5568", "5635", "5683",
      "5691", "5703", "5757", "5774", "5902", "5997", "6024", "6041", "6055",
      "6060", "608", "6109", "6116", "614", "6172", "6181", "6248", "6289",
      "6290", "6304", "6352", "6353", "6382", "6403", "6446", "6485", "6500",
      "6561", "6618", "6623", "6646", "665", "6672", "6722", "6732", "6755",
      "6774", "6789", "6801", "6865", "6902", "6909", "6928", "6964", "6980",
      "6992", "7000", "7039", "7044", "7140", "7194", "7258", "726", "7294",
      "7295", "7316", "7467", "7556", "7570", "7664", "7697", "7712", "7751",
      "7767", "7778", "7817", "7876", "7889", "7890", "7912", "7916", "7966",
      "7981", "80", "8013", "8033", "8042", "8119", "8125", "8260", "8280",
      "8284", "8320", "8322", "8358", "8431", "8449", "8461", "8498", "8579",
      "8653", "8674", "8868", "8888", "89", "8904", "8925", "8958", "8967",
      "897", "9054", "9088", "9121", "9124", "9187", "9225", "9233", "9236",
      "9303", "9317", "936", "9426", "9460", "9506", "9542", "9554", "9640",
      "967", "9827", "9844", "9851", "9909", "9952", "9955", "9964", "9969",
  ])

  return example_id in _compact_ids
