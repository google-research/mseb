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

import os
from typing import Any, Mapping

from etils import epath
from mseb import dataset
from mseb import types
from mseb import utils
from mseb.datasets import base
import pandas as pd


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


class SpeechMassiveDataset(base.MsebDataset):
  """SpeechMassive dataset."""

  def __init__(
      self,
      filename: str,
      base_path: str | None = None,
      repo_id: str = "FBK-MT/Speech-MASSIVE-test",
  ):
    """Initializes the dataset for a specific file pattern.

    Args:
      filename: The file name relative to the base path.
      base_path: The root directory to store/find the dataset.
      repo_id: The Hugging Face repository ID to download from. Defaults to the
        richer 'FBK-MT/Speech-MASSIVE' version, but the original
        'speechcolab/massive' is also supported.
    """
    super().__init__(base_path=base_path, split="no_used")
    self.base_path = dataset.get_base_path(self.base_path)
    self.repo_id = repo_id
    self.filename = filename
    self._data = self._load_data()

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
    utils.download_from_hf(self.repo_id, self.base_path)

  def _load_data(self) -> pd.DataFrame:
    """Loads the task data for the given task name.

    Returns:
      A pandas DataFrame containing the task data.

    Raises:
      FileNotFoundError: If the task file does not exist.
    """
    self._download_and_prepare()

    parquet_path = os.path.join(self.base_path, self.filename)
    parquet_files = tuple(
        epath.Path(os.path.dirname(parquet_path)).glob(
            os.path.basename(parquet_path)
        )
    )
    if not parquet_files:
      raise FileNotFoundError(f"No parquet files found for {parquet_path}")

    dfs = [pd.read_parquet(file) for file in parquet_files]
    df = pd.concat(dfs)

    def _wav_bytes_to_waveform(x):
      if "bytes" in x:
        samples, sample_rate = utils.wav_bytes_to_waveform(x.get("bytes"))
        return {"samples": samples, "sample_rate": sample_rate}
      else:
        return {"samples": x["waveform"], "sample_rate": x["sample_rate"]}

    df["audio"] = df["audio"].apply(_wav_bytes_to_waveform)
    return df

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
      self, task_name: str | None = None, dtype: Mapping[str, Any] | None = None
  ) -> pd.DataFrame:
    r"""Returns the entire dataset as a DataFrame.

    Args:
      task_name: The name of the task.
      dtype: The data types of the columns.

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
    return self._data
