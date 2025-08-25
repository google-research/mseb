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

"""Speech-MASSIVE dataset."""

import os
from typing import Any

from mseb import dataset
from mseb import types
from mseb import utils
import pandas as pd


class SpeechMassiveDataset(dataset.Dataset):
  """SpeechMassive dataset."""

  def __init__(
      self,
      base_path: str,
      language: str,
      split: str,
      target_sr: int | None = None,
      repo_id: str = "speechcolab/massive",
  ):
    """Initializes the dataset for a specific language and split.

    Args:
      base_path: The root directory to store/find the dataset.
      language: The language code (e.g., 'en-US', 'es-ES').
      split: The dataset split to load (e.g., 'train', 'validation', 'test').
      target_sr: If provided, all waveforms will be resampled to this rate.
      repo_id: The Hugging Face repository ID to download from.
        Defaults to the original 'speechcolab/massive'. The richer
        'FBK-MT/Speech-MASSIVE' is also supported.
    """
    self.language = language
    self.repo_id = repo_id
    super().__init__(base_path=base_path, split=split, target_sr=target_sr)

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
        homepage="https://huggingface.co/datasets/speechcolab/massive",
        version="2.1.0",
        license="cc-by-4.0",
        mseb_tasks=[
            "classification",
            "clustering",
            "reconstruction",
        ],
    )

  def _download_and_prepare(self) -> None:
    """Downloads the dataset from Hugging Face."""
    utils.download_from_hf(self.repo_id, self.base_path)

  def _load_metadata(self) -> pd.DataFrame:
    """Loads the metadata for the specified language and split."""
    self._download_and_prepare()

    lang_dir = os.path.join(self.base_path, self.language)
    metadata_path = os.path.join(lang_dir, f"{self.split}.jsonl")

    if not os.path.exists(metadata_path):
      raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    return pd.read_json(metadata_path, lines=True)

  def _get_sound(self, record: dict[str, Any]) -> types.Sound:
    """Loads a single utterance based on a metadata record."""
    audio_path = os.path.join(self.base_path, record["path"])
    waveform, sr = utils.read_audio(audio_path, target_sr=self.target_sr)

    # Safely parse speaker age, which can be a string, number, or missing.
    speaker_age_val = record.get("speaker_age")
    try:
      speaker_age_int = (
          int(speaker_age_val) if pd.notna(speaker_age_val) else None
      )
    except (ValueError, TypeError):
      speaker_age_int = None

    # Safely parse speaker gender, which is named 'speaker_sex' in some
    # versions of the dataset.
    gender_str = record.get("speaker_sex", record.get("speaker_gender"))

    context = types.SoundContextParams(
        id=str(record["id"]),
        sample_rate=sr,
        length=len(waveform),
        language=self.language,
        speaker_id=(
            str(record.get("speaker_id")) if record.get("speaker_id") else None
        ),
        speaker_age=speaker_age_int,
        speaker_gender=gender_str,
        text=record.get("utt"),
        waveform_start_second=0.0,
        waveform_end_second=len(waveform) / sr if sr > 0 else 0.0,
    )
    return types.Sound(waveform=waveform, context=context)
