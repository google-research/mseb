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

"""Birdset dataset."""

import json
import os

from mseb import dataset
from mseb import types
import pandas as pd


class BirdsetDataset:
  """Birdset dataset for bird sound classification.

  This class loads the Birdset, a large-scale collection of bird sound
  datasets standardized to the eBird taxonomy. This dataset is written
  to use data cached from the huggingface dataset, which has not been updated to
  remove remote python builder dependencies.
  """

  def __init__(
      self,
      base_path: str | None = None,
      split: str = "test",
      configuration: str = "AWC",
  ):
    """Initializes the dataset for a specific split and configuration.

    Args:
      base_path: The root directory to store/find the HF dataset cache.
      split: The dataset split to load. Must be 'train', 'test', or 'test_5s'.
      configuration: The Birdset configuration (sub-dataset) to load,
        e.g., 'HSN', 'PER', 'XCL'.
    """
    if split not in ["train", "test", "test_5s"]:
      raise ValueError(
          f"Split must be 'train', 'test', or 'test_5s', but got '{split}'."
      )

    self.base_path = dataset.get_base_path(base_path)
    self.split = split
    self.configuration = configuration
    self._native_sr = 32_000
    self._ebird_code_names: list[str] | None = None
    self._data = self._load_data()

  @property
  def metadata(self) -> types.DatasetMetadata:
    """Returns the structured metadata for the Birdset dataset."""
    return types.DatasetMetadata(
        name="Birdset",
        description=(
            "A meta-dataset of bird sound recordings, aggregating many "
            "datasets and standardizing them to the eBird 2021 taxonomy. "
            "Managed via the Hugging Face 'datasets' library."
        ),
        homepage="https://huggingface.co/datasets/DBD-research-group/BirdSet",
        version="1.0.0",
        license="See documentation; varies by clip.",
        mseb_tasks=["classification", "clustering", "retrieval"],
        citation="""
@misc{rauch2025birdsetlargescaledatasetaudio,
  title         = {BirdSet: A Large-Scale Dataset for Audio
                   Classification in Avian Bioacoustics},
  author        = {Lukas Rauch and
                   Raphael Schwinger and
                   Moritz Wirth and
                   René Heinrich and
                   Denis Huseljic and
                   Marek Herde and
                   Jonas Lange and
                   Stefan Kahl and
                   Bernhard Sick and
                   Sven Tomforde and
                   Christoph Scholz},
  year  =         {2025},
  eprint=         {2403.10380},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
  url           = {https://arxiv.org/abs/2403.10380},
}
""",
    )

  def __len__(self) -> int:
    return len(self._data)

  def _load_data(self) -> pd.DataFrame:
    """Loads and configures the Hugging Face dataset object."""
    cache_filename = f"birdset_{self.configuration}_{self.split}.parquet"
    cache_path = os.path.join(self.base_path, cache_filename)
    df = pd.read_parquet(cache_path)
    class_lists_path = os.path.join(self.base_path, "class_lists.json")
    with open(class_lists_path, "r") as f:
      class_lists = json.load(f)
    config_to_class_list_path = os.path.join(
        self.base_path, "config_to_class_list.json"
    )
    with open(config_to_class_list_path, "r") as f:
      config_to_class_list = json.load(f)
    class_list_name = config_to_class_list[self.configuration]
    self._ebird_code_names = class_lists[class_list_name]
    return df

  def get_task_data(self) -> pd.DataFrame:
    """Returns the entire dataset as a DataFrame."""
    return self._data

  def get_sound(self, record: pd.Series) -> types.Sound:
    """Creates a Sound object from the HF dataset record."""
    audio_data = record.audio
    waveform = audio_data["waveform"]
    sr = audio_data["sampling_rate"]

    # Get the string label for the ebird_code
    text_label = None
    if "ebird_code" in record and record.ebird_code is not None:
      if self._ebird_code_names is not None:
        text_label = self._ebird_code_names[record.ebird_code]

    context = types.SoundContextParams(
        id=str(record.filepath),
        sample_rate=sr,
        length=len(waveform),
        language=None,
        speaker_id=None,
        speaker_age=None,
        speaker_gender=record.get("sex"),
        text=text_label,
        waveform_start_second=record.get("start_time", 0.0),
        waveform_end_second=record.get(
            "end_time", len(waveform) / sr if sr > 0 else 0.0
        ),
    )
    return types.Sound(waveform=waveform, context=context)
