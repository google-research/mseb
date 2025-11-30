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

"""Downsamples a dataset and saves it as a parquet file."""

import importlib
import os

from absl import app
from absl import flags
from etils import epath
from mseb.datasets import base
import tqdm

_DATASET_CLASSNAME = flags.DEFINE_string(
    "dataset_classname",
    None,
    "The name of the dataset class to use.",
    required=True,
)
_TASK_NAME = flags.DEFINE_string(
    "task_name", None, "The name of the task to sample from.", required=True
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "The directory to save the output file to.",
    required=True,
)
_NUM_SAMPLES = flags.DEFINE_integer(
    "num_samples", 100, "The number of samples to downsample to."
)
_SPLIT = flags.DEFINE_string("split", None, "The split to use for the dataset.")
_FILTER_KEY = flags.DEFINE_string("filter_key", None, "The key to filter on.")
_FILTER_VALUE_REGEX = flags.DEFINE_string(
    "filter_value_regex", None, "The regex to filter on."
)


def main(_):
  """Script entry point."""
  # Dynamically import the dataset class.
  module_name, class_name = _DATASET_CLASSNAME.value.rsplit(".", 1)
  print(f"Loading dataset: {class_name}")
  module = importlib.import_module(f"mseb.datasets.{module_name}")
  dataset_class = getattr(module, class_name)

  # Instantiate the dataset.
  kwargs = {}
  if _SPLIT.value:
    kwargs["split"] = _SPLIT.value
  dataset: base.MsebDataset = dataset_class(**kwargs)

  # Get the task data.
  print(f"Getting task data for task: {_TASK_NAME.value}")
  task_data = dataset.get_task_data(_TASK_NAME.value)

  # Filter the data.
  if _FILTER_KEY.value:
    if _FILTER_KEY.value not in task_data.columns:
      print(f"Filter key '{_FILTER_KEY.value}' not found in dataset.")
      print(f"Available keys: {sorted(task_data.columns)}")
      return
    task_data = task_data[
        task_data[_FILTER_KEY.value].str.match(_FILTER_VALUE_REGEX.value)
    ]
    if task_data.empty:
      print(
          f"No values matched for key '{_FILTER_KEY.value}' with regex"
          f" '{_FILTER_VALUE_REGEX.value}'."
      )
      unique_values = sorted(
          dataset.get_task_data(_TASK_NAME.value)[_FILTER_KEY.value].unique()
      )
      print(f"Available values: {unique_values}")
      return

  # Sample the data.
  if len(task_data) > _NUM_SAMPLES.value:
    print(f"Sampling {len(task_data)} records down to {_NUM_SAMPLES.value}")
    task_data = task_data.sample(n=_NUM_SAMPLES.value, random_state=42)

  # Get the audio for each record.
  print("Getting audio for each record...")
  audio_data = []
  for _, record in tqdm.tqdm(task_data.iterrows(), total=len(task_data)):
    sound = dataset.get_sound(record.to_dict())
    audio_data.append({
        "waveform": sound.waveform,
        "sample_rate": sound.context.sample_rate,
    })
  task_data["audio"] = audio_data

  # Create the output directory if it does not exist.
  epath.Path(_OUTPUT_DIR.value).mkdir(parents=True, exist_ok=True)

  # Construct the output filename suffix.
  output_suffix = ""
  if _FILTER_KEY.value and _FILTER_VALUE_REGEX.value:
    output_suffix = f"_{_FILTER_KEY.value}_{_FILTER_VALUE_REGEX.value}"

  # Save the data to a parquet file.
  output_filename = f"{class_name}_{_TASK_NAME.value}{output_suffix}.parquet"
  output_path = os.path.join(_OUTPUT_DIR.value, output_filename)
  print(f"Saving data to {output_path}")
  task_data.to_parquet(output_path)

  print(f"Saved downsampled dataset to {output_path}")


if __name__ == "__main__":
  app.run(main)
