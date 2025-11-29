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

  # Save the data to a parquet file.
  output_filename = f"{class_name}_{_TASK_NAME.value}.parquet"
  output_path = os.path.join(_OUTPUT_DIR.value, output_filename)
  print(f"Saving data to {output_path}")
  task_data.to_parquet(output_path)

  print(f"Saved downsampled dataset to {output_path}")


if __name__ == "__main__":
  app.run(main)
