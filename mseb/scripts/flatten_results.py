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

"""Flatten leaderboard results from JSONL files."""

import glob
from typing import Sequence

from absl import app
from absl import flags

from mseb import leaderboard

_INPUT_GLOB = flags.DEFINE_string(
    "input_glob", None, "Glob pattern for input JSONL files.", required=True
)
_OUTPUT_FILE = flags.DEFINE_string(
    "output_file",
    None,
    "Output file for flattened results in JSONL format.",
    required=True,
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  input_files = glob.glob(_INPUT_GLOB.value)
  if not input_files:
    raise FileNotFoundError(f"No files found matching {_INPUT_GLOB.value}")

  def result_iterator():
    for file_path in input_files:
      with open(file_path, "r") as f:
        for line in f:
          yield line.strip()

  flattened_results = leaderboard.flatten_leaderboard_results(result_iterator())

  with open(_OUTPUT_FILE.value, "w") as f:
    leaderboard.write_dataclasses_to_jsonl(flattened_results, f)


if __name__ == "__main__":
  app.run(main)
