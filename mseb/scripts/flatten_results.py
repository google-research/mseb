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

"""Flatten leaderboard results from JSONL files."""

import dataclasses
import glob
from typing import Sequence

from absl import app
from absl import flags
from mseb import leaderboard
from mseb import task as task_lib
from mseb import tasks  # pylint: disable=unused-import
from mseb.encoders import encoder_registry

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
          result_json = line.strip()
          # Backfill metadata from registry if missing.
          result = leaderboard.LeaderboardResult.from_json(result_json)
          if not result.url:
            try:
              metadata = encoder_registry.get_encoder_metadata(result.name)
              result.url = metadata.url
            except ValueError:
              pass

          if not result.task_metadata.documentation_file:
            try:
              # task_metadata.name might be "Task/SubTask", we need "Task"
              base_task_name = result.task_metadata.name.split("/")[0]
              task_cls = task_lib.get_task_by_name(base_task_name)
              if task_cls.metadata:
                updates = {}
                if (
                    not result.task_metadata.documentation_file
                    and task_cls.metadata.documentation_file
                ):
                  updates["documentation_file"] = (
                      task_cls.metadata.documentation_file
                  )
                if (
                    not result.task_metadata.dataset_documentation_file
                    and task_cls.metadata.dataset_documentation_file
                ):
                  updates["dataset_documentation_file"] = (
                      task_cls.metadata.dataset_documentation_file
                  )

                if updates:
                  result.task_metadata = dataclasses.replace(
                      result.task_metadata, **updates
                  )
            except (ValueError, AttributeError):
              pass

          yield result.to_json()

  flattened_results = leaderboard.flatten_leaderboard_results(result_iterator())

  with open(_OUTPUT_FILE.value, "w") as f:
    leaderboard.write_dataclasses_to_jsonl(flattened_results, f)


if __name__ == "__main__":
  app.run(main)
