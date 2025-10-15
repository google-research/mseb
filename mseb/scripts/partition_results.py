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

"""Organizes raw benchmark results into encoder/task JSONL files for submission.

This script takes a glob of JSON files, each containing one or more multi-line
JSON-serialized LeaderboardResult objects, partitions them by encoder and task
name, and writes them to a directory structure of the form:
  <output_dir>/<encoder_name>/<task_name>.jsonl
"""

import json
import logging
import os
from typing import Sequence

from absl import app
from absl import flags
from etils import epath
from mseb import leaderboard


logger = logging.getLogger(__name__)


_INPUT_GLOB = flags.DEFINE_string(
    'input_glob', None, 'Glob pattern for input JSONL files.', required=True
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    None,
    'Output directory for partitioned results.',
    required=True,
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  input_files = tuple(
      epath.Path(os.path.dirname(_INPUT_GLOB.value)).glob(
          os.path.basename(_INPUT_GLOB.value)
      )
  )
  if not input_files:
    raise FileNotFoundError(f'No files found matching {_INPUT_GLOB.value}')

  def result_iterator():
    decoder = json.JSONDecoder()
    for file_path in input_files:
      with epath.Path(file_path).open('r') as f:
        content = f.read().strip()
        pos = 0
        while pos < len(content):
          _, end_pos = decoder.raw_decode(content[pos:])
          yield content[pos : pos + end_pos]
          pos += end_pos
          while pos < len(content) and content[pos].isspace():
            pos += 1

  partitions = leaderboard.partition_leaderboard_results(
      result_iterator(),
      naming_fn=lambda r: f'{r.name}/{r.task_metadata.name}',
  )

  for partition_name, results in partitions.items():
    output_path = os.path.join(_OUTPUT_DIR.value, f'{partition_name}.jsonl')
    logger.info('Writing results to %s', output_path)
    epath.Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    with epath.Path(output_path).open('w') as f:
      leaderboard.write_dataclasses_to_jsonl(results, f)

if __name__ == '__main__':
  app.run(main)
