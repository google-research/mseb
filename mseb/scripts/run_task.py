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

"""Run evaluation of an encoder on a task.

Usage:
run_task --task SVQClustering --encoder spectrogram_25_10_mean
"""

import os
from typing import Type
from absl import app
from absl import flags
from mseb import leaderboard
from mseb import runner as runner_lib
from mseb import task as task_lib
from mseb import tasks
from mseb.encoders import encoder_registry

FLAGS = flags.FLAGS


_ENCODER = flags.DEFINE_string(
    'encoder',
    None,
    'Name of the encoder.',
    required=True,
)

_TASK = flags.DEFINE_string(
    'task',
    None,
    'Name of the task.',
    required=True,
)

_CACHE_DIR = flags.DEFINE_string(
    'cache_dir',
    None,
    'Cache directory.',
)

_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size',
    0,
    'Batch size for the encoder.',
)

_NUM_THREADS = flags.DEFINE_integer(
    'num_threads',
    1,
    'Number of threads for the runner.',
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  encoder_name = _ENCODER.value
  encoder = encoder_registry.get_encoder_metadata(encoder_name).load()
  runner = runner_lib.DirectRunner(
      encoder=encoder,
      batch_size=_BATCH_SIZE.value,
      num_threads=_NUM_THREADS.value,
      output_path=os.path.join(_CACHE_DIR.value, encoder_name),
  )
  task_cls: Type[task_lib.MSEBTask] = tasks.get_task_by_name(_TASK.value)
  task = task_cls()
  task.setup()
  results = leaderboard.run_benchmark(
      encoder_name=encoder_name, runner=runner, task=task
  )
  for result in results:
    print(result.to_json())


if __name__ == '__main__':
  app.run(main)
