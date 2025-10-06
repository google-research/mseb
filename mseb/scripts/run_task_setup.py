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

r"""Run task setup.

This script is used to setup a task, such as creating embeddings cache or the
index for retrieval tasks.

Usage:
run_task_setup --encoder gecko_whisper_or_gecko \
    --task SVQEnUsPassageInLangRetrieval \
    --batch_size 2 \
    --cache_basepath /tmp/mseb_cache/gecko_whisper_or_gecko
"""

from typing import Type
from absl import app
from absl import flags
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

_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size',
    0,
    'Batch size for the encoder.',
)

_NUM_THREADS = flags.DEFINE_integer(
    'num_threads',
    1,
    'Number of threads for the encoder.',
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  encoder_name = _ENCODER.value
  encoder = encoder_registry.get_encoder_metadata(encoder_name).load()
  task_cls: Type[task_lib.MSEBTask] = tasks.get_task_by_name(_TASK.value)
  task = task_cls()
  runner = runner_lib.DirectRunner(
      encoder=encoder,
      batch_size=_BATCH_SIZE.value,
      num_threads=_NUM_THREADS.value,
      output_path=runner_lib.RUNNER_CACHE_BASEPATH.value,
  )
  task.setup(runner=runner)


if __name__ == '__main__':
  app.run(main)
