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

from typing import Type
from absl import app
from absl import flags
from mseb import leaderboard
from mseb import runner as runner_lib
from mseb import task as task_lib
from mseb import tasks
from mseb.encoders import raw_encoder

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


# TODO(tombagby): Use some generic encoder registry.
_NAME_TO_ENCODER = {
    'spectrogram_25_10_mean': raw_encoder.RawEncoder(
        transform_fn=raw_encoder.spectrogram_transform,
        pooling='mean',
        frame_length=(48000 // 1000 * 25),
        frame_step=(48000 // 1000 * 10),
    ),
}


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  encoder_name = _ENCODER.value
  encoder = _NAME_TO_ENCODER[encoder_name]
  runner = runner_lib.DirectRunner(sound_encoder=encoder)
  task_cls: Type[task_lib.MSEBTask] = tasks.get_name_to_task()[_TASK.value]
  task = task_cls()
  results = leaderboard.run_benchmark(
      encoder_name=encoder_name, runner=runner, task=task
  )
  for result in results:
    print(result.to_json())


if __name__ == '__main__':
  app.run(main)
