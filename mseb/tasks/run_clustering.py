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

"""Runs clustering example on svq."""

from absl import app
from absl import flags
from mseb.encoders import raw_encoder
from mseb.tasks import clustering

FLAGS = flags.FLAGS

_SVQ_BASE_PATH = flags.DEFINE_string(
    'svq_base_path',
    None,
    'Path to data.',
    required=True,
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  encoder = raw_encoder.RawEncoder(
      transform_fn=raw_encoder.spectrogram_transform,
      pooling='mean',
      frame_length=(48000 // 1000 * 25),
      frame_step=(48000 // 1000 * 10),
  )
  task = clustering.ClusteringTask(
      sound_encoder=encoder, base_path=_SVQ_BASE_PATH.value
  )
  scores = task.run()
  print('Scores: ', scores)

if __name__ == '__main__':
  app.run(main)
