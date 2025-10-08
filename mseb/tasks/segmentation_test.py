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

import os
import pathlib
from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from mseb import dataset
from mseb import runner as runner_lib
from mseb.encoders import raw_encoder
from mseb.tasks import segmentation

FLAGS = flags.FLAGS


def get_test_encoder():
  return raw_encoder.RawEncoder(
      transform_fn=raw_encoder.spectrogram_transform,
      frame_length=(48000 // 1000 * 25),
      frame_step=(48000 // 1000 * 10),
  )


class SegmentationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        flagsaver.flagsaver(
            (dataset._DATASET_BASEPATH, self.get_testdata_path())
        )
    )

  def get_testdata_path(self, *args):
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, "testdata"
    )
    return os.path.join(testdata_path, *args)

  def test_segmentation_task_svq(self):
    encoder = get_test_encoder()
    runner = runner_lib.DirectRunner(encoder=encoder)
    task = segmentation.SegmentationTaskSVQ()
    task.setup()
    embeddings = runner.run(task.sounds())
    scores = task.compute_scores(embeddings)
    self.assertNotEmpty(scores)


if __name__ == "__main__":
  absltest.main()
