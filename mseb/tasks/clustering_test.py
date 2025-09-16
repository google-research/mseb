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
from mseb import encoder as encoder_lib
from mseb import runner as runner_lib
from mseb import svq_data
from mseb.encoders import raw_encoder
from mseb.tasks import clustering

FLAGS = flags.FLAGS

# Ensure flags are parsed, for example when running with pytest
if not FLAGS.is_parsed():
  FLAGS([""])


def get_test_encoder():
  encoder = raw_encoder.RawEncoder(
      transform_fn=raw_encoder.spectrogram_transform,
      pooling="mean",
      frame_length=(48000 // 1000 * 25),
      frame_step=(48000 // 1000 * 10),
  )
  return encoder_lib.SoundEncoderAsMultiModalEncoder(encoder)


class ClusteringTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        flagsaver.flagsaver((svq_data.SVQ_BASEPATH, self.get_testdata_path()))
    )

  def get_testdata_path(self, *args):
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, "testdata"
    )
    return os.path.join(testdata_path, *args)

  def test_clustering_task(self):
    encoder = get_test_encoder()
    runner = runner_lib.DirectRunner(encoder=encoder)
    task = clustering.SVQClustering()
    self.assertEqual(
        task.sub_tasks, ["speaker_gender", "speaker_age", "speaker_id"]
    )
    embeddings = runner.run(task.sounds())
    scores = task.compute_scores(embeddings)
    self.assertLen(scores, 3)
    self.assertIn("speaker_gender", scores)
    self.assertLen(scores["speaker_gender"], 1)
    self.assertEqual(scores["speaker_gender"][0].metric, "VMeasure")
    self.assertIn("speaker_age", scores)
    self.assertIn("speaker_id", scores)


if __name__ == "__main__":
  absltest.main()
