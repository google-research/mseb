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
from absl.testing import absltest
from mseb.encoders import raw_encoder
from mseb.tasks import clustering


class ClusteringTest(absltest.TestCase):

  def get_testdata_path(self, *args):
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, "testdata"
    )
    return os.path.join(testdata_path, *args)

  def test_encode_svq_beam(self):
    encoder = raw_encoder.RawEncoder(
        transform_fn=raw_encoder.spectrogram_transform,
        pooling="mean",
        frame_length=(48000 // 1000 * 25),
        frame_step=(48000 // 1000 * 10),
    )
    base_path = self.get_testdata_path()
    encoded, labels = clustering.encode_svq(base_path, encoder)
    self.assertLen(encoded, 1)
    self.assertLen(labels, 1)

if __name__ == "__main__":
  absltest.main()

