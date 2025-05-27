# Copyright 2024 The MSEB Authors.
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
from mseb import svq_data


class SvqDataTest(absltest.TestCase):

  def get_testdata_path(self, *args):
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent, "testdata")
    return os.path.join(testdata_path, *args)

  def test_lookup(self):
    basedir = self.get_testdata_path()
    utt_lookup = svq_data.UttLookup(basedir)
    waveform = utt_lookup("utt_14868079180393484423")
    self.assertEqual(waveform.shape, (88320,))

  def test_generate_exapmles(self):
    filepath = self.get_testdata_path("test_task.jsonl")
    self.assertTrue(os.path.exists(filepath))
    examples = list(svq_data.generate_examples(filepath))
    self.assertLen(examples, 1)
    ex = examples[0]
    self.assertEqual(ex["text"], "When did the Ottoman empire conquer Italy?")
    self.assertEqual(ex["utt_id"], "utt_14868079180393484423")
    self.assertEqual(ex["waveform"].shape, (88320,))


if __name__ == "__main__":
  absltest.main()
