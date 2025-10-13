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

from os import path

from absl.testing import absltest
from mseb import runner as runner_lib
from mseb import types
from mseb.encoders import normalized_text_encoder_with_prompt as text_encoder
import numpy as np


class RunnerTest(absltest.TestCase):

  def test_save_and_load_embeddings(self):
    basedir = self.create_tempdir().full_path
    embeddings = {
        'utt_14868079180393484423': types.TextEmbedding(
            embedding=np.array(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
            ),
            spans=np.array([[0, -1]], dtype=np.int32),
            context=types.TextContextParams(
                id='utt_14868079180393484423',
            ),
            encoding_stats=types.EncodingStats(
                input_size_bytes=1000,
                embedding_size_bytes=100,
            ),
        )
    }

    runner_lib.save_embeddings(
        output_prefix=path.join(basedir, 'embeddings'),
        embeddings=embeddings,
    )
    embeddings_loaded = runner_lib.load_embeddings(
        path.join(basedir, 'embeddings')
    )

    self.assertLen(embeddings_loaded, len(embeddings))
    self.assertCountEqual(embeddings_loaded, embeddings)
    np.testing.assert_array_equal(
        embeddings_loaded['utt_14868079180393484423'].embedding,
        embeddings['utt_14868079180393484423'].embedding,
    )
    np.testing.assert_array_equal(
        embeddings_loaded['utt_14868079180393484423'].spans,
        embeddings['utt_14868079180393484423'].spans,
    )
    self.assertEqual(
        embeddings_loaded['utt_14868079180393484423'].context,
        embeddings['utt_14868079180393484423'].context,
    )

  def test_encoder_output_type(self):
    runner = runner_lib.DirectRunner(encoder=text_encoder.MockTextEncoder())
    self.assertEqual(runner.encoder_output_type(), types.TextEmbedding)


if __name__ == '__main__':
  absltest.main()
