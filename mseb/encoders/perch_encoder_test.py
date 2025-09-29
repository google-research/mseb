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

from mseb import types
from mseb.encoders import perch_encoder
import numpy as np
import tensorflow as tf


class PerchEncoderTest(tf.test.TestCase):

  def test_embedding(self):
    encoder = perch_encoder.PerchEncoder()
    encoder.setup()
    waveform = np.zeros(32000 * 5, dtype=np.float32)
    sound = types.Sound(
        waveform=waveform,
        context=types.SoundContextParams(
            id='test', sample_rate=32000, length=waveform.size
        ),
    )
    embeddings = encoder.encode([sound])
    self.assertLen(embeddings, 1)
    self.assertEqual(embeddings[0].embedding.shape, (1536,))

  def test_logits(self):
    encoder = perch_encoder.PerchEncoder(embedding_type='logits')
    encoder.setup()
    waveform = np.zeros(32000 * 5, dtype=np.float32)
    sound = types.Sound(
        waveform=waveform,
        context=types.SoundContextParams(
            id='test', sample_rate=32000, length=waveform.size
        ),
    )
    embeddings = encoder.encode([sound])
    self.assertLen(embeddings, 1)
    self.assertEqual(embeddings[0].embedding.shape, (14795,))


if __name__ == '__main__':
  tf.test.main()
