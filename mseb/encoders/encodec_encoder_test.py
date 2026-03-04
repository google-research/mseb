# Copyright 2026 The MSEB Authors.
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
from typing import cast
from unittest import mock

from absl.testing import absltest
from mseb import types
from mseb.encoders import encodec_encoder
import numpy as np
import numpy.testing as npt
import pyarrow.parquet as pq
import pytest


def encodec_cache_context(name: str):
  """Creates a unique cache directory for EnCodec tests."""
  original_xdg_cache_home = os.path.join(os.path.expanduser('~'), '.cache')
  new_xdg_cache_home = os.path.join(original_xdg_cache_home, f'{name}_encodec')
  return mock.patch.dict(os.environ, {'XDG_CACHE_HOME': new_xdg_cache_home})


@pytest.mark.encodec
class EncodecJointEncoderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(encodec_cache_context(self.__class__.__name__))
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, 'testdata'
    )
    self.svq_samples = pq.ParquetFile(
        os.path.join(testdata_path, 'en_us.parquet')
    )

  def test_encode_triple_representations_sync(self):
    enc = encodec_encoder.EncodecJointEncoder(
        model_path='facebook/encodec_24khz', device='cpu'
    )
    enc.setup()

    duration = 1.0
    sr = 24000
    waveform = np.zeros(int(sr * duration), dtype=np.float32)
    params = types.SoundContextParams(
        sample_rate=sr, length=waveform.shape[0], id='test_sync'
    )
    sound = types.Sound(waveform=waveform, context=params)

    results = enc.encode([sound])
    result = cast(types.SoundEmbeddingCollection, results[0])

    proj = cast(types.SoundEmbedding, result.embeddings['projected_latents'])
    quan = cast(types.SoundEmbedding, result.embeddings['quantized_latents'])
    code = cast(types.SoundEmbedding, result.embeddings['quantized_codes'])

    # 1. Temporal Sync Check: All layers must match the symbolic grid.
    num_frames = code.embedding.shape[0]
    self.assertEqual(proj.embedding.shape[0], num_frames)
    self.assertEqual(quan.embedding.shape[0], num_frames)

    # 2. Strict Dimensionality Check:
    # All continuous layers should natively be 128 dimensions in
    # this architecture.
    self.assertEqual(
        proj.embedding.shape[1], 128,
        msg=(
            'Expected projected dimension of 128, got '
            f'{proj.embedding.shape[1]}'
        )
    )
    self.assertEqual(quan.embedding.shape[1], 128)

    # 3. Timestamp Alignment
    npt.assert_almost_equal(proj.timestamps[-1, 1], duration, decimal=2)
    npt.assert_array_almost_equal(proj.timestamps, code.timestamps)

  def test_real_speech_drift(self):
    enc = encodec_encoder.EncodecJointEncoder(device='cpu')
    enc.setup()

    svq_example = self.svq_samples.read_row_group(0)
    waveform = (
        svq_example['waveform'].to_numpy()[0].astype(np.float32) / 32767.0
    )
    params = types.SoundContextParams(
        sample_rate=48000, length=waveform.shape[0], id='real_speech'
    )
    sound = types.Sound(waveform=waveform, context=params)
    results = enc.encode([sound])
    result = cast(types.SoundEmbeddingCollection, results[0])
    p_emb = cast(
        types.SoundEmbedding, result.embeddings['projected_latents']
    ).embedding
    q_emb = cast(
        types.SoundEmbedding, result.embeddings['quantized_latents']
    ).embedding
    # Check that RVQ actually alters the continuous latents.
    # No slicing needed anymore since both arrays are guaranteed to be 128-dim.
    self.assertFalse(
        np.array_equal(p_emb, q_emb),
        msg='Quantized and Projected layers are identical (No drift detected).'
    )

  def test_variable_length_batch_sync(self):
    enc = encodec_encoder.EncodecJointEncoder(
        model_path='facebook/encodec_24khz', device='cpu'
    )
    enc.setup()

    sr = 24000
    # Create a batch with different durations to force Hugging Face
    # to apply padding
    durations = [1.0, 0.5, 1.5]
    sounds = []

    for i, dur in enumerate(durations):
      # Use random noise so the model actually processes signal, not just zeros
      waveform = np.random.randn(int(sr * dur)).astype(np.float32)
      params = types.SoundContextParams(
          sample_rate=sr, length=waveform.shape[0], id=f'test_var_{i}'
      )
      sounds.append(types.Sound(waveform=waveform, context=params))

    # Process the entire batch at once
    results = enc.encode(sounds)

    self.assertLen(results, len(durations))

    # Check that synchronization holds up for every item in the batch
    for i, result in enumerate(results):
      res = cast(types.SoundEmbeddingCollection, result)
      proj = cast(types.SoundEmbedding, res.embeddings['projected_latents'])
      quan = cast(types.SoundEmbedding, res.embeddings['quantized_latents'])
      code = cast(types.SoundEmbedding, res.embeddings['quantized_codes'])

      num_frames = code.embedding.shape[0]

      # 1. Temporal Sync Check:
      # Even with padding in the batch, all 3 arrays for this specific audio
      # must end up exactly the same length.
      self.assertEqual(
          proj.embedding.shape[0], num_frames,
          msg=f'Projected desynced for item {i} (duration {durations[i]}s)'
      )
      self.assertEqual(
          quan.embedding.shape[0], num_frames,
          msg=f'Quantized desynced for item {i} (duration {durations[i]}s)'
      )

      # 2. Strict Dimensionality Check
      self.assertEqual(proj.embedding.shape[1], 128)
      self.assertEqual(quan.embedding.shape[1], 128)


if __name__ == '__main__':
  absltest.main()
