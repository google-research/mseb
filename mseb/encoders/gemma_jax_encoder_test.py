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

"""Unit tests for GemmaJaxEncoder in gemma_jax_encoder.py."""

from unittest import mock
from absl.testing import absltest
from mseb.encoders import gemma_jax_encoder
import numpy as np


class GemmaJaxEncoderTest(absltest.TestCase):

  @mock.patch('mseb.encoders.gemma_jax_encoder.gm.ckpts.load_params')
  @mock.patch('mseb.encoders.gemma_jax_encoder.gm.text.ChatSampler')
  def test_setup_and_encode(self, mock_chat_sampler, mock_load_params):
    mock_sampler_instance = mock.Mock()
    mock_chat_sampler.return_value = mock_sampler_instance
    mock_load_params.return_value = {}  # Dummy params

    # Mock sampler.chat behavior
    mock_sampler_instance.chat.return_value = 'Test response<turn|>'

    mock_model = mock.Mock()
    encoder = gemma_jax_encoder.GemmaJaxEncoder(
        model=mock_model,
        checkpoint_path='dummy_path',
    )
    encoder._setup()

    # Test encoding text only
    prompts = [('Test prompt', None)]
    results = encoder.prompt_encode_fn(prompts)

    self.assertEqual(results[0], 'Test response')
    mock_sampler_instance.chat.assert_called_once_with(
        prompt='Test prompt',
        audio=None,
    )

  @mock.patch('mseb.encoders.gemma_jax_encoder.utils.wav_bytes_to_waveform')
  @mock.patch('mseb.encoders.gemma_jax_encoder.librosa.resample')
  @mock.patch('mseb.encoders.gemma_jax_encoder.gm.ckpts.load_params')
  @mock.patch('mseb.encoders.gemma_jax_encoder.gm.text.ChatSampler')
  def test_encode_with_audio(
      self,
      mock_chat_sampler,
      mock_load_params,
      mock_resample,
      mock_wav_to_wf,
  ):
    mock_sampler_instance = mock.Mock()
    mock_chat_sampler.return_value = mock_sampler_instance
    mock_load_params.return_value = {}
    mock_sampler_instance.audio_sample_rate = 16000

    dummy_waveform = np.zeros(100)
    mock_wav_to_wf.return_value = (dummy_waveform, 16000)
    mock_resample.return_value = dummy_waveform

    mock_sampler_instance.chat.return_value = 'Test response with audio<turn|>'

    mock_model = mock.Mock()
    encoder = gemma_jax_encoder.GemmaJaxEncoder(
        model=mock_model,
        checkpoint_path='dummy_path',
    )
    encoder._setup()

    prompts = [('Test prompt', b'dummy audio bytes')]
    results = encoder.prompt_encode_fn(prompts)

    self.assertEqual(results[0], 'Test response with audio')
    mock_sampler_instance.chat.assert_called_once_with(
        prompt='Test prompt<|audio|>',
        audio=[dummy_waveform],
    )


if __name__ == '__main__':
  absltest.main()
