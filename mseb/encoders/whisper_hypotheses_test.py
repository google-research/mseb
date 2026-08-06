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
from unittest import mock

from absl.testing import absltest
import librosa
from mseb import types
from mseb.encoders import whisper_hypotheses
import numpy as np
import pyarrow.parquet as pq
import pytest
import whisper


def whisper_cache_context(name: str):
  # Use a unique cache directory for each test to avoid collisions when
  # running tests in parallel via pytest.
  original_xdg_cache_home = os.path.join(os.path.expanduser('~'), '.cache')
  new_xdg_cache_home = os.path.join(original_xdg_cache_home, f'{name}_whisper')
  return mock.patch.dict(os.environ, {'XDG_CACHE_HOME': new_xdg_cache_home})


@pytest.mark.whisper
@pytest.mark.optional
class WhisperHypothesesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(whisper_cache_context(self.__class__.__name__))
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, 'testdata'
    )
    self.svq_samples = pq.ParquetFile(
        os.path.join(testdata_path, 'en_us.parquet')
    )

  def test_get_hypotheses(self):
    model = whisper.load_model('base', device='cpu')
    num_hypotheses = 5
    task = whisper_hypotheses.HypothesesDecodingTask(
        model,
        whisper.decoding.DecodingOptions(
            language='en', temperature=1.0, best_of=num_hypotheses
        ),
    )

    svq_example = self.svq_samples.read_row_group(0)
    waveform = svq_example['waveform'].to_numpy()[0]
    waveform = waveform.astype(np.float32) / 32767.0
    sample_rate = 48000
    if sample_rate != whisper.audio.SAMPLE_RATE:
      waveform = librosa.resample(
          waveform, orig_sr=sample_rate, target_sr=whisper.audio.SAMPLE_RATE
      )

    mel = whisper.audio.log_mel_spectrogram(
        waveform,
        model.dims.n_mels,
        padding=whisper.audio.N_SAMPLES,
        device='cpu',
    )
    mel = whisper.audio.pad_or_trim(mel, whisper.audio.N_FRAMES)
    mel = mel.unsqueeze(0)

    result = task.run_hypotheses(mel)

    self.assertLen(result.hypotheses, 1)
    self.assertLen(result.hypotheses[0], num_hypotheses)
    print(result.hypotheses[0])

  def test_whisper_hypotheses_encoder(self):
    num_hypotheses = 5
    encoder = whisper_hypotheses.WhisperHypothesesEncoder(
        model_path='base',
        num_hypotheses=num_hypotheses,
        device='cpu',
        temperature=0.0,
    )
    encoder.setup()

    svq_example = self.svq_samples.read_row_group(0)
    waveform = svq_example['waveform'].to_numpy()[0]
    waveform = waveform.astype(np.float32) / 32767.0
    sample_rate = 48000
    params = types.SoundContextParams(
        sample_rate=sample_rate,
        length=waveform.shape[0],
        language='en',
        id='test',
    )
    sound = types.Sound(waveform=waveform, context=params)
    results = encoder.encode([sound])
    print(results[0])
    self.assertLessEqual(len(results[0].embeddings), num_hypotheses)  # pyrefly: ignore[missing-attribute]
    self.assertNotEmpty(results[0].embeddings)  # pyrefly: ignore[missing-attribute]
    print(results[0].embeddings['hypothesis_0'].embedding)  # pyrefly: ignore[missing-attribute]
    self.assertEqual(
        results[0].embeddings['hypothesis_0'].embedding,  # pyrefly: ignore[missing-attribute]
        ['How many members does the National Labor Relations Board have?'],
    )

  def test_whisper_hypotheses_encoder_with_word_timestamps(self):
    num_hypotheses = 5
    encoder = whisper_hypotheses.WhisperHypothesesEncoder(
        model_path='base',
        num_hypotheses=num_hypotheses,
        device='cpu',
        temperature=0.0,
        word_timestamps=True,
    )
    encoder.setup()

    svq_example = self.svq_samples.read_row_group(0)
    waveform = svq_example['waveform'].to_numpy()[0]
    waveform = waveform.astype(np.float32) / 32767.0
    sample_rate = 48000
    params = types.SoundContextParams(
        sample_rate=sample_rate,
        length=waveform.shape[0],
        language='en',
        id='test',
    )
    sound = types.Sound(waveform=waveform, context=params)
    results = encoder.encode([sound])
    print(results[0])
    self.assertLessEqual(len(results[0].embeddings), num_hypotheses)  # pyrefly: ignore[missing-attribute]
    self.assertNotEmpty(results[0].embeddings)  # pyrefly: ignore[missing-attribute]
    print(results[0].embeddings['hypothesis_0'].embedding)  # pyrefly: ignore[missing-attribute]
    self.assertSameElements(
        results[0].embeddings['hypothesis_0'].embedding,  # pyrefly: ignore[missing-attribute]
        [
            'How',
            ' many',
            ' members',
            ' does',
            ' the',
            ' National',
            ' Labor',
            ' Relations',
            ' Board',
            ' have',
            '?',
        ],
    )

  def test_hypotheses_decoding_result_validation(self):
    # Valid result should not raise error
    valid_res = whisper_hypotheses.HypothesesDecodingResult(
        tokens=[],
        sum_logprobs=[[-1.0, -2.0]],
        no_speech_probs=[],
        hypotheses=[['hello', 'world']],
    )
    self.assertLen(valid_res.hypotheses, 1)

    # Mismatched outer length
    with self.assertRaises(ValueError):
      whisper_hypotheses.HypothesesDecodingResult(
          tokens=[],
          sum_logprobs=[[-1.0, -2.0]],
          no_speech_probs=[],
          hypotheses=[],
      )

    # Mismatched inner length
    with self.assertRaises(ValueError):
      whisper_hypotheses.HypothesesDecodingResult(
          tokens=[],
          sum_logprobs=[[-1.0, -2.0]],
          no_speech_probs=[],
          hypotheses=[['hello']],
      )

if __name__ == '__main__':
  absltest.main()
