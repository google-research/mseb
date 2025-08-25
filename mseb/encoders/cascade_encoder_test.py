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
from typing import Any, Callable
from unittest import mock

from absl.testing import absltest
from mseb import types
from mseb.encoders import cascade_encoder
from mseb.encoders import normalized_text_encoder_with_prompt as text_encoder
import numpy as np
import numpy.testing as npt
import pyarrow.parquet as pq


class MockTextEncoder(text_encoder.NormalizedTextEncoderWithPrompt):
  """A concrete implementation of TextEncoder for testing purposes."""

  def setup(self):
    pass

  def __init__(
      self,
      text_encode_fn: Callable[[str], np.ndarray] | None = None,
      normalizer: Callable[[str], str] | None = None,
      prompt_template: str | None = None,
      **kwargs: Any
  ):
    super().__init__(normalizer, prompt_template, **kwargs)
    if text_encode_fn is not None:
      self.text_encode_fn = text_encode_fn
    else:
      self.text_encode_fn = mock.MagicMock(return_value=np.zeros((2, 8)))
    self.setup = mock.MagicMock(side_effect=self._setup_impl)

  def _setup_impl(self):
    assert self.text_encode_fn is not None
    self._model_loaded = True


class CascadeEncoderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, 'testdata'
    )
    self.svq_samples = pq.ParquetFile(
        os.path.join(testdata_path, 'en_us.parquet')
    )

    sample_rate = 48000
    svq_example = self.svq_samples.read_row_group(0)
    self.waveform1 = svq_example['waveform'].to_numpy()[0][:100]
    self.waveform1 = self.waveform1.astype(np.float32) / 32767.0
    self.params1 = types.SoundContextParams(
        sample_rate=sample_rate,
        length=self.waveform1.shape[0],
        language='en',
        text='This is the transcript truth.',
        waveform_start_second=0.0,
        waveform_end_second=self.waveform1.shape[0] / sample_rate,
        id='test1',
    )
    self.sound1 = types.Sound(waveform=self.waveform1, context=self.params1)
    self.waveform2 = svq_example['waveform'].to_numpy()[0][100:]
    self.waveform2 = self.waveform2.astype(np.float32) / 32767.0
    self.params2 = types.SoundContextParams(
        sample_rate=sample_rate,
        length=self.waveform2.shape[0],
        language='en',
        text='This is another transcript truth.',
        waveform_start_second=0.0,
        waveform_end_second=self.waveform2.shape[0] / sample_rate,
        id='test2',
    )
    self.sound2 = types.Sound(waveform=self.waveform2, context=self.params2)

  def test_cascade_encoder_encode(self):
    enc = cascade_encoder.CascadeEncoder(
        model_path='dummy_model_path',
        text_encoder_cls=MockTextEncoder,
        text_encoder_kwargs={},
    )
    result = enc.encode(self.sound1)
    enc.text_encoder.setup.assert_called_once()  # pytype: disable=attribute-error
    enc.text_encoder.text_encode_fn.assert_called_once_with(  # pytype: disable=attribute-error
        ['This is the transcript truth.']
    )
    npt.assert_equal(result.timestamps.shape, [1, 2])
    npt.assert_equal(result.timestamps[0, 0] == 0.0, True)
    npt.assert_equal(
        result.timestamps[0, 1]
        == self.waveform1.shape[0] / self.params1.sample_rate,
        True,
    )
    npt.assert_equal(result.embedding, np.zeros((1, 8)))

  def test_gecko_transcript_truth_encoder_encode_batch(self):
    enc = cascade_encoder.CascadeEncoder(
        model_path='dummy_model_path',
        text_encoder_cls=MockTextEncoder,
        text_encoder_kwargs={},
    )
    result1 = enc.encode(self.sound1)
    result2 = enc.encode(self.sound2)
    results_batch = enc.encode_batch([self.sound1, self.sound2])
    enc.text_encoder.setup.assert_called_once()  # pytype: disable=attribute-error
    enc.text_encoder.text_encode_fn.assert_called_with(  # pytype: disable=attribute-error
        ['This is the transcript truth.', 'This is another transcript truth.']
    )
    npt.assert_equal(len(results_batch), 2)
    self.assertEqual(
        results_batch[0].embedding.tolist(), result1.embedding.tolist()
    )
    self.assertEqual(
        results_batch[0].timestamps.tolist(), result1.timestamps.tolist()
    )
    self.assertEqual(
        results_batch[1].embedding.tolist(), result2.embedding.tolist()
    )
    self.assertEqual(
        results_batch[1].timestamps.tolist(), result2.timestamps.tolist()
    )


if __name__ == '__main__':
  absltest.main()
