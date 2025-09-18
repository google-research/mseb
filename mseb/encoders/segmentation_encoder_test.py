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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from mseb import types
from mseb.encoders import segmentation_encoder
from mseb.encoders import whisper_encoder
import numpy as np
import numpy.testing as npt
import pyarrow.parquet as pq


IDF_TABLE = {
    'national': 3.0,
    'labor': 2.0,
    'relations': 4.0,
    'board': 3.0,
}


def whisper_cache_context(name: str):
  # Use a unique cache directory for each test to avoid collisions when
  # running tests in parallel via pytest.
  original_xdg_cache_home = os.path.join(os.path.expanduser('~'), '.cache')
  new_xdg_cache_home = os.path.join(original_xdg_cache_home, f'{name}_whisper')
  return mock.patch.dict(os.environ, {'XDG_CACHE_HOME': new_xdg_cache_home})


class SegmentationEncoderTests(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(whisper_cache_context(self.__class__.__name__))
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, 'testdata')
    self.svq_samples = pq.ParquetFile(
        os.path.join(testdata_path, 'en_us.parquet'))
    self.whisper_encoder = whisper_encoder.SpeechToTextEncoder(
        model_path='base', device='cpu'
    )
    self.encode_kwargs = {'word_timestamps': True}

  @parameterized.named_parameters(
      dict(
          testcase_name='spacy_retokenizer',
          segmenter=segmentation_encoder.TokenIDFSegmenter(
              IDF_TABLE, segmentation_encoder.SpacyRetokenizer()
          ),
      ),
      dict(
          testcase_name='normalizing_retokenizer',
          segmenter=segmentation_encoder.TokenIDFSegmenter(
              IDF_TABLE, segmentation_encoder.NormalizingRetokenizer()
          ),
      ),
      dict(
          testcase_name='longest_prefix_idf_segmenter',
          segmenter=segmentation_encoder.LongestPrefixIDFSegmenter(IDF_TABLE),
      ),
  )
  def test_encode(self, segmenter):
    seg_encoder = segmentation_encoder.CascadedSegmentationEncoder(
        self.whisper_encoder, segmenter, top_k=2
    )
    svq_example = self.svq_samples.read_row_group(0)
    waveform = svq_example['waveform'].to_numpy()[0]
    waveform = waveform.astype(np.float32) / 32767.0
    sample_rate = 48000
    context = types.SoundContextParams(
        id='0',
        length=waveform.shape[0],
        language='en',
        sample_rate=sample_rate,
        text=svq_example['text'].to_numpy()[0],
    )
    sound_embedding = seg_encoder.encode(
        types.Sound(waveform=waveform, context=context), **self.encode_kwargs)
    npt.assert_equal(sound_embedding.timestamps.shape, [2, 2])
    npt.assert_array_almost_equal(
        sound_embedding.timestamps, [[3.58, 4.08], [2.76, 3.2]], decimal=1
    )
    npt.assert_equal(
        sound_embedding.timestamps[0, 1] <= waveform.shape[0] / sample_rate,
        True,
    )
    npt.assert_equal(
        sound_embedding.timestamps[1, 1] <= waveform.shape[0] / sample_rate,
        True,
    )
    npt.assert_equal(
        sound_embedding.embedding, [['relations', 4.0], ['national', 3.0]]
    )


class SegmentationEncoderUsingTruthTests(SegmentationEncoderTests):

  def setUp(self):
    super().setUp()
    self.whisper_encoder = whisper_encoder.ForcedAlignmentEncoder(
        model_path='base', device='cpu', language='en'
    )
    self.encode_kwargs = {}

if __name__ == '__main__':
  absltest.main()
