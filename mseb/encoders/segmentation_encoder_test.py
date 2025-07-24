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
from absl.testing import parameterized
from mseb import encoder
from mseb.encoders import segmentation_encoder
from mseb.encoders import whisper_encoder
import numpy as np
import numpy.testing as npt
import pyarrow.parquet as pq
import whisper


IDF_TABLE = {
    'national': 3.0,
    'labor': 2.0,
    'relations': 4.0,
    'board': 3.0,
}


class SegmentationEncoderTests(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, 'testdata')
    self.svq_samples = pq.ParquetFile(
        os.path.join(testdata_path, 'en_us.parquet'))
    self.model = whisper.load_model('base', device='cpu')
    self.whisper_encoder = whisper_encoder.SpeechToTextEncoder(self.model)
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
        self.whisper_encoder, segmenter
    )
    svq_example = self.svq_samples.read_row_group(0)
    waveform = svq_example['waveform'].to_numpy()[0]
    waveform = waveform.astype(np.float32) / 32767.0
    sample_rate = 48000
    context = encoder.ContextParams(
        language='en', sample_rate=sample_rate,
        text=svq_example['text'].to_numpy()[0])
    timestamps, embeddings = seg_encoder.encode(
        waveform, context, **self.encode_kwargs)
    npt.assert_equal(timestamps.shape, [1, 2])
    npt.assert_equal(timestamps, [[3.58, 4.08]])
    npt.assert_equal(timestamps[0, 1] <= waveform.shape[0] / sample_rate, True)
    npt.assert_equal(embeddings, [['relations', 4.0]])


class SegmentationEncoderUsingTruthTests(SegmentationEncoderTests):

  def setUp(self):
    super().setUp()
    self.whisper_encoder = whisper_encoder.ForcedAlignmentEncoder(
        self.model, 'en'
    )
    self.encode_kwargs = {}


if __name__ == '__main__':
  absltest.main()
