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
from mseb import types
from mseb.datasets import simple_voice_questions
from mseb.encoders import segmentation_encoder
from mseb.encoders import whisper_encoder
import numpy as np
import numpy.testing as npt


IDF_TABLE = {
    'anatolian': 0.9,
    'shepherds': 0.8,
    'rare': 0.7,
    'national': 3.0,
    'labor': 2.0,
    'relations': 4.0,
    'board': 3.0,
}


class TextSegmenterEncoderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_input_embedding = types.SoundEmbedding(
        embedding=np.array(['national', ' labor', ' relations', ' board']),
        timestamps=np.array([[0.1, 0.5], [0.6, 1.0], [1.1, 1.5], [1.6, 2.0]]),
        context=types.SoundContextParams(
            id='test_utt',
            sample_rate=16000,
            length=32000,
            language='en',
            text='dummy'
        )
    )

  def test_spacy_retokenizer_in_isolation(self):
    words = ['national', ' labor', ' relations', ' board']
    expected_output = [
        ('national', 0, 0),
        ('labor', 1, 1),
        ('relations', 2, 2),
        ('board', 3, 3)
    ]
    retokenizer = segmentation_encoder.SpacyRetokenizer(language='en')
    actual_output = list(retokenizer.retokenize(words))
    self.assertEqual(actual_output, expected_output)

  def test_encode_selects_top_k_segments(self):
    retokenizer = segmentation_encoder.SpacyRetokenizer(language='en')
    segmenter = segmentation_encoder.TokenIDFSegmenter(IDF_TABLE, retokenizer)
    encoder = segmentation_encoder.TextSegmenterEncoder(segmenter, top_k=2)
    output_embeddings = encoder.encode([self.mock_input_embedding])
    self.assertLen(output_embeddings, 1)
    result = output_embeddings[0]
    self.assertIsInstance(result, types.SoundEmbedding)
    expected_terms = np.array(['relations', 'national'])
    expected_scores = np.array([4.0, 3.0])
    expected_timestamps = np.array([[1.1, 1.5], [0.1, 0.5]])
    npt.assert_array_equal(result.embedding, expected_terms)
    self.assertIsNotNone(result.scores)
    npt.assert_array_equal(result.scores, expected_scores)
    npt.assert_array_equal(result.timestamps, expected_timestamps)

  def test_longest_prefix_segmenter_handles_numeric_keys_from_table(self):
    japanese_idf_table_with_numbers = {
        '日本': 4.5,      # "Japan"
        2025: 1.8,        # A numeric token that pandas would read as a number
        '東京': 3.2,      # "Tokyo"
    }
    segmenter = segmentation_encoder.LongestPrefixIDFSegmenter(
        japanese_idf_table_with_numbers
    )
    # "To Japan in 2025"
    segments = list(segmenter.segment(['2025', '年', 'に', '日本', 'へ']))
    found_terms = {s[0] for s in segments}
    self.assertIn('2025', found_terms)
    self.assertIn('日本', found_terms)


class MaxIDFSegmentEncoderFactoryTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent, 'testdata'
    )
    self.mini_dataset_path = os.path.join(testdata_path, 'svq_mini')
    self.assertTrue(
        os.path.exists(self.mini_dataset_path),
        f'Mini dataset not found at {self.mini_dataset_path}. '
        'Please run create_test_dataset.py first.'
    )
    dataset = simple_voice_questions.SimpleVoiceQuestionsDataset(
        base_path=self.mini_dataset_path
    )
    self.sound_input = dataset.get_sound_by_id('utt_6844631007344632667')
    if self.sound_input.context.text:
      self.sound_input.context.text = self.sound_input.context.text.lower()

  def test_factory_creates_and_runs_real_encoder(self):
    asr_encoder = whisper_encoder.ForcedAlignmentEncoder(
        model_path='base.en', device='cpu', language='en'
    )
    cascade_encoder = segmentation_encoder.create_max_idf_segment_encoder(
        asr_encoder=asr_encoder,
        idf_table=IDF_TABLE,
        language='en',
        top_k=2
    )
    cascade_encoder.setup()
    outputs = cascade_encoder.encode([self.sound_input])
    self.assertLen(outputs, 1)
    result = outputs[0]
    self.assertIsInstance(result, types.SoundEmbedding)
    found_terms = result.embedding.tolist()
    self.assertNotEmpty(result.embedding, 'No salient terms were found.')
    self.assertIsNotNone(result.scores, 'Scores should not be None.')
    self.assertEqual(
        result.embedding.shape, result.scores.shape,
        'Shape of embeddings and scores should match.'
    )
    self.assertNotEqual(result.timestamps.tolist(), [[0.0, 0.0]])
    self.assertIn('anatolian', found_terms)
    self.assertIn('shepherds', found_terms)
    duration = (
        len(self.sound_input.waveform) / self.sound_input.context.sample_rate
    )
    for timestamp in result.timestamps:
      self.assertLessEqual(timestamp[0], timestamp[1])
      self.assertGreaterEqual(timestamp[0], 0)
      self.assertLess(timestamp[1], duration + 1.0)


if __name__ == '__main__':
  absltest.main()
