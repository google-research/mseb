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

from typing import Sequence, Tuple, Union

from absl.testing import absltest
from mseb import encoder
from mseb.evaluators import segmentation_evaluator
import numpy as np
import numpy.testing as npt


class IdentityEncoder(encoder.Encoder):

  def __init__(self,
               timestamps: list[list[list[float]]],
               embeddings: list[list[str]]):
    self.timestamps = timestamps
    self.embeddings = embeddings

  def encode(self,
             sequence: Union[str, Sequence[float]],
             context: encoder.ContextParams,
             index: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    timestamps = np.array(self.timestamps[index])
    embeddings = np.array(self.embeddings[index])
    return timestamps, embeddings


class SegmentationEvaluatorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.timestamps = [
        [[0, 1.24], [1.24, 1.4], [1.4, 1.82], [1.82, 2.52]],
        [[0, 1.10], [1.10, 1.4], [1.4, 2.01], [2.01, 2.70]],
        [[0.21, 1.24], [1.24, 1.61], [1.61, 1.82], [1.82, 2.73]],
        [[0, 1.27], [1.27, 1.43], [1.43, 2.52]]
    ]
    self.embeddings = [
        ['Massive', ' Sound', ' Embedding', ' Benchmark'],
        ['Benchmark', ' Embedding', ' Sound', ' Massive'],
        ['Massive', ' Sound', ' Embedding', ' Benchmark'],
        ['Massive', ' Speech', ' Embedding']
    ]
    self.identity_encoder = IdentityEncoder(
        timestamps=self.timestamps,
        embeddings=self.embeddings)
    self.context = encoder.ContextParams()

  def test_exact_match(self):
    evaluator = segmentation_evaluator.SegmentationEvaluator(
        self.identity_encoder, {'index': 0})
    reference_timestamps = np.array(self.timestamps[0])
    reference_embeddings = np.array(self.embeddings[0])
    scores = evaluator(np.array([1.0, 2.0, 3.0]), self.context,
                       reference_timestamps, reference_embeddings, 0.0)
    npt.assert_equal(list(scores.values()), [4, 4, 4, 4])

  def test_timestamps_hits(self):
    evaluator = segmentation_evaluator.SegmentationEvaluator(
        self.identity_encoder, {'index': 1})
    reference_timestamps = np.array(self.timestamps[0])
    reference_embeddings = np.array(self.embeddings[0])
    scores = evaluator(np.array([1.0, 2.0, 3.0]), self.context,
                       reference_timestamps, reference_embeddings, 0.2)
    npt.assert_equal(scores['TimestampsHits'], 4)
    npt.assert_equal(scores['EmbeddingsHits'], 2)
    npt.assert_equal(scores['TimestampsAndEmbeddingsHits'], 0)
    npt.assert_equal(scores['NumSegments'], 4)

  def test_embeddings_hits(self):
    evaluator = segmentation_evaluator.SegmentationEvaluator(
        self.identity_encoder, {'index': 2})
    reference_timestamps = np.array(self.timestamps[0])
    reference_embeddings = np.array(self.embeddings[0])
    scores = evaluator(np.array([1.0, 2.0, 3.0]), self.context,
                       reference_timestamps, reference_embeddings, 0.2)
    npt.assert_equal(scores['TimestampsHits'], 0)
    npt.assert_equal(scores['EmbeddingsHits'], 4)
    npt.assert_equal(scores['TimestampsAndEmbeddingsHits'], 0)
    npt.assert_equal(scores['NumSegments'], 4)

  def test_timestamps_and_embeddings_hits(self):
    evaluator = segmentation_evaluator.SegmentationEvaluator(
        self.identity_encoder, {'index': 3})
    reference_timestamps = np.array(self.timestamps[0])
    reference_embeddings = np.array(self.embeddings[0])
    scores = evaluator(np.array([1.0, 2.0, 3.0]), self.context,
                       reference_timestamps, reference_embeddings, 0.1)
    npt.assert_equal(scores['TimestampsHits'], 2)
    npt.assert_equal(scores['EmbeddingsHits'], 2)
    npt.assert_equal(scores['TimestampsAndEmbeddingsHits'], 1)
    npt.assert_equal(scores['NumSegments'], 4)

  def test_combine_scores(self):
    evaluator = segmentation_evaluator.SegmentationEvaluator(
        self.identity_encoder, {'index': 0})
    scores = [
        {
            'TimestampsHits': 4,
            'EmbeddingsHits': 4,
            'TimestampsAndEmbeddingsHits': 4,
            'NumSegments': 4,
        },
        {
            'TimestampsHits': 4,
            'EmbeddingsHits': 2,
            'TimestampsAndEmbeddingsHits': 0,
            'NumSegments': 4,
        },
        {
            'TimestampsHits': 0,
            'EmbeddingsHits': 4,
            'TimestampsAndEmbeddingsHits': 0,
            'NumSegments': 4,
        },
        {
            'TimestampsHits': 2,
            'EmbeddingsHits': 2,
            'TimestampsAndEmbeddingsHits': 1,
            'NumSegments': 4,
        },
    ]
    combined_scores = evaluator.combine_scores(scores)
    npt.assert_equal(combined_scores['TimestampsHits'], 10)
    npt.assert_equal(combined_scores['EmbeddingsHits'], 12)
    npt.assert_equal(combined_scores['TimestampsAndEmbeddingsHits'], 5)
    npt.assert_equal(combined_scores['TimestampsAccuracy'], 0.625)
    npt.assert_equal(combined_scores['EmbeddingsAccuracy'], 0.75)
    npt.assert_equal(combined_scores['TimestampsAndEmbeddingsAccuracy'], 0.3125)
    npt.assert_equal(combined_scores['NumSegments'], 16)


if __name__ == '__main__':
  absltest.main()
