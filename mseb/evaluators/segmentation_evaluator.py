# Copyright 2024 The MSEB Authors.
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

"""Segmentation evaluators metrics include.

1.  TimestampsAccuracy: Measures the percentage of segments where both the
    predicted start and end timestamps fall within an tau distance from
    their respective reference timestamps. This metric assesses the temporal
    precision.

2.  EmbeddingsAccuracy: Measures the percentage of segments where the predicted
    embedding exactly matches the reference embedding, regardless of timestamp
    accuracy. This metric verifies if the transcribed content is correct.

3.  TimestampsAndEmbeddingAccuracy: Measures the percentage of segments where
    both the embedding is correct AND its corresponding timestamps (start and
    end) are within the tau distance of the reference. This represents
    the overall accuracy of both content and timing.
"""

from typing import Sequence, Union

from mseb import encoder
from mseb import evaluator
import numpy as np


class SegmentationEvaluator(evaluator.Evaluator):
  """Segmentation evaluator."""

  def __call__(self,
               sequence: Union[str, Sequence[float]],
               context: encoder.ContextParams,
               reference_timestamps: np.ndarray = np.empty((0, 2)),
               reference_embeddings: np.ndarray = np.empty(0),
               tau: float = 0
               ) -> dict[str, float]:
    """Evaluates segmentation quality of the encoder.


    Metrics are extracted to evaluate quality of segmentation, with respect
    to timestamp only, embedding only, and both.

    Args:
      sequence: Input sound sequence to encode. String-type sequences are
                interpreted as sound file paths.
      context: Encoder input context parameters.
      reference_timestamps: A NumPy array of shape [N, 2] representing
                            the ground truth segment start and end times.
      reference_embeddings: A NumPy array of shape [N] representing
                            the ground truth embeddings (e.g.,
                            transcribed words/sentences) corresponding to
                            reference_timestamps.
      tau: Acceptable threshold (in seconds) for timestamp start and end.
           If a candidate's timestamp start or end falls further than `tau`
           seconds from the closest reference segment's start/end, it's
           considered a missegmentation.

    Returns:
      Dictionary of four metrics:
      TimestampsHits: Measures the number of segments for which the predicted
                      timestamps are within tau of the reference timestamp.
      EmbeddingsHits: Measures the number of matches between reference's
                      and candidate's embeddings (exact string match assumed).
      TimestampsAndEmbeddingsHits: Measures the number of matches where both
                                   timestamp criteria are met AND the
                                   corresponding embedding matches.
      NumSegments: Total number of segments in the reference.

    Raises:
      ValueError: If either of reference_timestamps or reference_embeddings
                  is empty array or None.
    """
    if reference_timestamps is None or reference_timestamps.size == 0:
      raise ValueError('The reference_timestamps should not be None.')
    if reference_embeddings is None or reference_embeddings.size == 0:
      raise ValueError('The reference_embeddings should not be None.')

    candidate_timestamps, candidate_embeddings = self.sound_encoder.encode(
        sequence=sequence,
        context=context,
        **self.encode_kwargs
    )

    num_reference_segments = len(reference_timestamps)
    num_candidate_segments = len(candidate_timestamps)

    timestamps_hits = 0
    embeddings_hits = 0
    timestamps_and_embeddings_hits = 0

    if num_reference_segments == 0 or num_candidate_segments == 0:
      return {
          'TimestampsHits': 0.0,
          'EmbeddingsHits': 0.0,
          'TimestampsAndEmbeddingsHits': 0.0,
          'NumSegments': float(num_reference_segments),
      }

    for i in range(num_reference_segments):
      ref_start, ref_end = reference_timestamps[i]
      ref_emb = str(reference_embeddings[i])

      best_time_match = False
      best_embedding_match = False
      time_and_embedding_match = False

      # Find the best candidate match for the current reference segment
      # This nested loop can be optimized for large datasets
      # (e.g., using spatial trees)
      for j in range(num_candidate_segments):
        cand_start, cand_end = candidate_timestamps[j]
        cand_emb = str(candidate_embeddings[j])

        # Check timestamp match (within tau for both start and end)
        time_match = (abs(ref_start - cand_start) <= tau and
                      abs(ref_end - cand_end) <= tau)

        # Check embedding match (exact string match)
        embedding_match = (ref_emb.lower() == cand_emb.lower())

        if time_match:
          best_time_match = True
        if embedding_match:
          best_embedding_match = True
        if time_match and embedding_match and i == j:
          time_and_embedding_match = True
          break

      if best_time_match:
        timestamps_hits += 1
      if best_embedding_match:
        embeddings_hits += 1
      if time_and_embedding_match:
        timestamps_and_embeddings_hits += 1

    return {
        'TimestampsHits': float(timestamps_hits),
        'EmbeddingsHits': float(embeddings_hits),
        'TimestampsAndEmbeddingsHits': float(timestamps_and_embeddings_hits),
        'NumSegments': float(num_reference_segments),
    }

  def combine_scores(self, scores: list[dict[str, float]]) -> dict[str, float]:
    """Combines individual segmentation evaluation scores by averaging them.

    Args:
      scores: A list of dictionaries, where each dictionary contains the
              segmentation metrics for a single example (e.g., TimestampsHits,
              EmbeddingsHits, TimestampsAndEmbeddingsHits, NumSegments).

    Returns:
      A single dictionary with the averaged metrics across all examples.
      For NumSegments, it will return the sum of segments across all examples.
    """
    if not scores:
      return {
          'TimestampsHits': 0.0,
          'EmbeddingsHits': 0.0,
          'TimestampsAndEmbeddingsHits': 0.0,
          'NumSegments': 0.0,
      }

    combined_metrics: dict[str, float] = {
        'TimestampsHits': 0.0,
        'EmbeddingsHits': 0.0,
        'TimestampsAndEmbeddingsHits': 0.0,
        'NumSegments': 0.0,
        'TimestampsAccuracy': 0.0,
        'EmbeddingsAccuracy': 0.0,
        'TimestampsAndEmbeddingsAccuracy': 0.0,
    }

    for score_dict in scores:
      combined_metrics['TimestampsHits'] += score_dict.get(
          'TimestampsHits', 0.0)
      combined_metrics['EmbeddingsHits'] += score_dict.get(
          'EmbeddingsHits', 0.0)
      combined_metrics['TimestampsAndEmbeddingsHits'] += score_dict.get(
          'TimestampsAndEmbeddingsHits', 0.0)
      combined_metrics['NumSegments'] += score_dict.get(
          'NumSegments', 0.0)

    accuracy = lambda x, y: x / y
    if combined_metrics['NumSegments'] > 0:
      combined_metrics['TimestampsAccuracy'] = accuracy(
          combined_metrics['TimestampsHits'], combined_metrics['NumSegments'])
      combined_metrics['EmbeddingsAccuracy'] = accuracy(
          combined_metrics['EmbeddingsHits'], combined_metrics['NumSegments'])
      combined_metrics['TimestampsAndEmbeddingsAccuracy'] = accuracy(
          combined_metrics['TimestampsAndEmbeddingsHits'],
          combined_metrics['NumSegments'])

    return combined_metrics
