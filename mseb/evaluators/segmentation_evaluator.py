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

"""Segmentation evaluators metrics include."""

import collections
from typing import Any

from mseb import evaluator
from mseb import types
import numpy as np


_METRIC_DESCRIPTIONS: dict[str, str] = {
    "TimestampsAccuracy": (
        "Temporal Precision: The percentage of predicted segments whose start "
        "and end times are both within a specified tolerance (tau) of the "
        "reference timestamps."
    ),
    "EmbeddingsAccuracy": (
        "Content Accuracy: The percentage of predicted segments where the "
        "embedding (e.g., a transcribed label) exactly matches the reference, "
        "irrespective of its timing."
    ),
    "TimestampsAndEmbeddingAccuracy": (
        "Overall Accuracy: The percentage of segments where the embedding is "
        "correct AND its associated start/end times are within the tolerance "
        "(tau). This is the strictest metric."
    ),
    "TimestampsHits": (
        "The raw count of reference segments with a temporally aligned "
        "prediction."
    ),
    "EmbeddingsHits": (
        "The raw count of reference segments with a content-matched prediction."
    ),
    "TimestampsAndEmbeddingsHits": (
        "The raw count of segments matching in both content and time."
    ),
    "NumSegments": (
        "The total number of ground-truth segments in the reference."
    ),
}


def timestamps_accuracy_score(value: float = 0.0):
  return types.Score(
      metric="TimestampsAccuracy",
      description=_METRIC_DESCRIPTIONS["TimestampsAccuracy"],
      value=value,
      min=0,
      max=1,
  )


def embeddings_accuracy_score(value: float = 0.0):
  return types.Score(
      metric="EmbeddingsAccuracy",
      description=_METRIC_DESCRIPTIONS["EmbeddingsAccuracy"],
      value=value,
      min=0,
      max=1,
  )


def timestamps_and_embedding_accuracy_score(value: float = 0.0):
  return types.Score(
      metric="TimestampsAndEmbeddingAccuracy",
      description=_METRIC_DESCRIPTIONS["TimestampsAndEmbeddingAccuracy"],
      value=value,
      min=0,
      max=1,
  )


class SegmentationEvaluator(evaluator.SoundEmbeddingEvaluator):
  """Evaluates segmentation quality of sound encodings.

  This evaluator calculates metrics by comparing predicted segments
  (waveform_embeddings, embedding_timestamps) against a ground truth
  reference. It measures how well the encoder identifies the content
  and timing of sound events.
  """

  def __init__(self, **kwargs: Any):
    """Initializes the evaluator.

    Args:
      **kwargs: Keyword arguments for evaluation. Expected keys include:
        - `tau` (float): The acceptable tolerance in seconds for a timestamp
          to be considered a match. Defaults to 0.0.
    """
    super().__init__(**kwargs)
    self.tau = self._kwargs.get("tau", 0.0)

  def _validate_timestamps(
      self,
      timestamps: np.ndarray,
      max_length: float,
      name: str
  ):
    """Checks if all timestamps are within the valid range [0, max_length]."""
    if timestamps.size == 0:
      return  # Nothing to validate.
    if np.any(timestamps < 0):
      raise ValueError(
          f"'{name}' contains negative timestamps, which is invalid."
      )
    if np.any(timestamps > max_length):
      raise ValueError(
          f"'{name}' contains a timestamp ({np.max(timestamps):.2f}s) that"
          f" exceeds the total waveform length ({max_length:.2f}s)."
      )
    if np.any(timestamps[:, 0] > timestamps[:, 1]):
      raise ValueError(
          f"'{name}' contains entries where start_time > end_time."
      )

  def evaluate(
      self,
      waveform_embeddings: np.ndarray,
      embedding_timestamps: np.ndarray,
      params: types.SoundContextParams,
      **kwargs: Any,
  ) -> list[types.Score]:
    """Evaluates segmentation quality for a single example.

    Args:
      waveform_embeddings: A 2D array from the encoder.
      embedding_timestamps: A 2D array of [start, end] sample seconds.
      params: The waveform context parameters, used as a source of ground truth.
      **kwargs: Additional runtime arguments. MUST contain:
        - `reference_waveform_embeddings` (np.ndarray): Ground truth
          waveform_embeddings.
        - `reference_embedding_timestamps` (np.ndarray): Ground truth
          embedding_timestamps.
        - `tau` (Optional[float]): Overrides the init `tau` value.

    Returns:
      A list of Score objects containing the raw hit counts and total
      segments for this single example.
    """
    reference_embeddings = kwargs.get("reference_waveform_embeddings")
    reference_timestamps = kwargs.get("reference_embedding_timestamps")

    if reference_embeddings is None or reference_embeddings.size == 0:
      raise ValueError(
          "Missing required kwarg: `reference_waveform_embeddings`."
      )
    if reference_timestamps is None or reference_timestamps.size == 0:
      raise ValueError(
          "Missing required kwarg: `reference_embedding_timestamps`."
      )

    max_duration_seconds = params.length / params.sample_rate
    self._validate_timestamps(
        embedding_timestamps,
        max_duration_seconds,
        "Predicted `embedding_timestamps`"
    )
    self._validate_timestamps(
        reference_timestamps,
        max_duration_seconds,
        "Ground-truth `reference_embedding_timestamps`",
    )

    tau_seconds = kwargs.get("tau", self.tau)
    num_reference_segments = len(reference_timestamps)
    num_candidate_segments = len(embedding_timestamps)
    timestamps_hits, embeddings_hits, timestamps_and_embeddings_hits = 0, 0, 0

    if num_reference_segments > 0 and num_candidate_segments > 0:
      cand_embeds_str = [str(e).lower() for e in waveform_embeddings]
      for i in range(num_reference_segments):
        ref_start, ref_end = reference_timestamps[i]
        ref_emb_str = str(reference_embeddings[i]).lower()

        if i < num_candidate_segments:
          cand_start, cand_end = embedding_timestamps[i]
          time_match = (
              abs(ref_start - cand_start) <= tau_seconds
              and abs(ref_end - cand_end) <= tau_seconds
          )
          embedding_match = ref_emb_str == cand_embeds_str[i]
          if time_match and embedding_match:
            timestamps_and_embeddings_hits += 1

        if any(
            (abs(ref_start - cs) <= tau_seconds
             and abs(ref_end - ce) <= tau_seconds)
            for cs, ce in embedding_timestamps
        ):
          timestamps_hits += 1

        if ref_emb_str in cand_embeds_str:
          embeddings_hits += 1

    max_val = float(num_reference_segments)
    return [
        types.Score(
            metric="TimestampsHits",
            description=_METRIC_DESCRIPTIONS["TimestampsHits"],
            value=float(timestamps_hits),
            min=0.0,
            max=max_val
        ),
        types.Score(
            metric="EmbeddingsHits",
            description=_METRIC_DESCRIPTIONS["EmbeddingsHits"],
            value=float(embeddings_hits),
            min=0.0,
            max=max_val
        ),
        types.Score(
            metric="TimestampsAndEmbeddingsHits",
            description=_METRIC_DESCRIPTIONS["TimestampsAndEmbeddingsHits"],
            value=float(timestamps_and_embeddings_hits),
            min=0.0,
            max=max_val
        ),
        types.Score(
            metric="NumSegments",
            description=_METRIC_DESCRIPTIONS["NumSegments"],
            value=max_val,
            min=0.0,
            max=max_val
        ),
    ]

  def combine_scores(
      self, scores_per_example: list[list[types.Score]]
  ) -> list[types.Score]:
    """Combines raw hit counts from all examples and computes final accuracy."""
    if not scores_per_example:
      return []

    aggregated_counts = collections.defaultdict(float)
    for example_scores in scores_per_example:
      for score in example_scores:
        if score.metric in _METRIC_DESCRIPTIONS:
          aggregated_counts[score.metric] += score.value

    final_scores = []
    total_segments = aggregated_counts.get("NumSegments", 0.0)

    for name in ["TimestampsHits",
                 "EmbeddingsHits",
                 "TimestampsAndEmbeddingsHits",
                 "NumSegments"]:
      value = aggregated_counts.get(name, 0.0)
      final_scores.append(
          types.Score(
              metric=name,
              description=_METRIC_DESCRIPTIONS[name],
              value=value,
              min=0.0,
              max=total_segments
          )
      )
    accuracy_names = [
        "TimestampsAccuracy",
        "EmbeddingsAccuracy",
        "TimestampsAndEmbeddingAccuracy"
    ]
    if total_segments > 0:
      final_scores.extend([
          types.Score(
              metric="TimestampsAccuracy",
              description=_METRIC_DESCRIPTIONS["TimestampsAccuracy"],
              value=aggregated_counts["TimestampsHits"] / total_segments,
              min=0.0,
              max=1.0
          ),
          types.Score(
              metric="EmbeddingsAccuracy",
              description=_METRIC_DESCRIPTIONS["EmbeddingsAccuracy"],
              value=aggregated_counts["EmbeddingsHits"] / total_segments,
              min=0.0,
              max=1.0
          ),
          types.Score(
              metric="TimestampsAndEmbeddingAccuracy",
              description=_METRIC_DESCRIPTIONS[
                  "TimestampsAndEmbeddingAccuracy"],
              value=(
                  aggregated_counts["TimestampsAndEmbeddingsHits"]
                  / total_segments
              ),
              min=0.0,
              max=1.0
          ),
      ])
    else:
      for name in accuracy_names:
        final_scores.append(
            types.Score(
                metric=name,
                description=_METRIC_DESCRIPTIONS[name],
                value=0.0,
                min=0.0,
                max=1.0
            )
        )

    return final_scores
