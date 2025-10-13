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
import dataclasses
from typing import Sequence

import jiwer
from mseb import types
import numpy as np
from sklearn import metrics


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
    "TimestampsAndEmbeddingsAccuracy": (
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
    "mAP": (
        "Mean Average Precision, a ranking metric that evaluates detection"
        " performance based on confidence scores."
    ),
    "NDCG": (
        "Normalized Discounted Cumulative Gain. A metric that evaluates "
        "the quality of a sequence by rewarding correct terms found in the "
        "correct order."
    ),
    "WordErrorRate": (
        "Word Error Rate (WER) between the predicted and reference sequences. "
        "Lower is better."
    ),
}


def timestamps_accuracy(value: float = 0.0) -> types.Score:
  return types.Score(
      metric="TimestampsAccuracy",
      description=_METRIC_DESCRIPTIONS["TimestampsAccuracy"],
      value=value,
      min=0.0,
      max=1.0
  )


def embeddings_accuracy(value: float = 0.0) -> types.Score:
  return types.Score(
      metric="EmbeddingsAccuracy",
      description=_METRIC_DESCRIPTIONS["EmbeddingsAccuracy"],
      value=value,
      min=0.0,
      max=1.0
  )


def timestamps_and_embeddings_accuracy(value: float = 0.0) -> types.Score:
  return types.Score(
      metric="TimestampsAndEmbeddingsAccuracy",
      description=_METRIC_DESCRIPTIONS["TimestampsAndEmbeddingsAccuracy"],
      value=value,
      min=0.0,
      max=1.0
  )


def timestamps_hits(value: float = 0.0, total: float = 0.0) -> types.Score:
  return types.Score(
      metric="TimestampsHits",
      description=_METRIC_DESCRIPTIONS["TimestampsHits"],
      value=value,
      min=0.0,
      max=total
  )


def embeddings_hits(value: float = 0.0, total: float = 0.0) -> types.Score:
  return types.Score(
      metric="EmbeddingsHits",
      description=_METRIC_DESCRIPTIONS["EmbeddingsHits"],
      value=value,
      min=0.0,
      max=total
  )


def timestamps_and_embeddings_hits(
    value: float = 0.0,
    total: float = 0.0
) -> types.Score:
  return types.Score(
      metric="TimestampsAndEmbeddingsHits",
      description=_METRIC_DESCRIPTIONS["TimestampsAndEmbeddingsHits"],
      value=value,
      min=0.0,
      max=total
  )


def num_segments(value: float = 0.0) -> types.Score:
  return types.Score(
      metric="NumSegments",
      description=_METRIC_DESCRIPTIONS["NumSegments"],
      value=value,
      min=0.0,
      max=value
  )


def mean_average_precision(value: float = 0.0) -> types.Score:
  return types.Score(
      metric="mAP",
      description=_METRIC_DESCRIPTIONS["mAP"],
      value=value,
      min=0.0,
      max=1.0
  )


def normalized_discounted_cumulative_gain(value: float = 0.0) -> types.Score:
  return types.Score(
      metric="NDCG",
      description=_METRIC_DESCRIPTIONS["NDCG"],
      value=value,
      min=0.0,
      max=1.0
  )


def word_error_rate(value: float = 0.0) -> types.Score:
  return types.Score(
      metric="WordErrorRate",
      description=_METRIC_DESCRIPTIONS["WordErrorRate"],
      value=value,
      min=0.0,
      max=float("inf"),  # WER can be > 1.0
  )


@dataclasses.dataclass(frozen=True)
class Segment:
  embedding: str
  start_time: float
  end_time: float
  confidence: float = 1.0


@dataclasses.dataclass
class SegmentationReference:
  example_id: str
  segments: list[Segment]


@dataclasses.dataclass
class SegmentationScores:
  timestamps_hits: int
  embeddings_hits: int
  timestamps_and_embeddings_hits: int
  num_reference_segments: int
  ndcg: float
  edit_distance: float
  num_reference_words: int


@dataclasses.dataclass
class SegmentationScoringResult:
  per_example_scores: list[SegmentationScores]
  all_predictions_for_map: list[tuple[str, Segment]]
  ground_truths_for_map: dict[str, list[Segment]]


class SegmentationEvaluator:
  """Evaluates segmentation quality with accuracy and ranking metrics.

  This evaluator measures performance by comparing a model's predicted sequence
  of salient terms against a ground-truth reference sequence for each audio
  example. It provides a comprehensive analysis by computing three distinct
  categories of metrics.

  Core Concepts:
    - The **Reference** for an audio clip consists of a ground-truth ordered
      list of top-k salient terms and their corresponding start/end timestamps.
    - The **Prediction** from a model consists of its predicted ordered list of
      top-k salient terms, their timestamps, and a confidence or saliency
      score for each term.

  Metric Categories:
    1.  **Hit-Based Accuracy (TimestampsAccuracy, EmbeddingsAccuracy, etc.):**
        These metrics are based on a **greedy bipartite matching** algorithm
        that pairs each reference segment to the best available predicted
        segment.
        - A **time match** is successful if a predicted segment's start and end
          times are both within the `tau` tolerance of a reference segment's
          timestamps.
        - An **embedding match** is successful if the string labels of a
          predicted and reference segment are identical (case-insensitive).

    2.  **Sequence Order Metrics (WER, NDCG):**
        These metrics directly compare the two sequences of labels to measure
        ordering and transcription quality. For example, it measures the Word
        Error Rate (WER) between the predicted list
        `["tires screeech", "engine revs"]` and the reference list
        `["engine revs", "tires screech"]`.

    3.  **Ranking Metric (mAP):**
        This metric evaluates the model's ability to assign higher confidence
        scores to correct detections than to incorrect ones, across the entire
        dataset. It is agnostic to the sequence order within a single example.
  """

  def __init__(self, tau: float = 0.05):
    if tau < 0:
      raise ValueError("Time tolerance `tau` cannot be negative.")
    self.tau = tau

  def compute_scores(
      self,
      predictions: types.MultiModalEmbeddingCache,
      references: Sequence[SegmentationReference],
  ) -> SegmentationScoringResult:
    """Calculates all per-example scores and prepares data for final metrics.

    In a single pass, this method computes hit counts, NDCG, and Edit Distance
    for each example. It also gathers the necessary raw data for the subsequent
    mAP calculation in `compute_metrics`.

    Args:
      predictions: A mapping from example_id to `SoundEmbedding` objects from
        the model.
      references: A sequence of `SegmentationReference` ground-truth objects.

    Returns:
      A `SegmentationScoringResult` object containing the collected per-example
      scores and the data required for dataset-level metrics.

    Raises:
      TypeError: If any value in the `predictions` cache is not a
        `types.SoundEmbedding`.
      ValueError: If an example's number of embeddings does not match its
        number of timestamps or scores.
    """
    per_example_scores = []
    all_preds_for_map = []
    gts_for_map = collections.defaultdict(list)

    for ref in references:
      gt_labels = [seg.embedding.lower() for seg in ref.segments]
      for seg in ref.segments:
        gts_for_map[ref.example_id].append(seg)

      prediction_obj = predictions.get(ref.example_id)
      pred_segments = []
      if prediction_obj:
        if not isinstance(prediction_obj, types.SoundEmbedding):
          raise TypeError(
              "Evaluator expected a SoundEmbedding for example_id "
              f"'{ref.example_id}', but received a "
              f"{type(prediction_obj).__name__}."
          )
        embeds = prediction_obj.embedding
        scores = prediction_obj.scores
        timestamps = prediction_obj.timestamps
        if len(embeds) != len(timestamps) or (
            scores is not None and len(embeds) != len(scores)
        ):
          raise ValueError(
              f"Inconsistent lengths in example '{ref.example_id}'."
          )
        for i in range(len(embeds)):
          confidence = scores[i] if scores is not None else 1.0
          seg = Segment(
              str(embeds[i]), timestamps[i, 0], timestamps[i, 1], confidence
          )
          pred_segments.append(seg)
          if scores is not None:
            all_preds_for_map.append((ref.example_id, seg))

      pred_labels = [seg.embedding.lower() for seg in pred_segments]

      # --- Hit Count Logic ---
      ex_ts_hits, ex_emb_hits, ex_ts_emb_hits = 0, 0, 0
      if pred_segments and ref.segments:
        preds_used = [False] * len(pred_segments)
        ref_matched_on_both = [False] * len(ref.segments)
        ref_matched_on_time = [False] * len(ref.segments)
        ref_matched_on_emb = [False] * len(ref.segments)
        for i, ref_seg in enumerate(ref.segments):
          for j, pred_seg in enumerate(pred_segments):
            if preds_used[j]:
              continue
            time_match = (
                abs(ref_seg.start_time - pred_seg.start_time) <= self.tau
                and abs(ref_seg.end_time - pred_seg.end_time) <= self.tau
            )
            if (
                time_match
                and ref_seg.embedding.lower() == pred_seg.embedding.lower()
            ):
              ref_matched_on_both[i] = True
              preds_used[j] = True
              break
        for i, ref_seg in enumerate(ref.segments):
          if ref_matched_on_both[i]:
            continue
          for j, pred_seg in enumerate(pred_segments):
            if preds_used[j]:
              continue
            if (
                abs(ref_seg.start_time - pred_seg.start_time) <= self.tau
                and abs(ref_seg.end_time - pred_seg.end_time) <= self.tau
            ):
              ref_matched_on_time[i] = True
              preds_used[j] = True
              break
        for i, ref_seg in enumerate(ref.segments):
          if ref_matched_on_both[i] or ref_matched_on_time[i]:
            continue
          for j, pred_seg in enumerate(pred_segments):
            if preds_used[j]:
              continue
            if ref_seg.embedding.lower() == pred_seg.embedding.lower():
              ref_matched_on_emb[i] = True
              preds_used[j] = True
              break
        ex_ts_emb_hits = sum(ref_matched_on_both)
        ex_ts_hits = ex_ts_emb_hits + sum(ref_matched_on_time)
        ex_emb_hits = ex_ts_emb_hits + sum(ref_matched_on_emb)

      # --- Sequence Logic ---
      gt_sentence = " ".join(gt_labels)
      pred_sentence = " ".join(pred_labels)
      measures = jiwer.compute_measures(gt_sentence, pred_sentence)
      edit_dist = (
          measures["substitutions"] +
          measures["deletions"] +
          measures["insertions"]
      )
      num_ref_words = int(
          measures["hits"] +
          measures["substitutions"] +
          measures["deletions"]
      )
      dcg = sum(
          1.0 / np.log2(i + 2)
          for i, pl in enumerate(pred_labels)
          if i < len(gt_labels) and pl == gt_labels[i]
      )
      idcg = sum(1.0 / np.log2(i + 2) for i in range(len(gt_labels)))
      ndcg = dcg / idcg if idcg > 0 else 0.0

      per_example_scores.append(
          SegmentationScores(
              timestamps_hits=ex_ts_hits,
              embeddings_hits=ex_emb_hits,
              timestamps_and_embeddings_hits=ex_ts_emb_hits,
              num_reference_segments=len(ref.segments),
              ndcg=ndcg,
              edit_distance=edit_dist,
              num_reference_words=num_ref_words,
          )
      )
    return SegmentationScoringResult(
        per_example_scores=per_example_scores,
        all_predictions_for_map=all_preds_for_map,
        ground_truths_for_map=gts_for_map,
    )

  def compute_metrics(
      self,
      result: SegmentationScoringResult
  ) -> list[types.Score]:
    """Aggregates intermediate scores into the final list of metrics.

    This method takes the output of `compute_scores` and performs the final
    aggregation for per-example metrics (accuracy, NDCG, Edit Distance) and
    calculates the dataset-level mAP score.

    Args:
      result: The `SegmentationScoringResult` object returned by the
        `compute_scores` method.

    Returns:
      A list of all computed `types.Score` objects.
    """
    final_scores = []

    # --- Aggregate Per-Example Metrics ---
    total_ref_segments = sum(
        s.num_reference_segments for s in result.per_example_scores
    )
    if total_ref_segments > 0:
      total_ts_hits = sum(s.timestamps_hits for s in result.per_example_scores)
      total_emb_hits = sum(
          s.embeddings_hits for s in result.per_example_scores
      )
      total_ts_emb_hits = sum(
          s.timestamps_and_embeddings_hits for s in result.per_example_scores
      )
      mean_ndcg = np.mean([s.ndcg for s in result.per_example_scores])
      total_edit_distance = sum(
          s.edit_distance for s in result.per_example_scores
      )
      total_reference_words = sum(
          s.num_reference_words for s in result.per_example_scores
      )
      corpus_wer = (
          total_edit_distance / total_reference_words
          if total_reference_words > 0
          else 0.0
      )

      ts_acc = total_ts_hits / total_ref_segments
      emb_acc = total_emb_hits / total_ref_segments
      ts_emb_acc = total_ts_emb_hits / total_ref_segments

      final_scores.extend([
          timestamps_and_embeddings_hits(
              float(total_ts_emb_hits),
              float(total_ref_segments)
          ),
          timestamps_hits(
              float(total_ts_hits),
              float(total_ref_segments)
          ),
          embeddings_hits(
              float(total_emb_hits),
              float(total_ref_segments)
          ),
          num_segments(float(total_ref_segments)),
          timestamps_and_embeddings_accuracy(float(ts_emb_acc)),
          timestamps_accuracy(float(ts_acc)),
          embeddings_accuracy(float(emb_acc)),
          normalized_discounted_cumulative_gain(float(mean_ndcg)),
          word_error_rate(corpus_wer),
      ])

    # --- Calculate Dataset-Level mAP ---
    all_preds = result.all_predictions_for_map
    ground_truths = result.ground_truths_for_map
    if not all_preds:
      map_score = mean_average_precision(0.0)
    else:
      all_preds.sort(key=lambda x: x[1].confidence, reverse=True)
      gt_used = {
          ex_id: [False] * len(gts) for ex_id, gts in ground_truths.items()
      }
      y_true = []
      y_score = []
      for ex_id, pred_seg in all_preds:
        y_score.append(pred_seg.confidence)
        gts_for_example = ground_truths.get(ex_id, [])
        is_match_found = False
        for i, gt_seg in enumerate(gts_for_example):
          if gt_used.get(ex_id) and gt_used[ex_id][i]:
            continue
          time_match = (
              abs(pred_seg.start_time - gt_seg.start_time) <= self.tau
              and abs(pred_seg.end_time - gt_seg.end_time) <= self.tau
          )
          if (
              time_match
              and pred_seg.embedding.lower() == gt_seg.embedding.lower()
          ):
            gt_used[ex_id][i] = True
            is_match_found = True
            break
        y_true.append(1 if is_match_found else 0)

      if np.any(y_true):
        map_value = metrics.average_precision_score(
            np.array(y_true), np.array(y_score)
        )
      else:
        map_value = 0.0
      map_score = mean_average_precision(map_value)

    final_scores.append(map_score)
    return final_scores
