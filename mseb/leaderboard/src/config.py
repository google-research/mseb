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

"""Centralized configuration and constants for the MSEB Gradio Leaderboard."""

from __future__ import annotations

import dataclasses
import enum
import math
import os
import pathlib
from typing import Dict, List, Optional, Set

Enum = enum.Enum


class MetricDirection(Enum):
  """Orientation of a benchmark metric."""

  HIGHER_IS_BETTER = "higher"
  LOWER_IS_BETTER = "lower"

  @property
  def symbol(self) -> str:
    """Unicode arrow indicator for UI headers."""
    return "⬆️" if self == MetricDirection.HIGHER_IS_BETTER else "⬇️"


@dataclasses.dataclass(frozen=True)
class TaskSpec:
  """Specification and metadata for an MSEB task."""

  key: str
  display_name: str
  col_name: str
  primary_metric: str
  metric_direction: MetricDirection
  category: str
  description: str
  is_active: bool
  secondary_metrics: List[str] = dataclasses.field(default_factory=list)
  default_unit: str = "percentage"  # "percentage" or "raw"


# ---------------------------------------------------------------------------
# Canonical MSEB Tasks
# ---------------------------------------------------------------------------
MSEB_TASKS: Dict[str, TaskSpec] = {
    "classification": TaskSpec(
        key="classification",
        display_name="Classification",
        col_name="Classification ⬆️",
        primary_metric="Accuracy",
        metric_direction=MetricDirection.HIGHER_IS_BETTER,
        category="speech/audio",
        description=(
            "Categorize spoken audio or sound events into target classes."
        ),
        is_active=True,
        secondary_metrics=[
            "mAP",
            "Balanced Accuracy",
            "Top-5 Accuracy",
            "Weighted F1-Score",
            "Weighted Precision",
            "Weighted Recall",
            "Macro F1",
            "Micro F1",
            "Hamming Loss",
            "Subset Accuracy",
            "InvalidResultRate",
            "MissingResultRate",
        ],
        default_unit="percentage",
    ),
    "clustering": TaskSpec(
        key="clustering",
        display_name="Clustering",
        col_name="Clustering ⬆️",
        primary_metric="VMeasure",
        metric_direction=MetricDirection.HIGHER_IS_BETTER,
        category="speech/audio",
        description=(
            "Unsupervised grouping of audio embeddings (speaker gender, age,"
            " ID)."
        ),
        is_active=True,
        secondary_metrics=["Homogeneity", "Completeness"],
        default_unit="percentage",
    ),
    "reasoning": TaskSpec(
        key="reasoning",
        display_name="Reasoning",
        col_name="Reasoning ⬆️",
        primary_metric="GmeanF1",
        metric_direction=MetricDirection.HIGHER_IS_BETTER,
        category="speech",
        description=(
            "Spoken question answering and target text span identification."
        ),
        is_active=True,
        secondary_metrics=["F1", "InvalidResultRate", "MissingResultRate"],
        default_unit="percentage",
    ),
    "reranking": TaskSpec(
        key="reranking",
        display_name="Reranking",
        col_name="Reranking ⬆️",
        primary_metric="MAP",
        metric_direction=MetricDirection.HIGHER_IS_BETTER,
        category="speech",
        description=(
            "Re-ranking candidate text transcripts given spoken audio"
            " embedding."
        ),
        is_active=True,
        secondary_metrics=[
            "MRR",
            "WER",
            "CER",
            "InvalidResultRate",
            "NoResultRate",
        ],
        default_unit="percentage",
    ),
    "retrieval": TaskSpec(
        key="retrieval",
        display_name="Retrieval",
        col_name="Retrieval ⬆️",
        primary_metric="MRR",
        metric_direction=MetricDirection.HIGHER_IS_BETTER,
        category="speech/multimodal",
        description=(
            "Nearest-neighbor search across document/passage/image corpora."
        ),
        is_active=True,
        secondary_metrics=[
            "EM",
            "NDCG@10",
            "Recall@10",
            "Recall@Inf",
            "InvalidResultRate",
            "NoResultRate",
        ],
        default_unit="percentage",
    ),
    "segmentation": TaskSpec(
        key="segmentation",
        display_name="Segmentation",
        col_name="Segmentation ⬆️",
        primary_metric="TimestampsAndEmbeddingsAccuracy",
        metric_direction=MetricDirection.HIGHER_IS_BETTER,
        category="speech",
        description="Salient term boundary timestamp and label detection.",
        is_active=False,
        secondary_metrics=[
            "TimestampsAccuracy",
            "EmbeddingsAccuracy",
            "mAP",
            "NDCG",
            "WordErrorRate",
            "InvalidResultRate",
            "MissingResultRate",
        ],
        default_unit="percentage",
    ),
    "transcription": TaskSpec(
        key="transcription",
        display_name="Transcription",
        col_name="Transcription ⬇️",
        primary_metric="WER",
        metric_direction=MetricDirection.LOWER_IS_BETTER,
        category="speech",
        description=(
            "Speech-to-text automatic speech recognition across languages."
        ),
        is_active=True,
        secondary_metrics=[
            "SER",
            "NoResultRate",
            "UtteranceCount",
            "WordCount",
        ],
        default_unit="percentage",
    ),
}

ORDERED_TASK_KEYS: List[str] = [
    "classification",
    "clustering",
    "reasoning",
    "reranking",
    "retrieval",
    "segmentation",
    "transcription",
]

ACTIVE_TASK_KEYS: List[str] = [k for k, t in MSEB_TASKS.items() if t.is_active]

TASK_PRIMARY_METRICS: Dict[str, str] = {
    k: t.primary_metric for k, t in MSEB_TASKS.items()
}
TASK_PRIMARY_METRIC = TASK_PRIMARY_METRICS


# ---------------------------------------------------------------------------
# Metric Directionality Dictionary & Lookup
# ---------------------------------------------------------------------------
METRIC_DIRECTIONS: Dict[str, MetricDirection] = {
    # Higher is better (⬆️)
    "Accuracy": MetricDirection.HIGHER_IS_BETTER,
    "accuracy": MetricDirection.HIGHER_IS_BETTER,
    "mAP": MetricDirection.HIGHER_IS_BETTER,
    "map": MetricDirection.HIGHER_IS_BETTER,
    "MAP": MetricDirection.HIGHER_IS_BETTER,
    "VMeasure": MetricDirection.HIGHER_IS_BETTER,
    "vmeasure": MetricDirection.HIGHER_IS_BETTER,
    "GmeanF1": MetricDirection.HIGHER_IS_BETTER,
    "gmeanf1": MetricDirection.HIGHER_IS_BETTER,
    "F1": MetricDirection.HIGHER_IS_BETTER,
    "f1": MetricDirection.HIGHER_IS_BETTER,
    "MRR": MetricDirection.HIGHER_IS_BETTER,
    "mrr": MetricDirection.HIGHER_IS_BETTER,
    "EM": MetricDirection.HIGHER_IS_BETTER,
    "em": MetricDirection.HIGHER_IS_BETTER,
    "NDCG@10": MetricDirection.HIGHER_IS_BETTER,
    "NDCG": MetricDirection.HIGHER_IS_BETTER,
    "ndcg": MetricDirection.HIGHER_IS_BETTER,
    "Recall@10": MetricDirection.HIGHER_IS_BETTER,
    "RecallAt10": MetricDirection.HIGHER_IS_BETTER,
    "Recall@Inf": MetricDirection.HIGHER_IS_BETTER,
    "RecallAtInf": MetricDirection.HIGHER_IS_BETTER,
    "recall": MetricDirection.HIGHER_IS_BETTER,
    "Balanced Accuracy": MetricDirection.HIGHER_IS_BETTER,
    "Top-5 Accuracy": MetricDirection.HIGHER_IS_BETTER,
    "Weighted F1-Score": MetricDirection.HIGHER_IS_BETTER,
    "Weighted Precision": MetricDirection.HIGHER_IS_BETTER,
    "Weighted Recall": MetricDirection.HIGHER_IS_BETTER,
    "Macro F1": MetricDirection.HIGHER_IS_BETTER,
    "Micro F1": MetricDirection.HIGHER_IS_BETTER,
    "Subset Accuracy": MetricDirection.HIGHER_IS_BETTER,
    "Overall Accuracy": MetricDirection.HIGHER_IS_BETTER,
    "overall_accuracy": MetricDirection.HIGHER_IS_BETTER,
    "TimestampsAndEmbeddingsAccuracy": MetricDirection.HIGHER_IS_BETTER,
    "TimestampsAccuracy": MetricDirection.HIGHER_IS_BETTER,
    "timestamps_accuracy": MetricDirection.HIGHER_IS_BETTER,
    "EmbeddingsAccuracy": MetricDirection.HIGHER_IS_BETTER,
    "embeddings_accuracy": MetricDirection.HIGHER_IS_BETTER,
    "Homogeneity": MetricDirection.HIGHER_IS_BETTER,
    "Completeness": MetricDirection.HIGHER_IS_BETTER,
    # Lower is better (⬇️)
    "WER": MetricDirection.LOWER_IS_BETTER,
    "wer": MetricDirection.LOWER_IS_BETTER,
    "WordErrorRate": MetricDirection.LOWER_IS_BETTER,
    "SER": MetricDirection.LOWER_IS_BETTER,
    "ser": MetricDirection.LOWER_IS_BETTER,
    "CER": MetricDirection.LOWER_IS_BETTER,
    "cer": MetricDirection.LOWER_IS_BETTER,
    "CED": MetricDirection.LOWER_IS_BETTER,
    "ced": MetricDirection.LOWER_IS_BETTER,
    "Corpus_Mean_CED": MetricDirection.LOWER_IS_BETTER,
    "Corpus_Mean_UED": MetricDirection.LOWER_IS_BETTER,
    "Mean_Local_IS_CED": MetricDirection.LOWER_IS_BETTER,
    "Mean_Local_IS_UED": MetricDirection.LOWER_IS_BETTER,
    "Corpus_Mean_DTW": MetricDirection.LOWER_IS_BETTER,
    "dtw": MetricDirection.LOWER_IS_BETTER,
    "Corpus_Mean_L2": MetricDirection.LOWER_IS_BETTER,
    "l2": MetricDirection.LOWER_IS_BETTER,
    "ued": MetricDirection.LOWER_IS_BETTER,
    "FAD": MetricDirection.LOWER_IS_BETTER,
    "fad": MetricDirection.LOWER_IS_BETTER,
    "KAD": MetricDirection.LOWER_IS_BETTER,
    "kad": MetricDirection.LOWER_IS_BETTER,
    "Embedding MSE": MetricDirection.LOWER_IS_BETTER,
    "mse": MetricDirection.LOWER_IS_BETTER,
    "InvalidResultRate": MetricDirection.LOWER_IS_BETTER,
    "MissingResultRate": MetricDirection.LOWER_IS_BETTER,
    "NoResultRate": MetricDirection.LOWER_IS_BETTER,
    "Hamming Loss": MetricDirection.LOWER_IS_BETTER,
}

LOWER_IS_BETTER_METRICS: Set[str] = {
    "wer",
    "worderrorrate",
    "ser",
    "cer",
    "ced",
    "corpus_mean_ced",
    "corpus_mean_ued",
    "mean_local_is_ced",
    "mean_local_is_ued",
    "corpus_mean_dtw",
    "dtw",
    "corpus_mean_l2",
    "l2",
    "ued",
    "fad",
    "kad",
    "embedding mse",
    "mse",
    "invalidresultrate",
    "missingresultrate",
    "noresultrate",
    "hamming loss",
}

HIGHER_IS_BETTER_METRICS: Set[str] = {
    "accuracy",
    "map",
    "vmeasure",
    "gmeanf1",
    "f1",
    "mrr",
    "em",
    "ndcg@10",
    "ndcg",
    "recall@10",
    "recallat10",
    "recall@inf",
    "recallatinf",
    "recall",
    "balanced accuracy",
    "top-5 accuracy",
    "weighted f1-score",
    "weighted precision",
    "weighted recall",
    "macro f1",
    "micro f1",
    "subset accuracy",
    "overall accuracy",
    "overall_accuracy",
    "timestampsandembeddingsaccuracy",
    "timestampsaccuracy",
    "timestamps_accuracy",
    "embeddingsaccuracy",
    "embeddings_accuracy",
    "homogeneity",
    "completeness",
}


def get_metric_direction(metric_name: str) -> MetricDirection:
  """Returns the direction of a metric, defaulting to HIGHER_IS_BETTER."""
  clean = metric_name.strip()
  if clean in METRIC_DIRECTIONS:
    return METRIC_DIRECTIONS[clean]
  if clean.lower() in LOWER_IS_BETTER_METRICS:
    return MetricDirection.LOWER_IS_BETTER
  return MetricDirection.HIGHER_IS_BETTER


def is_lower_is_better(metric_name: str) -> bool:
  """Returns True if the metric is lower-is-better."""
  return get_metric_direction(metric_name) == MetricDirection.LOWER_IS_BETTER


def normalize_metric_for_svq(score: float, metric_name: str) -> float:
  """Normalize score for SVQ aggregation so higher is always better (0-100 scale)."""
  if score is None or math.isnan(score):
    return float("nan")

  is_lib = is_lower_is_better(metric_name)
  m_lower = metric_name.lower().strip()

  if is_lib:
    if m_lower in ("wer", "ser", "cer", "worderrorrate"):
      # Raw speech recognition error rate (fraction or percentage)
      val = score * 100.0 if score <= 1.0 and score >= 0.0 else score
      return max(0.0, 100.0 - val)
    elif m_lower in (
        "fad",
        "kad",
        "mse",
        "embedding mse",
        "ced",
        "corpus_mean_ced",
        "dtw",
        "corpus_mean_dtw",
        "l2",
        "corpus_mean_l2",
        "ued",
        "corpus_mean_ued",
    ):
      # Distance / error metric: map lower to higher score
      return max(0.0, 100.0 / (1.0 + max(0.0, score)))
    else:
      val = score * 100.0 if score <= 1.0 and score >= 0.0 else score
      return max(0.0, 100.0 - val)
  else:
    # Higher is better
    return score * 100.0 if 0.0 <= score <= 1.0 else score


def get_source_file_url(file_path: Optional[str]) -> Optional[str]:
  """Converts a source file path to a clickable URL using MSEB_URL_BASE."""
  if not file_path:
    return None
  p = str(file_path).strip()
  if not p:
    return None
  if p.startswith("http://") or p.startswith("https://"):
    return p

  base_url = os.environ.get("MSEB_URL_BASE")
  if not base_url:
    return f"file://{p}"

  results_dir = resolve_results_dir()

  if os.path.isabs(p) and p.startswith(results_dir):
    rel_path = os.path.relpath(p, results_dir)
  else:
    # Fallback if somehow not inside results_dir: just use the parent dir and
    # filename which effectively forms 'model_name/basename.jsonl'
    parent_dir = os.path.basename(os.path.dirname(p))
    rel_path = f"{parent_dir}/{os.path.basename(p)}"

  base_url = base_url.rstrip("/")
  return f"{base_url}/{rel_path.replace(os.sep, '/')}"


# ---------------------------------------------------------------------------
# Subtask Noise Conditions & Patterns
# ---------------------------------------------------------------------------
NOISE_CONDITIONS: Set[str] = {
    "clean",
    "media_noise",
    "traffic_noise",
    "background_speech",
}

CLUSTERING_TARGETS: Set[str] = {
    "speaker_gender",
    "speaker_age",
    "speaker_id",
}

RESOURCE_METRIC_KEYS: Set[str] = {
    "flops",
    "mean_encoding_size_bytes",
}


def is_noise_slice(sub_task_name: str) -> bool:
  """Returns True if sub_task_name is a noise condition slice (contains a colon)."""
  return ":" in sub_task_name


def get_noise_condition(sub_task_name: str) -> Optional[str]:
  """Extracts the noise condition string from a sub_task_name, or None if aggregate."""
  if ":" in sub_task_name:
    return sub_task_name.split(":", 1)[1]
  return None


def get_base_subtask_name(sub_task_name: str) -> str:
  """Returns the base sub-task name without the noise condition suffix."""
  if ":" in sub_task_name:
    return sub_task_name.split(":", 1)[0]
  return sub_task_name


# ---------------------------------------------------------------------------
# Filesystem Paths & Directory Resolution
# ---------------------------------------------------------------------------


def resolve_results_dir(custom_path: Optional[str] = None) -> str:
  """Returns the evaluation results directory."""
  if custom_path:
    return os.path.abspath(custom_path)

  env_path = os.environ.get("MSEB_RESULTS_DIR")
  if env_path and os.path.isdir(env_path):
    return os.path.abspath(env_path)

  # config.py is in src/, so app.py is in the parent directory
  base_dir = pathlib.Path(__file__).resolve().parent.parent
  results_dir = str(base_dir / "results")
  return results_dir


# ---------------------------------------------------------------------------
# UI Column Definitions & Metadata Taxonomies
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ColumnConfig:
  """Configuration descriptor for a leaderboard table column."""

  id: str
  label: str
  datatype: str  # "str", "number", "markdown", "bool"
  displayed_by_default: bool = True
  never_hidden: bool = False
  hidden: bool = False


OVERALL_COLUMNS: List[ColumnConfig] = [
    ColumnConfig("model_type_symbol", "T", "str", True, never_hidden=True),
    ColumnConfig("model", "Model", "markdown", True, never_hidden=True),
    ColumnConfig(
        "overall_svq", "Overall SVQ ⬆️", "number", True, never_hidden=True
    ),
    ColumnConfig("classification", "Classification ⬆️", "number", True),
    ColumnConfig("clustering", "Clustering ⬆️", "number", True),
    ColumnConfig("reasoning", "Reasoning ⬆️", "number", True),
    ColumnConfig("reranking", "Reranking ⬆️", "number", True),
    ColumnConfig("retrieval", "Retrieval ⬆️", "number", True),
    ColumnConfig("segmentation", "Segmentation ⬆️", "number", False),
    ColumnConfig("transcription", "Transcription ⬇️", "number", True),
    ColumnConfig("model_type", "Type", "str", False),
    ColumnConfig("architecture", "Architecture", "str", False),
    ColumnConfig("precision", "Precision", "str", False),
    ColumnConfig("params", "#Params (B)", "number", False),
    ColumnConfig("license", "Hub License", "str", False),
    ColumnConfig("still_on_hub", "Available on Hub", "bool", False),
]


@dataclasses.dataclass(frozen=True)
class ModelTypeDetail:
  name: str
  symbol: str


class ModelType(Enum):
  """Model training/tuning type."""

  PT = ModelTypeDetail("pretrained", "🟢")
  FT = ModelTypeDetail("fine-tuned", "🔶")
  IFT = ModelTypeDetail("instruction-tuned", "⭕")
  RL = ModelTypeDetail("RL-tuned", "🟦")
  UNKNOWN = ModelTypeDetail("", "?")

  def to_str(self, separator: str = " ") -> str:
    return f"{self.value.symbol}{separator}{self.value.name}"

  @staticmethod
  def from_str(type_str: str) -> ModelType:
    if not type_str:
      return ModelType.UNKNOWN
    type_lower = type_str.lower()
    if "fine-tuned" in type_lower or "🔶" in type_str:
      return ModelType.FT
    if "pretrained" in type_lower or "🟢" in type_str:
      return ModelType.PT
    if "rl-tuned" in type_lower or "🟦" in type_str:
      return ModelType.RL
    if "instruction-tuned" in type_lower or "⭕" in type_str:
      return ModelType.IFT
    return ModelType.UNKNOWN
