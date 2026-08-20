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

"""Typed data structures and models for MSEB Leaderboard."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Union


@dataclasses.dataclass
class ModelEntry:
  """Represents a distinct evaluated model or subconfiguration row entry."""

  entry_id: str  # e.g. "gemini-2.5-flash/asr=truth"
  top_model: str  # e.g. "gemini-2.5-flash"
  sub_config: Optional[str] = None  # e.g. "asr=truth" or None
  display_name: str = ""  # e.g. "gemini-2.5-flash (asr=truth)"
  url: str = ""  # Model card / documentation URL
  dir_path: str = ""  # Absolute filesystem path
  jsonl_files: Any = dataclasses.field(
      default_factory=list
  )  # list or tuple of .jsonl files
  tags: List[str] = dataclasses.field(default_factory=list)
  metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

  def __post_init__(self):
    if not self.display_name:
      if self.sub_config:
        self.display_name = f"{self.top_model} ({self.sub_config})"
      else:
        self.display_name = self.top_model
    if isinstance(self.jsonl_files, tuple):
      self.jsonl_files = list(self.jsonl_files)

  @property
  def has_sub_config(self) -> bool:
    """Returns True if this model entry represents a subfolder configuration."""
    return self.sub_config is not None

  @property
  def is_asr_variant(self) -> bool:
    """Returns True if this model entry is an ASR sub-configuration variant."""
    return self.sub_config is not None

  @property
  def markdown_link(self) -> str:
    """Returns a clickable Markdown link for the model in UI tables."""
    if self.url and (
        self.url.startswith("http://") or self.url.startswith("https://")
    ):
      return f"[{self.display_name}]({self.url})"
    return self.display_name


@dataclasses.dataclass(frozen=True)
class ScoreSpec:
  """Represents a single metric score measurement."""

  metric: str = ""
  description: str = ""
  value: float = 0.0
  min: Union[float, int] = 0
  max: Union[float, int] = 1
  std: Optional[float] = None

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> ScoreSpec:
    """Deserializes a ScoreSpec from a JSON dict."""
    if not isinstance(data, dict):
      return cls()

    metric_raw = data.get("metric")
    metric_str = str(metric_raw) if metric_raw is not None else ""

    desc_raw = data.get("description")
    desc_str = str(desc_raw) if desc_raw is not None else ""

    val_raw = data.get("value")
    try:
      val = float(val_raw) if val_raw is not None else 0.0
    except (ValueError, TypeError):
      val = 0.0

    min_raw = data.get("min")
    try:
      min_val = float(min_raw) if min_raw is not None else 0
    except (ValueError, TypeError):
      min_val = 0

    max_raw = data.get("max")
    try:
      max_val = float(max_raw) if max_raw is not None else 1
    except (ValueError, TypeError):
      max_val = 1

    std_raw = data.get("std")
    try:
      std_val = float(std_raw) if std_raw is not None else None
    except (ValueError, TypeError):
      std_val = None

    return cls(
        metric=metric_str,
        description=desc_str,
        value=val,
        min=min_val,
        max=max_val,
        std=std_val,
    )

  def to_dict(self) -> Dict[str, Any]:
    """Serializes ScoreSpec to a JSON-compatible dict."""
    return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class DatasetMetadata:
  """Metadata describing dataset source and revision."""

  path: str = ""
  revision: str = "1.0.0"
  documentation_file: Optional[str] = None
  name: Optional[str] = None

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> DatasetMetadata:
    """Deserializes a DatasetMetadata from a JSON dict."""
    if not isinstance(data, dict):
      return cls(path="")
    path_raw = data.get("path")
    path_str = str(path_raw) if path_raw is not None else ""

    rev_raw = data.get("revision")
    rev_str = str(rev_raw) if rev_raw is not None else "1.0.0"

    doc_raw = data.get("documentation_file")
    doc_str = str(doc_raw) if doc_raw is not None else None

    name_raw = data.get("name")
    name_str = str(name_raw) if name_raw is not None else None

    return cls(
        path=path_str,
        revision=rev_str,
        documentation_file=doc_str,
        name=name_str,
    )

  def to_dict(self) -> Dict[str, Any]:
    return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class TaskMetadata:
  """Metadata describing an MSEB benchmark evaluation task."""

  name: str = ""
  description: str = ""
  reference: str = ""
  type: str = ""
  category: str = "speech"
  main_score: str = "Accuracy"
  revision: str = "1.0.0"
  dataset: Optional[DatasetMetadata] = None
  scores: List[ScoreSpec] = dataclasses.field(default_factory=list)
  eval_splits: List[str] = dataclasses.field(default_factory=list)
  eval_langs: List[str] = dataclasses.field(default_factory=list)
  domains: List[str] = dataclasses.field(default_factory=list)
  task_subtypes: List[str] = dataclasses.field(default_factory=list)
  documentation_file: Optional[str] = None
  dataset_documentation_file: Optional[str] = None

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> TaskMetadata:
    """Deserializes a TaskMetadata from a JSON dict."""
    if not isinstance(data, dict):
      return cls(name="")

    ds_raw = data.get("dataset")
    ds_meta = (
        DatasetMetadata.from_dict(ds_raw) if isinstance(ds_raw, dict) else None
    )

    raw_scores = data.get("scores")
    if isinstance(raw_scores, list):
      scores = [
          ScoreSpec.from_dict(s) for s in raw_scores if isinstance(s, dict)
      ]
    else:
      scores = []

    def _safe_str_list(raw_val: Any) -> List[str]:
      if isinstance(raw_val, (list, tuple)):
        return [str(x) for x in raw_val if x is not None]
      return []

    eval_splits = _safe_str_list(data.get("eval_splits"))
    eval_langs = _safe_str_list(data.get("eval_langs"))
    domains = _safe_str_list(data.get("domains"))
    task_subtypes = _safe_str_list(data.get("task_subtypes"))

    name_raw = data.get("name")
    name_str = str(name_raw) if name_raw is not None else ""

    desc_raw = data.get("description")
    desc_str = str(desc_raw) if desc_raw is not None else ""

    ref_raw = data.get("reference")
    ref_str = str(ref_raw) if ref_raw is not None else ""

    type_raw = data.get("type")
    if isinstance(type_raw, (list, tuple)) and type_raw:
      type_str = str(type_raw[0])
    elif type_raw is not None:
      type_str = str(type_raw)
    else:
      type_str = ""

    cat_raw = data.get("category")
    cat_str = str(cat_raw) if cat_raw is not None else "speech"

    main_score_raw = data.get("main_score")
    main_score_str = (
        str(main_score_raw) if main_score_raw is not None else "Accuracy"
    )

    rev_raw = data.get("revision")
    rev_str = str(rev_raw) if rev_raw is not None else "1.0.0"

    doc_raw = data.get("documentation_file")
    doc_str = str(doc_raw) if doc_raw is not None else None

    ds_doc_raw = data.get("dataset_documentation_file")
    ds_doc_str = str(ds_doc_raw) if ds_doc_raw is not None else None

    return cls(
        name=name_str,
        description=desc_str,
        reference=ref_str,
        type=type_str,
        category=cat_str,
        main_score=main_score_str,
        revision=rev_str,
        dataset=ds_meta,
        scores=scores,
        eval_splits=eval_splits,
        eval_langs=eval_langs,
        domains=domains,
        task_subtypes=task_subtypes,
        documentation_file=doc_str,
        dataset_documentation_file=ds_doc_str,
    )

  def to_dict(self) -> Dict[str, Any]:
    return dataclasses.asdict(self)


@dataclasses.dataclass
class EvaluationRecord:
  """Represents an individual task/dataset evaluation record extracted from a JSONL file."""

  model_entry: ModelEntry
  task_name: str
  dataset_name: str
  sub_task_name: str = "default"
  is_svq: bool = False
  main_score_name: str = ""
  main_score_value: float = 0.0
  all_scores: Dict[str, float] = dataclasses.field(default_factory=dict)
  condition_scores: Dict[str, float] = dataclasses.field(default_factory=dict)
  is_noise_slice: bool = False
  noise_condition: Optional[str] = None
  raw_record_name: Optional[str] = None
  tags: List[str] = dataclasses.field(default_factory=list)
  prompt: Optional[str] = None
  task_metadata: Optional[TaskMetadata] = None
  eval_splits: List[str] = dataclasses.field(default_factory=list)
  eval_langs: List[str] = dataclasses.field(default_factory=list)
  source_file: str = ""

  def get_score(
      self, metric_name: str, default: Optional[float] = None
  ) -> Optional[float]:
    """Fetches a score by metric name from all_scores."""
    return self.all_scores.get(metric_name, default)


@dataclasses.dataclass
class TaskScoreSummary:
  """Aggregated metrics for a model on a specific MSEB task."""

  task_name: str
  primary_metric: str
  all_mean: Optional[float] = None
  svq_mean: Optional[float] = None
  normalized_svq_score: Optional[float] = None
  dataset_scores: Dict[str, float] = dataclasses.field(default_factory=dict)
  dataset_count: int = 0
  svq_dataset_count: int = 0


@dataclasses.dataclass
class ModelScores:
  """Aggregated evaluation metrics for a model across all tasks."""

  model_entry: ModelEntry
  task_summaries: Dict[str, TaskScoreSummary] = dataclasses.field(
      default_factory=dict
  )
  overall_svq_score: Optional[float] = None
  evaluated_task_count: int = 0
  evaluated_dataset_count: int = 0
  records_count: int = 0

  def get_task_score(
      self, task_key: str, default: Optional[float] = None
  ) -> Optional[float]:
    """Returns the all_mean score for a task, or default."""
    summary = self.task_summaries.get(task_key)
    return (
        summary.all_mean
        if summary and summary.all_mean is not None
        else default
    )

  def get_task_svq_score(
      self, task_key: str, default: Optional[float] = None
  ) -> Optional[float]:
    """Returns the svq_mean score for a task, or default."""
    summary = self.task_summaries.get(task_key)
    return (
        summary.svq_mean
        if summary and summary.svq_mean is not None
        else default
    )


@dataclasses.dataclass
class LeaderboardData:
  """Leaderboard container containing summary and task DataFrames."""

  overall_df: Any  # pd.DataFrame
  task_dfs: Dict[str, Any]  # task_key -> pd.DataFrame
  models: List[ModelEntry]
  model_scores: List[ModelScores] = dataclasses.field(default_factory=list)
