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

"""MSEB Metric Aggregation, SVQ Macro-Averaging, and DataFrame Matrix Engine.

This module aggregates raw EvaluationRecord data across tasks and datasets,
computes task averages, normalizes metric directions for overall SVQ
macro-averaging, generates wide task DataFrames and the Overall summary
leaderboard DataFrame, and provides interactive search and column filtering.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from src.config import get_source_file_url
from src.config import MSEB_TASKS
from src.config import normalize_metric_for_svq
from src.config import ORDERED_TASK_KEYS
from src.config import resolve_results_dir
from src.config import TASK_PRIMARY_METRICS
from src.data.models import EvaluationRecord
from src.data.models import LeaderboardData
from src.data.models import ModelEntry
from src.data.models import ModelScores
from src.data.models import TaskScoreSummary
from src.data.parser import discover_models
from src.data.parser import load_all_evaluation_data

logger = logging.getLogger(__name__)


def format_score_for_task_table(val: Optional[float], task_key: str) -> float:
  """Formats a raw score for display in a task-specific DataFrame."""
  del task_key
  if val is None:
    return np.nan
  try:
    f_val = float(val)
  except (ValueError, TypeError):
    return np.nan
  if math.isnan(f_val):
    return np.nan

  if 0.0 <= f_val <= 1.0:
    return round(f_val * 100.0, 2)
  else:
    return round(f_val, 2)


def _get_base_model_name(entry: ModelEntry) -> str:
  """Returns the base model name without sub_config suffix."""
  name = entry.display_name or entry.top_model
  if entry.sub_config and name.endswith(f" ({entry.sub_config})"):
    return name[: -len(f" ({entry.sub_config})")].strip()
  return name


def build_overall_dataframe(
    records: Optional[List[EvaluationRecord]] = None,
    models: Optional[List[ModelEntry]] = None,
) -> pd.DataFrame:
  """Builds the master summary DataFrame for the Overall Leaderboard tab.

  Args:
      records: List of parsed EvaluationRecord objects.
      models: Optional list of discovered ModelEntry objects. If None, extracted
        from records.

  Returns:
      pd.DataFrame sorted descending by 'Average ⬆️' with task score columns,
      metadata columns, and NaN for unevaluated tasks.
  """
  task_cols = [t.capitalize() for t in ORDERED_TASK_KEYS]

  if not records and not models:
    return pd.DataFrame(columns=["Model", "Tags", "Average ⬆️"] + task_cols)

  # 1. Index models
  models_dict: Dict[str, ModelEntry] = {}
  if models:
    for m in models:
      models_dict[m.entry_id] = m
  for rec in records or []:
    if rec.model_entry.entry_id not in models_dict:
      models_dict[rec.model_entry.entry_id] = rec.model_entry

  if not models_dict:
    return pd.DataFrame(columns=["Model", "Tags", "Average ⬆️"] + task_cols)

  # 2. Accumulate raw and SVQ scores per model, task, and dataset
  raw_dataset_scores: Dict[str, Dict[str, Dict[str, List[float]]]] = {
      mid: {t: {} for t in ORDERED_TASK_KEYS} for mid in models_dict
  }
  svq_dataset_scores: Dict[str, Dict[str, Dict[str, List[float]]]] = {
      mid: {t: {} for t in ORDERED_TASK_KEYS} for mid in models_dict
  }

  task_source_files: Dict[str, Dict[str, str]] = {
      mid: {} for mid in models_dict
  }

  for rec in records or []:
    mid = rec.model_entry.entry_id
    tname = rec.task_name
    if tname not in raw_dataset_scores[mid]:
      continue

    val = rec.main_score_value
    if val is None or math.isnan(val):
      continue

    if rec.source_file:
      task_source_files[mid][tname] = rec.source_file

    ds = rec.dataset_name
    raw_dataset_scores[mid][tname].setdefault(ds, []).append(val)

    # Compute normalized SVQ score
    norm_s = normalize_metric_for_svq(val, rec.main_score_name)
    if not math.isnan(norm_s):
      svq_dataset_scores[mid][tname].setdefault(ds, []).append(norm_s)

  # 3. Build row per model
  rows: List[Dict[str, Any]] = []
  for mid, entry in models_dict.items():
    row: Dict[str, Any] = {
        "Model": _get_base_model_name(entry),
        "Tags": entry.sub_config or "-",
        "entry_id": entry.entry_id,
        "top_model": entry.top_model,
        "sub_config": entry.sub_config or "",
        "url": entry.url,
    }

    for t in ORDERED_TASK_KEYS:
      col_name = t.capitalize()
      t_raw_ds = raw_dataset_scores[mid][t]

      src_f = task_source_files[mid].get(t)
      if src_f:
        row[f"{col_name}__url"] = get_source_file_url(src_f)

      if t_raw_ds:
        # Average per dataset first (handles multiple noise runs/subtasks)
        ds_means = [
            sum(scores) / len(scores) for scores in t_raw_ds.values() if scores
        ]
        if ds_means:
          task_raw_mean = sum(ds_means) / len(ds_means)
          if 0.0 <= task_raw_mean <= 1.0:
            disp_val = task_raw_mean * 100.0
          else:
            disp_val = task_raw_mean
          row[col_name] = round(disp_val, 2)
        else:
          row[col_name] = np.nan
      else:
        row[col_name] = np.nan

    # SVQ Macro-average across all tasks, treating missing tasks as 0.0
    task_norm_scores: List[float] = []
    for t in ORDERED_TASK_KEYS:
      col_name = t.capitalize()
      disp_val = row.get(col_name)
      if pd.notna(disp_val):
        try:
          f_val = float(disp_val)
          if t == "transcription":
            norm_val = 100.0 - f_val
          else:
            norm_val = f_val
          task_norm_scores.append(norm_val)
        except (ValueError, TypeError):
          task_norm_scores.append(0.0)
      else:
        task_norm_scores.append(0.0)

    if task_norm_scores:
      row["Average ⬆️"] = round(float(np.mean(task_norm_scores)), 2)
    else:
      row["Average ⬆️"] = 0.0

    rows.append(row)

  df = pd.DataFrame(rows)
  if not df.empty:
    meta_cols = ["entry_id", "top_model", "sub_config", "url"]
    task_cols = [
        t.capitalize()
        for t in ORDERED_TASK_KEYS
        if t.capitalize() in df.columns
    ]
    url_cols = [c for c in df.columns if c.endswith("__url")]
    other_cols = [
        c
        for c in df.columns
        if c
        not in ["Model", "Tags", "Average ⬆️"]
        + meta_cols
        + task_cols
        + url_cols
    ]
    ordered_cols = (
        ["Model", "Tags", "Average ⬆️"]
        + task_cols
        + [m for m in meta_cols if m in df.columns]
        + url_cols
        + other_cols
    )
    df = df[[c for c in ordered_cols if c in df.columns]]
    if "Average ⬆️" in df.columns:
      df = df.sort_values(by=["Average ⬆️"], ascending=False).reset_index(
          drop=True
      )
  return df


def build_task_dataframe(
    records: Optional[List[EvaluationRecord]],
    task_name: str,
    models: Optional[List[ModelEntry]] = None,
) -> pd.DataFrame:
  """Builds a fine-grained DataFrame for an individual MSEB task tab.

  Args:
      records: List of parsed EvaluationRecord objects.
      task_name: Normalized task key (e.g. 'reasoning', 'classification').
      models: Optional list of discovered ModelEntry objects.

  Returns:
      pd.DataFrame with columns ['Model', 'Tags', <Dataset_1>, ..., <Dataset_N>,
      'Task Average']
      sorted descending by 'Task Average'.
  """
  del models
  t_clean = task_name.lower().strip()
  task_records = [r for r in records or [] if r.task_name == t_clean]

  if not task_records:
    return pd.DataFrame(columns=["Model", "Tags", "Task Average"])

  # Discover all evaluated datasets for this task
  all_datasets = sorted(list(set(r.dataset_name for r in task_records)))

  # Index models evaluated in this task
  model_recs: Dict[str, Tuple[ModelEntry, Dict[str, List[float]]]] = {}
  dataset_source_files: Dict[str, Dict[str, str]] = {}
  for r in task_records:
    mid = r.model_entry.entry_id
    if mid not in model_recs:
      model_recs[mid] = (r.model_entry, {})
      dataset_source_files[mid] = {}
    if r.source_file:
      dataset_source_files[mid][r.dataset_name] = r.source_file

    val = r.main_score_value
    if val is not None:
      try:
        f_val = float(val)
        if not math.isnan(f_val):
          model_recs[mid][1].setdefault(r.dataset_name, []).append(f_val)
      except (ValueError, TypeError):
        pass

  rows: List[Dict[str, Any]] = []
  for mid, (entry, ds_dict) in model_recs.items():
    row: Dict[str, Any] = {
        "Model": _get_base_model_name(entry),
        "Tags": entry.sub_config or "-",
        "entry_id": entry.entry_id,
        "top_model": entry.top_model,
        "sub_config": entry.sub_config or "",
        "url": entry.url,
    }
    dataset_vals: List[float] = []

    for ds in all_datasets:
      src_f = dataset_source_files.get(mid, {}).get(ds)
      if src_f:
        row[f"{ds}__url"] = get_source_file_url(src_f)

      scores = ds_dict.get(ds, [])
      if scores:
        mean_s = sum(scores) / len(scores)
        disp_s = format_score_for_task_table(mean_s, t_clean)
        row[ds] = disp_s
        dataset_vals.append(disp_s)
      else:
        row[ds] = np.nan
        dataset_vals.append(0.0)

    if dataset_vals:
      row["Task Average"] = round(float(np.mean(dataset_vals)), 2)
    else:
      row["Task Average"] = 0.0

    rows.append(row)

  df = pd.DataFrame(rows)
  if not df.empty:
    meta_cols = ["entry_id", "top_model", "sub_config", "url"]
    ds_cols = [d for d in all_datasets if d in df.columns]
    url_cols = [c for c in df.columns if c.endswith("__url")]
    ordered_cols = (
        ["Model", "Tags", "Task Average"]
        + ds_cols
        + [m for m in meta_cols if m in df.columns]
        + url_cols
    )
    df = df[[c for c in ordered_cols if c in df.columns]]
    if "Task Average" in df.columns:
      spec = MSEB_TASKS.get(t_clean)
      is_lib = (
          spec.metric_direction.name == "LOWER_IS_BETTER" if spec else False
      )
      df = df.sort_values(
          by=["Task Average"], ascending=is_lib, na_position="last"
      ).reset_index(drop=True)
  return df


def _build_typed_model_scores(
    records: Optional[List[EvaluationRecord]],
    models: List[ModelEntry],
    overall_df: pd.DataFrame,
) -> List[ModelScores]:
  """Constructs typed ModelScores and TaskScoreSummary objects."""
  recs = records or []
  # Map entry_id to overall_svq_score from overall_df
  overall_scores_map: Dict[str, float] = {}
  if (
      not overall_df.empty
      and "entry_id" in overall_df.columns
      and "Average ⬆️" in overall_df.columns
  ):
    for _, r in overall_df.iterrows():
      if pd.notna(r["Average ⬆️"]):
        overall_scores_map[r["entry_id"]] = float(r["Average ⬆️"])

  # Group records by (mid, task)
  rec_group: Dict[str, Dict[str, List[EvaluationRecord]]] = {
      m.entry_id: {t: [] for t in ORDERED_TASK_KEYS} for m in models
  }
  for r in recs:
    mid = r.model_entry.entry_id
    t = r.task_name.lower().strip()
    if mid in rec_group and t in rec_group[mid]:
      rec_group[mid][t].append(r)

  result: List[ModelScores] = []
  for m in models:
    mid = m.entry_id
    task_summaries: Dict[str, TaskScoreSummary] = {}
    total_eval_datasets = 0
    total_recs = 0

    for t in ORDERED_TASK_KEYS:
      t_recs = rec_group[mid][t]
      total_recs += len(t_recs)
      primary_metric = TASK_PRIMARY_METRICS.get(t, "Score")

      if not t_recs:
        task_summaries[t] = TaskScoreSummary(
            task_name=t,
            primary_metric=primary_metric,
        )
        continue

      # Group scores by dataset
      ds_raw_scores: Dict[str, List[float]] = {}
      ds_svq_scores: Dict[str, List[float]] = {}
      for r in t_recs:
        if r.main_score_value is not None:
          try:
            f_val = float(r.main_score_value)
            if not math.isnan(f_val):
              ds_raw_scores.setdefault(r.dataset_name, []).append(f_val)
              norm_s = normalize_metric_for_svq(f_val, r.main_score_name)
              if not math.isnan(norm_s):
                ds_svq_scores.setdefault(r.dataset_name, []).append(norm_s)
          except (ValueError, TypeError):
            pass

      ds_means: Dict[str, float] = {
          ds: float(np.mean(vals)) for ds, vals in ds_raw_scores.items() if vals
      }
      svq_means: Dict[str, float] = {
          ds: float(np.mean(vals)) for ds, vals in ds_svq_scores.items() if vals
      }

      all_mean = float(np.mean(list(ds_means.values()))) if ds_means else None
      svq_mean = float(np.mean(list(svq_means.values()))) if svq_means else None

      total_eval_datasets += len(ds_means)

      task_summaries[t] = TaskScoreSummary(
          task_name=t,
          primary_metric=primary_metric,
          all_mean=round(all_mean, 4) if all_mean is not None else None,
          svq_mean=round(svq_mean, 4) if svq_mean is not None else None,
          normalized_svq_score=round(svq_mean, 4)
          if svq_mean is not None
          else None,
          dataset_scores=ds_means,
          dataset_count=len(ds_means),
          svq_dataset_count=len(svq_means),
      )

    eval_task_cnt = sum(
        1 for ts in task_summaries.values() if ts.dataset_count > 0
    )

    result.append(
        ModelScores(
            model_entry=m,
            task_summaries=task_summaries,
            overall_svq_score=overall_scores_map.get(mid),
            evaluated_task_count=eval_task_cnt,
            evaluated_dataset_count=total_eval_datasets,
            records_count=total_recs,
        )
    )

  return result


def build_leaderboard_tables(
    records: Optional[List[EvaluationRecord]] = None,
    results_dir: Optional[str] = None,
    models: Optional[List[ModelEntry]] = None,
) -> LeaderboardData:
  """High-level coordinator that ingests results and constructs all leaderboard tables.

  Args:
      records: Optional list of parsed EvaluationRecord objects.
      results_dir: Optional directory path to search for results if records is
        None.
      models: Optional list of ModelEntry objects.

  Returns:
      LeaderboardData dataclass containing overall_df, task_dfs, models, and
      model_scores.
  """
  if records is None:
    resolved_dir = resolve_results_dir(results_dir)
    models = discover_models(resolved_dir)
    records = load_all_evaluation_data(resolved_dir)

  if models is None:
    models_map = {r.model_entry.entry_id: r.model_entry for r in records or []}
    models = list(models_map.values())

  overall_df = build_overall_dataframe(records=records, models=models)

  task_dfs: Dict[str, pd.DataFrame] = {}
  for t in ORDERED_TASK_KEYS:
    task_dfs[t] = build_task_dataframe(
        records=records, task_name=t, models=models
    )

  model_scores = _build_typed_model_scores(
      records=records, models=models, overall_df=overall_df
  )

  return LeaderboardData(
      overall_df=overall_df,
      task_dfs=task_dfs,
      models=models,
      model_scores=model_scores,
  )


def filter_leaderboard_df(
    df: pd.DataFrame,
    search_query: str = "",
    selected_columns: Optional[List[str]] = None,
    cant_deselect: Optional[List[str]] = None,
) -> pd.DataFrame:
  """Filters a leaderboard DataFrame by model name search query and visible column selection.

  Args:
      df: Input DataFrame.
      search_query: Search string to filter Model names.
      selected_columns: Subset of columns to display.
      cant_deselect: Mandatory columns that cannot be hidden (e.g., 'Model').

  Returns:
      Filtered pd.DataFrame with reset index.
  """
  if df.empty:
    return df.copy()

  filtered = df.copy()

  # 1. Search Query Filtering
  query = (search_query or "").strip()
  if query:
    mask = pd.Series(False, index=filtered.index)
    if "Model" in filtered.columns:
      mask |= (
          filtered["Model"]
          .astype(str)
          .str.contains(query, case=False, regex=False)
      )
    if "Tags" in filtered.columns:
      mask |= (
          filtered["Tags"]
          .astype(str)
          .str.contains(query, case=False, regex=False)
      )
    if "top_model" in filtered.columns:
      mask |= (
          filtered["top_model"]
          .astype(str)
          .str.contains(query, case=False, regex=False)
      )
    if "sub_config" in filtered.columns:
      mask |= (
          filtered["sub_config"]
          .astype(str)
          .str.contains(query, case=False, regex=False)
      )
    if "entry_id" in filtered.columns:
      mask |= (
          filtered["entry_id"]
          .astype(str)
          .str.contains(query, case=False, regex=False)
      )
    filtered = filtered[mask]

  # 2. Column Selection & Retention
  if selected_columns is not None:
    raw_keep = list(selected_columns)
    if cant_deselect:
      for c in reversed(cant_deselect):
        if c in df.columns and c not in raw_keep:
          raw_keep.insert(0, c)
    # Deduplicate while preserving order
    keep = []
    for c in raw_keep:
      if c not in keep and c in filtered.columns:
        keep.append(c)
    if keep:
      filtered = filtered[keep]

  return filtered.reset_index(drop=True)


def compute_dynamic_overall_average(
    df: pd.DataFrame,
    selected_task_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
  """Dynamically recalculates the 'Average ⬆️' column.

  Computes the macro-average of the currently selected task columns for each
  model row, properly taking metric orientation into account (e.g. inverting
  lower-is-better WER for Transcription), and re-sorting rows by the dynamic
  Average in descending order.

  Args:
      df: Overall leaderboard DataFrame containing model rows and task columns.
      selected_task_columns: List of visible task column names (e.g.
        ['Classification', 'Reasoning']).

  Returns:
      pd.DataFrame with recomputed 'Average ⬆️' sorted descending.
  """
  if df.empty:
    return df.copy()

  df_copy = df.copy()
  task_keys = [t.capitalize() for t in ORDERED_TASK_KEYS]

  if selected_task_columns is None:
    active_tasks = [c for c in task_keys if c in df_copy.columns]
  elif len(selected_task_columns) == 0:
    active_tasks = []
  else:
    active_tasks = [
        c
        for c in selected_task_columns
        if c in df_copy.columns and c in task_keys
    ]

  averages: List[float] = []
  for _, row in df_copy.iterrows():
    row_scores: List[float] = []
    for col in active_tasks:
      val = row[col]
      if pd.notna(val):
        try:
          f_val = float(val)
          t_key = col.lower().strip()
          if t_key == "transcription":
            norm_val = 100.0 - f_val
          else:
            norm_val = f_val
          row_scores.append(norm_val)
        except (ValueError, TypeError):
          row_scores.append(0.0)
      else:
        row_scores.append(0.0)

    if active_tasks:
      averages.append(round(float(np.mean(row_scores)), 2))
    else:
      averages.append(np.nan)

  df_copy["Average ⬆️"] = averages

  # Sort descending by Average ⬆️ with NaN at bottom
  if "Average ⬆️" in df_copy.columns:
    df_copy = df_copy.sort_values(
        by=["Average ⬆️"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

  return df_copy


def compute_dynamic_task_average(
    df: pd.DataFrame,
    selected_dataset_columns: Optional[List[str]] = None,
    task_key: str = "",
) -> pd.DataFrame:
  """Dynamically recalculates the 'Task Average' column.

  Computes the average of the currently selected dataset score columns for
  each model row in a task tab, and re-sorts rows by the dynamic Task Average.

  Args:
      df: Task DataFrame containing model rows and dataset columns.
      selected_dataset_columns: List of visible dataset column names.
      task_key: Normalized task key (e.g. 'classification', 'transcription').

  Returns:
      pd.DataFrame with recomputed 'Task Average' sorted appropriately.
  """
  if df.empty:
    return df.copy()

  df_copy = df.copy()
  non_ds_cols = {
      "Model",
      "Tags",
      "Task Average",
      "entry_id",
      "top_model",
      "sub_config",
      "url",
      "Rank",
  }
  all_dataset_cols = [
      c
      for c in df_copy.columns
      if c not in non_ds_cols and not c.endswith("__url")
  ]

  if selected_dataset_columns is None:
    active_datasets = all_dataset_cols
  elif len(selected_dataset_columns) == 0:
    active_datasets = []
  else:
    active_datasets = [
        c
        for c in selected_dataset_columns
        if c in df_copy.columns and c in all_dataset_cols
    ]

  averages: List[float] = []
  for _, row in df_copy.iterrows():
    row_scores: List[float] = []
    for col in active_datasets:
      val = row[col]
      if pd.notna(val):
        try:
          f_val = float(val)
          row_scores.append(f_val)
        except (ValueError, TypeError):
          row_scores.append(0.0)
      else:
        row_scores.append(0.0)

    if active_datasets:
      averages.append(round(float(np.mean(row_scores)), 2))
    else:
      averages.append(np.nan)

  df_copy["Task Average"] = averages

  # Sort appropriately by Task Average
  if "Task Average" in df_copy.columns:
    t_clean = task_key.lower().strip()
    spec = MSEB_TASKS.get(t_clean)
    is_lib = spec.metric_direction.name == "LOWER_IS_BETTER" if spec else False
    df_copy = df_copy.sort_values(
        by=["Task Average"],
        ascending=is_lib,
        na_position="last",
    ).reset_index(drop=True)

  return df_copy
