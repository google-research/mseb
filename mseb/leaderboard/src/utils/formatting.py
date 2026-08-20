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

"""Presentation and formatting utilities for the MSEB Gradio Leaderboard."""

from __future__ import annotations

import math
from typing import Optional, Union

import pandas as pd
from src.config import ModelType
from src.config import ORDERED_TASK_KEYS


def format_score(
    score: Optional[Union[float, int, str]],
    precision: int = 2,
    default: str = "-",
) -> str:
  """Formats a floating-point score to a fixed-decimal string, returning default for NaN/None."""
  if score is None:
    return default
  try:
    val = float(score)
  except (ValueError, TypeError):
    return default
  if math.isnan(val) or math.isinf(val):
    return default
  return f"{val:.{precision}f}"


def format_percentage(
    score: Optional[Union[float, int, str]],
    precision: int = 1,
    default: str = "-",
) -> str:
  """Formats a score as a percentage string (e.g. 85.2%)."""
  if score is None:
    return default
  try:
    val = float(score)
  except (ValueError, TypeError):
    return default
  if math.isnan(val) or math.isinf(val):
    return default
  scaled = val * 100.0 if 0.0 <= val <= 1.0 else val
  return f"{scaled:.{precision}f}%"


def format_params(
    params_b: Optional[Union[float, int, str]],
    default: str = "-",
) -> str:
  """Formats parameter count in Billions (e.g., 2.5B, 70B, 500M, -)."""
  if params_b is None:
    return default
  try:
    val = float(params_b)
  except (ValueError, TypeError):
    return str(params_b) if params_b else default
  if math.isnan(val) or val <= 0:
    return default
  if val >= 1.0:
    formatted = f"{val:.1f}B"
    if formatted.endswith(".0B"):
      formatted = f"{int(val)}B"
    return formatted
  return f"{val * 1000:.0f}M"


def format_markdown_link(text: str, url: Optional[str] = None) -> str:
  """Wraps text in a clickable Markdown link if URL is a valid http(s) string."""
  if (
      url
      and isinstance(url, str)
      and (url.startswith("http://") or url.startswith("https://"))
  ):
    return f"[{text}]({url})"
  return str(text) if text is not None else ""


def format_rank_badge(rank: int, add_medal: bool = False) -> str:
  """Returns rank string with medal emoji for top 3 positions."""
  if not add_medal:
    return str(rank)
  if rank == 1:
    return "🥇 1"
  elif rank == 2:
    return "🥈 2"
  elif rank == 3:
    return "🥉 3"
  return str(rank)


def format_model_type_badge(model_type_str: str) -> str:
  """Returns styled symbol + label for model type (e.g., 🟢 pretrained)."""
  mt = ModelType.from_str(model_type_str)
  if mt == ModelType.UNKNOWN:
    return "?"
  return mt.to_str()


def format_dataframe_for_display(
    df: pd.DataFrame,
    is_overall: bool = True,
    task_key: Optional[str] = None,
    add_rank: bool = True,
) -> pd.DataFrame:
  """Converts a raw numeric DataFrame into a UI-ready presentation DataFrame.

  Transformations:
  - Adds 'Rank' column with medal badges (🥇, 🥈, 🥉).
  - Formats Model column into clickable Markdown links using 'url' column.
  - Formats numeric score columns to 2 decimal places.
  - Replaces NaNs with '-' placeholders.
  - Preserves sort order.

  Args:
    df: Raw numeric DataFrame.
    is_overall: Whether this is the overall summary leaderboard.
    task_key: Optional task key for task-specific tables.
    add_rank: Whether to prepend a Rank column.

  Returns:
    Formatted DataFrame ready for display.
  """
  del task_key
  if df.empty:
    return df.copy()

  display_df = df.copy()

  # 1. Add Rank column at position 0 if requested
  if add_rank and "Rank" not in display_df.columns:
    ranks = [format_rank_badge(i + 1) for i in range(len(display_df))]
    display_df.insert(0, "Rank", ranks)

  # 2. Convert Model name to Markdown link if url present
  if "Model" in display_df.columns and "url" in display_df.columns:
    display_df["Model"] = display_df.apply(
        lambda row: format_markdown_link(row["Model"], row["url"]),
        axis=1,
    )

  # 3. Format numeric score columns (linking to source jsonl if __url metadata
  # exists)
  non_numeric_cols = {
      "Rank",
      "Model",
      "Tags",
      "entry_id",
      "top_model",
      "sub_config",
      "url",
      "model_type",
      "license",
      "architecture",
      "precision",
      "tags",
      "model_type_symbol",
      "still_on_hub",
      "T",
      "Type",
      "Hub License",
      "Architecture",
      "Precision",
      "Available on Hub",
      "#Params (B)",
  }
  for col in list(display_df.columns):
    if col not in non_numeric_cols and not col.endswith("__url"):
      url_col = f"{col}__url"
      has_url = url_col in display_df.columns

      def _fmt_cell(row_val, url_val):
        formatted = format_score(row_val, precision=2, default="-")
        if formatted != "-" and url_val and pd.notna(url_val):
          return f"[{formatted}]({url_val})"
        return formatted

      if pd.api.types.is_numeric_dtype(display_df[col]) or has_url:
        if has_url:
          display_df[col] = [
              _fmt_cell(v, u)
              for v, u in zip(display_df[col], display_df[url_col])
          ]
        else:
          display_df[col] = display_df[col].apply(
              lambda v: format_score(v, precision=2, default="-")
          )

  # 4. Drop internal metadata columns and all __url columns so they don't
  # clutter the UI
  internal_meta_cols = ["entry_id", "top_model", "sub_config", "url"]
  cols_to_drop = [
      c
      for c in display_df.columns
      if c in internal_meta_cols or c.endswith("__url")
  ]
  if cols_to_drop:
    display_df = display_df.drop(columns=cols_to_drop)

  # 5. Enforce standard leading column order: Rank, Model, Tags, Average / Task
  # Average, followed by sorted task/dataset columns
  cols = list(display_df.columns)
  if is_overall and "Average ⬆️" in cols:
    lead = [c for c in ["Rank", "Model", "Tags", "Average ⬆️"] if c in cols]
    task_canonical = [t.capitalize() for t in ORDERED_TASK_KEYS]
    task_cols = [c for c in task_canonical if c in cols and c not in lead]
    other_cols = [c for c in cols if c not in lead and c not in task_cols]
    display_df = display_df[lead + task_cols + other_cols]
  elif not is_overall and "Task Average" in cols:
    lead = [c for c in ["Rank", "Model", "Tags", "Task Average"] if c in cols]
    ds_cols = sorted([c for c in cols if c not in lead])
    display_df = display_df[lead + ds_cols]

  return display_df


def clean_nan_values(df: pd.DataFrame, placeholder: str = "-") -> pd.DataFrame:
  """Replaces all NaN/None values in DataFrame with a placeholder string."""
  return df.fillna(placeholder)


def export_dataframe_to_csv(df: pd.DataFrame, file_path: str) -> None:
  """Exports DataFrame to CSV with UTF-8 encoding."""
  df.to_csv(file_path, index=False, encoding="utf-8")


def export_dataframe_to_json(df: pd.DataFrame, file_path: str) -> None:
  """Exports DataFrame to JSON format."""
  df.to_json(file_path, orient="records", indent=2, force_ascii=False)
