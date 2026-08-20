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

"""Reusable UI components for the MSEB Gradio Leaderboard."""

from __future__ import annotations

import gradio as gr
import pandas as pd
from src.config import TaskSpec


def create_header() -> gr.HTML:
  """Renders the top banner, title, description, and benchmark statistics."""
  header_html = """
    <div class="header-container">
        <h1 class="header-title">MSEB Leaderboard</h1>
        <p class="header-subtitle">
            <strong>Massive Sound Embedding Benchmark (MSEB)</strong> —
            A comprehensive multilingual benchmark for evaluating speech, audio,
            and multimodal models. See <a href="https://github.com/google-research/mseb" target="_blank">git</a>/<a href="https://neurips.cc/virtual/2025/loc/san-diego/poster/121597" target="_blank">poster</a>/<a href="https://proceedings.neurips.cc/paper_files/paper/2025/file/2c878aec1c052a835511c262033ad348-Paper-Datasets_and_Benchmarks_Track.pdf" target="_blank">paper</a> for more details.
        </p>
    </div>
    """
  return gr.HTML(header_html)


def create_search_bar(
    placeholder: str = "🔍 Search models by name (e.g., 'gemini', 'flash', 'asr=truth')...",
    label: str = "Search Models",
) -> gr.Textbox:
  """Creates a text input component for dynamic model name filtering."""
  return gr.Textbox(
      show_label=False,
      placeholder=placeholder,
      label=label,
      elem_classes=["search-input"],
      interactive=True,
  )


def create_column_selector(
    choices: list[str],
    default_selected: list[str],
    label: str = "Select Columns to Display:",
    show_label: bool = True,
) -> gr.CheckboxGroup:
  """Creates an interactive checkbox group for toggling visible columns."""
  return gr.CheckboxGroup(
      choices=choices,
      value=default_selected,
      label=label,
      show_label=show_label,
      elem_classes=["column-selector-group"],
      interactive=True,
  )


def create_leaderboard_table(
    df: pd.DataFrame,
) -> gr.DataFrame:
  """Creates a Gradio DataFrame configured for tabular leaderboard presentation.

  Supports clickable Markdown model links and formatted numeric columns.

  Args:
    df: DataFrame containing leaderboard data.

  Returns:
    gr.DataFrame configured for display.
  """
  return gr.DataFrame(
      value=df,
      interactive=False,
      datatype="markdown",
      elem_classes=["leaderboard-table"],
      max_height=768,
      wrap=True,
  )


def create_task_header(
    task_spec: TaskSpec,
    dataset_count: int = 0,
) -> gr.HTML:
  """Renders task description and primary metric info callout banner."""
  symbol = task_spec.metric_direction.symbol
  direction_desc = (
      "Higher is better"
      if task_spec.metric_direction.name == "HIGHER_IS_BETTER"
      else "Lower is better"
  )
  unit_desc = (
      "Percentage [0–100%]"
      if task_spec.default_unit == "percentage"
      else "Raw distance metric"
  )

  html = f"""
    <div class="task-info-banner">
        <div class="task-info-title">{task_spec.display_name} ({symbol} {direction_desc})</div>
        <div class="task-info-desc">
            {task_spec.description} &bull;
            <strong>Primary Metric:</strong> <code>{task_spec.primary_metric}</code> ({unit_desc}) &bull;
            <strong>Evaluated Datasets:</strong> {dataset_count}
        </div>
    </div>
    """
  return gr.HTML(html)


def create_inactive_task_notice(task_spec: TaskSpec) -> gr.HTML:
  """Renders a notice for tasks with no evaluation records in the active results catalog."""
  html = f"""
    <div class="inactive-task-banner">
        ⚠️ <strong>Note:</strong> No evaluated models are currently available inn
        <code>results/google/</code> for <strong>{task_spec.display_name}</sstrong>.
        Showing the canonical column schema.
    </div>
    """
  return gr.HTML(html)
