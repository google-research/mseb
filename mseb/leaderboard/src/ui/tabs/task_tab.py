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

"""Task-specific Leaderboard Tab for MSEB Gradio Application."""

from __future__ import annotations

from typing import List, Optional, Tuple

import gradio as gr
import pandas as pd
from src.config import MetricDirection
from src.config import MSEB_TASKS
from src.config import TaskSpec
from src.data.aggregator import compute_dynamic_task_average
from src.data.aggregator import filter_leaderboard_df
from src.data.models import LeaderboardData
from src.ui.components import create_column_selector
from src.ui.components import create_inactive_task_notice
from src.ui.components import create_leaderboard_table
from src.ui.components import create_search_bar
from src.ui.components import create_task_header
from src.utils.formatting import format_dataframe_for_display


def create_task_tab(task_key: str, data: LeaderboardData) -> gr.TabItem:
  """Constructs a dedicated tab for an individual MSEB task.

  Args:
      task_key: Key of the MSEB task (e.g. 'classification', 'transcription').
      data: LeaderboardData containing task DataFrames.

  Returns:
      gr.TabItem component containing the rendered task UI.
  """
  task_key_clean = task_key.lower().strip()
  task_spec = MSEB_TASKS.get(
      task_key_clean,
      TaskSpec(
          key=task_key_clean,
          display_name=task_key_clean.capitalize(),
          col_name=f"{task_key_clean.capitalize()} ⬆️",
          primary_metric="Score",
          metric_direction=MetricDirection.HIGHER_IS_BETTER,
          category="speech",
          description=f"Evaluations for {task_key_clean}.",
          is_active=False,
      ),
  )

  tab_label = task_spec.display_name
  tab_id = f"tab-{task_key_clean}"

  # Retrieve raw task DataFrame
  raw_task_df = data.task_dfs.get(
      task_key_clean,
      pd.DataFrame(columns=["Model", "Task Average"]),
  )

  # Collect dataset columns
  non_dataset_cols = {
      "Model",
      "Tags",
      "Task Average",
      "entry_id",
      "top_model",
      "sub_config",
      "url",
      "Rank",
  }
  dataset_cols = [
      c
      for c in raw_task_df.columns
      if c not in non_dataset_cols and not c.endswith("__url")
  ]

  with gr.TabItem(tab_label, id=tab_id):
    # 1. Task Header / Description Banner
    create_task_header(task_spec, dataset_count=len(dataset_cols))

    # Inactive task notice if empty
    if raw_task_df.empty or len(raw_task_df) == 0:
      create_inactive_task_notice(task_spec)

    # 2. Controls Panel: Search & Dataset Column Selector
    with gr.Row(elem_classes=["filter-panel"]):
      with gr.Column(scale=2):
        search_box = create_search_bar(
            placeholder=f"🔍 Search models in {task_spec.display_name}...",
            label=f"Search {task_spec.display_name}",
        )
      with gr.Column(scale=3):
        with gr.Row(elem_classes=["column-selector-header"]):
          gr.Markdown(
              f"**Select Datasets ({len(dataset_cols)} total):**",
              elem_classes=["selector-label"],
          )
          toggle_btn = gr.Button(
              "Deselect all",
              size="sm",
              variant="secondary",
              elem_classes=["toggle-all-btn"],
          )
        col_selector = create_column_selector(
            choices=dataset_cols,
            default_selected=dataset_cols,
            label=f"Select Datasets ({len(dataset_cols)} total):",
            show_label=False,
        )

    # 3. Initial Display Table
    initial_display_df = format_dataframe_for_display(
        raw_task_df,
        is_overall=False,
        task_key=task_key_clean,
        add_rank=True,
    )
    table = create_leaderboard_table(initial_display_df)

    # 4. Interactive update handler
    def update_task_view(
        search_query: str,
        selected_datasets: Optional[List[str]],
    ) -> pd.DataFrame:
      if raw_task_df.empty:
        return format_dataframe_for_display(
            raw_task_df,
            is_overall=False,
            task_key=task_key_clean,
            add_rank=True,
        )

      # 1. Filter by search query on base DataFrame
      filtered = filter_leaderboard_df(
          raw_task_df,
          search_query=search_query,
          selected_columns=None,
      )

      # 2. Dynamically recompute Task Average based on selected_datasets
      filtered = compute_dynamic_task_average(
          filtered,
          selected_dataset_columns=selected_datasets,
          task_key=task_key_clean,
      )

      # 3. Select visible columns (maintaining canonical dataset order)
      if selected_datasets is None:
        selected_datasets = dataset_cols

      selected_set = set(selected_datasets)
      ordered_selected_datasets = [
          c
          for c in dataset_cols
          if c in selected_set and c in raw_task_df.columns
      ]
      desired_cols = [
          "Model",
          "Tags",
          "Task Average",
      ] + ordered_selected_datasets
      for meta_col in ("entry_id", "top_model", "sub_config", "url"):
        if meta_col in raw_task_df.columns and meta_col not in desired_cols:
          desired_cols.append(meta_col)
      for ds_col in ordered_selected_datasets:
        url_col = f"{ds_col}__url"
        if url_col in raw_task_df.columns and url_col not in desired_cols:
          desired_cols.append(url_col)

      filtered = filtered[[c for c in desired_cols if c in filtered.columns]]

      # 4. Format for display with dynamic rank badges
      return format_dataframe_for_display(
          filtered,
          is_overall=False,
          task_key=task_key_clean,
          add_rank=True,
      )

    # 5. Wire event listeners
    search_box.change(
        fn=update_task_view,
        inputs=[search_box, col_selector],
        outputs=[table],
    )

    def on_task_col_change(
        search_query: str,
        selected_datasets: Optional[List[str]],
    ) -> Tuple[pd.DataFrame, str]:
      selected = selected_datasets if selected_datasets is not None else []
      updated_df = update_task_view(search_query, selected)
      btn_label = (
          "Select all" if not selected or len(selected) == 0 else "Deselect all"
      )
      return updated_df, btn_label

    col_selector.change(
        fn=on_task_col_change,
        inputs=[search_box, col_selector],
        outputs=[table, toggle_btn],
    )

    def on_task_toggle_click(
        search_query: str,
        current_selected: Optional[List[str]],
    ) -> Tuple[List[str], str, pd.DataFrame]:
      if current_selected and len(current_selected) > 0:
        new_selected = []
        new_label = "Select all"
      else:
        new_selected = dataset_cols
        new_label = "Deselect all"
      updated_df = update_task_view(search_query, new_selected)
      return new_selected, new_label, updated_df

    toggle_btn.click(
        fn=on_task_toggle_click,
        inputs=[search_box, col_selector],
        outputs=[col_selector, toggle_btn, table],
    )
