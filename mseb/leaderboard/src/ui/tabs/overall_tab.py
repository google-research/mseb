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

"""Overall Leaderboard Tab for the MSEB Gradio Application."""

from __future__ import annotations

from typing import List, Optional, Tuple

import gradio as gr
import pandas as pd
from src.config import ORDERED_TASK_KEYS
from src.data.aggregator import compute_dynamic_overall_average
from src.data.aggregator import filter_leaderboard_df
from src.data.models import LeaderboardData
from src.ui.components import create_column_selector
from src.ui.components import create_leaderboard_table
from src.ui.components import create_search_bar
from src.utils.formatting import format_dataframe_for_display


def create_overall_tab(data: LeaderboardData) -> gr.TabItem:
  """Constructs the Overall summary leaderboard tab.

  Features:
  - Summary ranking based on macro-averaged SVQ scores across evaluated tasks.
  - Real-time search filter across model names and configurations.
  - Dynamic column selector allowing toggling of task columns.
  - Clickable model repository links and medal rank badges (🥇, 🥈, 🥉).

  Args:
    data: LeaderboardData containing overall and per-task data.

  Returns:
    gr.TabItem for the overall tab.
  """
  with gr.TabItem("Overall", id="tab-overall"):
    gr.Markdown(
        """
            ### Overall MSEB Leaderboard
            Models are ranked by their **Average ⬆️** score — the unweighted macro-average
            of normalized SVQ task scores (scaled to 0–100 where higher is always better).
            Use the search box to filter models and toggle task columns as needed.
            """,
        elem_classes=["markdown-text"],
    )

    with gr.Row(elem_classes=["filter-panel"]):
      with gr.Column(scale=2):
        search_box = create_search_bar(
            placeholder=(
                "🔍 Search model name (e.g. 'gemini', 'whisper',"
                " 'asr=truth')..."
            ),
            label="Search Models",
        )
      with gr.Column(scale=3):
        with gr.Row(elem_classes=["column-selector-header"]):
          gr.Markdown(
              "**Visible Task Columns:**", elem_classes=["selector-label"]
          )
          toggle_btn = gr.Button(
              "Deselect all",
              size="sm",
              variant="secondary",
              elem_classes=["toggle-all-btn"],
          )
        task_column_choices = [t.capitalize() for t in ORDERED_TASK_KEYS]
        col_selector = create_column_selector(
            choices=task_column_choices,
            default_selected=task_column_choices,
            label="Visible Task Columns:",
            show_label=False,
        )

    # Initial display DataFrame
    initial_display_df = format_dataframe_for_display(
        data.overall_df,
        is_overall=True,
        add_rank=True,
    )
    table = create_leaderboard_table(initial_display_df)

    # Interactive event handler for search and column toggle
    def update_overall_view(
        search_query: str,
        selected_tasks: Optional[List[str]],
    ) -> pd.DataFrame:
      if selected_tasks is None:
        selected_tasks = task_column_choices

      # 1. Filter by search query on base DataFrame
      filtered = filter_leaderboard_df(
          data.overall_df,
          search_query=search_query,
          selected_columns=None,
      )

      # 2. Dynamically recompute Average ⬆️ based on selected_tasks
      filtered = compute_dynamic_overall_average(
          filtered,
          selected_task_columns=selected_tasks,
      )

      # 3. Select visible columns (maintaining alphabetical / canonical task
      # order)
      selected_set = set(selected_tasks)
      ordered_selected_tasks = [
          c
          for c in task_column_choices
          if c in selected_set and c in filtered.columns
      ]
      desired_cols = ["Model", "Tags", "Average ⬆️"] + ordered_selected_tasks
      for meta_col in ("entry_id", "top_model", "sub_config", "url"):
        if meta_col in filtered.columns and meta_col not in desired_cols:
          desired_cols.append(meta_col)
      for task_col in ordered_selected_tasks:
        url_col = f"{task_col}__url"
        if url_col in filtered.columns and url_col not in desired_cols:
          desired_cols.append(url_col)

      filtered = filtered[[c for c in desired_cols if c in filtered.columns]]

      # 4. Format for display with dynamic rank badges
      return format_dataframe_for_display(
          filtered,
          is_overall=True,
          add_rank=True,
      )

    # Wire up dynamic event listeners
    search_box.change(
        fn=update_overall_view,
        inputs=[search_box, col_selector],
        outputs=[table],
    )

    def on_col_selector_change(
        search_query: str,
        selected_tasks: Optional[List[str]],
    ) -> Tuple[pd.DataFrame, str]:
      selected = selected_tasks if selected_tasks is not None else []
      updated_df = update_overall_view(search_query, selected)
      btn_label = (
          "Select all" if not selected or len(selected) == 0 else "Deselect all"
      )
      return updated_df, btn_label

    col_selector.change(
        fn=on_col_selector_change,
        inputs=[search_box, col_selector],
        outputs=[table, toggle_btn],
    )

    def on_toggle_btn_click(
        search_query: str,
        current_selected: Optional[List[str]],
    ) -> Tuple[List[str], str, pd.DataFrame]:
      if current_selected and len(current_selected) > 0:
        new_selected = []
        new_label = "Select all"
      else:
        new_selected = task_column_choices
        new_label = "Deselect all"
      updated_df = update_overall_view(search_query, new_selected)
      return new_selected, new_label, updated_df

    toggle_btn.click(
        fn=on_toggle_btn_click,
        inputs=[search_box, col_selector],
        outputs=[col_selector, toggle_btn, table],
    )
