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

"""MSEB Multi-Tab Gradio Leaderboard Application.

This module provides the main application factory `create_app()` and standalone
entrypoint for launching the interactive multi-tab MSEB Gradio Leaderboard.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import gradio as gr
from src.config import ORDERED_TASK_KEYS
from src.config import resolve_results_dir
from src.data.aggregator import build_leaderboard_tables
from src.ui.components import create_header
from src.ui.styles import CUSTOM_CSS
from src.ui.tabs.about_tab import create_about_tab
from src.ui.tabs.overall_tab import create_overall_tab
from src.ui.tabs.task_tab import create_task_tab

# Disable Gradio and Hugging Face telemetry and analytics requests
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mseb_leaderboard")


def create_app(results_dir: Optional[str] = None) -> gr.Blocks:
  """Constructs and returns the full MSEB Gradio Leaderboard Blocks application.

  Args:
      results_dir: Optional path to directory containing JSONL evaluation
        results. If None, resolves to canonical
        `third_party/py/mseb/results/google/`.

  Returns:
      gr.Blocks instance containing the complete multi-tab application.
  """
  logger.info("Initializing MSEB Leaderboard dataset and models...")
  resolved_results_dir = resolve_results_dir(results_dir)
  logger.info("Loading evaluation records from: %s", resolved_results_dir)

  data = build_leaderboard_tables(results_dir=resolved_results_dir)
  logger.info(
      "Loaded %d model configurations across %d tasks.",
      len(data.models),
      len(data.task_dfs),
  )

  theme = gr.themes.Soft(
      primary_hue="blue",
      secondary_hue="slate",
      font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
      font_mono=[
          gr.themes.GoogleFont("JetBrains Mono"),
          "ui-monospace",
          "monospace",
      ],
  )

  head_js = """
    <script>
    // Allow native browser right-click context menu by stopping propagation of Gradio contextmenu overrides
    window.addEventListener('contextmenu', function(e) {
        if (e.target && (e.target.closest('.leaderboard-table') || e.target.closest('table') || e.target.closest('td') || e.target.closest('th') || e.target.closest('a'))) {
            e.stopImmediatePropagation();
        }
    }, true);
    </script>
    """

  with gr.Blocks(
      title="MSEB Leaderboard",
      css=CUSTOM_CSS,
      head=head_js,
      theme=theme,
      analytics_enabled=False,
  ) as app_demo:
    # Header Banner & Benchmark Intro
    create_header()

    # Multi-Tab Navigation Container
    with gr.Tabs(elem_classes=["tab-buttons"]):
      # 1. Overall Summary Tab
      create_overall_tab(data)

      # 2. Dedicated Task Tabs
      for task_key in ORDERED_TASK_KEYS:
        create_task_tab(task_key, data)

      # 3. About & Documentation Tab
      create_about_tab()

  return app_demo


# Expose top-level `demo` instance for Gradio auto-reload CLI (`gradio app.py`)
demo = create_app()


def main() -> None:
  """Standalone CLI entrypoint to boot the MSEB Gradio Leaderboard server."""
  port = int(os.environ.get("PORT", "7860"))
  server_name = os.environ.get("SERVER_NAME", "0.0.0.0")

  logger.info("Launching server on %s:%d...", server_name, port)
  demo.launch(
      server_name=server_name,
      server_port=port,
      show_error=True,
      share=False,
      ssr_mode=False,  # SSR breaks column selection buttons.
  )


if __name__ == "__main__":
  main()
