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

"""Tab builders for the MSEB Gradio Leaderboard."""

from src.ui.tabs.about_tab import create_about_tab
from src.ui.tabs.overall_tab import create_overall_tab
from src.ui.tabs.task_tab import create_task_tab

__all__ = [
    "create_overall_tab",
    "create_task_tab",
    "create_about_tab",
]
