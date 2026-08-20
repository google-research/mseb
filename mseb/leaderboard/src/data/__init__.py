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

"""Data models and ingestion parser for MSEB Leaderboard."""

from __future__ import annotations

from src.data.models import DatasetMetadata
from src.data.models import EvaluationRecord
from src.data.models import LeaderboardData
from src.data.models import ModelEntry
from src.data.models import ModelScores
from src.data.models import ScoreSpec
from src.data.models import TaskMetadata
from src.data.models import TaskScoreSummary
from src.data.parser import discover_models
from src.data.parser import load_all_evaluation_data
from src.data.parser import parse_jsonl_file

__all__ = [
    "DatasetMetadata",
    "EvaluationRecord",
    "LeaderboardData",
    "ModelEntry",
    "ModelScores",
    "ScoreSpec",
    "TaskMetadata",
    "TaskScoreSummary",
    "discover_models",
    "load_all_evaluation_data",
    "parse_jsonl_file",
]
