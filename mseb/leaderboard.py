# Copyright 2025 The MSEB Authors.
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

"""Leaderboard generation.

This module contains functions for running task/encoder pairs and recording
and reporting the results.
"""

import dataclasses
import json
from mseb import runner as runner_lib
from mseb import task as task_lib
from mseb import types


@dataclasses.dataclass
class LeaderboardResult:
  """Results of a single task evaluation."""

  name: str
  sub_task_name: str  # Update as sub-tasks removed/merged into task metadata.
  task_metadata: types.TaskMetadata
  scores: list[types.Score]

  def to_json(self) -> str:
    """Convert metrics to JSON string."""
    return json.dumps(dataclasses.asdict(self), indent=2)

  @staticmethod
  def from_json(json_str: str) -> 'LeaderboardResult':
    """Convert metrics from JSON string."""
    data = json.loads(json_str)
    scores = [types.Score(**x) for x in data['scores']]
    task_metadata_dict = data['task_metadata']
    dataset_dict = task_metadata_dict['dataset']
    task_metadata = types.TaskMetadata(
        dataset=types.Dataset(**dataset_dict),
        scores=[types.Score(**x) for x in task_metadata_dict['scores']],
        **{
            k: v
            for k, v in task_metadata_dict.items()
            if k not in ['dataset', 'scores']
        }
    )
    return LeaderboardResult(
        name=data['name'],
        sub_task_name=data['sub_task_name'],
        task_metadata=task_metadata,
        scores=scores,
    )


def run_benchmark(
    encoder_name: str,  # Maybe this can come from Encoder metadata?
    runner: runner_lib.EncoderRunner,
    task: task_lib.MSEBTask,
) -> list[LeaderboardResult]:
  """Run a task evaluation."""
  embeddings = runner.run(task.sounds())
  scores = task.compute_scores(embeddings)
  return [
      LeaderboardResult(
          name=encoder_name,
          sub_task_name=sub_task_name,
          task_metadata=task.metadata,
          scores=scores,
      )
      for sub_task_name, scores in scores.items()
  ]
