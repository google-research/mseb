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

import collections
import dataclasses
import json
from typing import Any, Callable, Dict, Iterable, Iterator, List, TextIO
from mseb import runner as runner_lib
from mseb import task as task_lib
from mseb import types
import numpy as np


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


def get_encoding_scores(
    embeddings: types.MultiModalEmbeddingCache,
) -> list[types.Score]:
  """Get scores for encoding statistics."""
  stats = [x.encoding_stats for x in embeddings.values() if x.encoding_stats]
  return [
      types.Score(
          metric='mean_encoding_size_bytes',
          description='Mean encoding size in bytes.',
          value=np.mean([x.embedding_size_bytes for x in stats]),
          min=0,
          max=np.inf,
      ),
      types.Score(
          metric='flops',
          description='Flops used to encode.',
          value=stats[0].flops,  # TODO(tombagby): Correct aggregation?
          min=0,
          max=np.inf,
      ),
  ]


def run_benchmark(
    encoder_name: str,  # Maybe this can come from Encoder metadata?
    runner: runner_lib.EncoderRunner,
    task: task_lib.MSEBTask,
) -> list[LeaderboardResult]:
  """Run a task evaluation."""
  embeddings = runner.run(task.sounds())
  scores = task.compute_scores(embeddings)
  encoding_scores = get_encoding_scores(embeddings)
  return [
      LeaderboardResult(
          name=encoder_name,
          sub_task_name=sub_task_name,
          task_metadata=task.metadata,
          scores=scores + encoding_scores,
      )
      for sub_task_name, scores in scores.items()
  ]


@dataclasses.dataclass
class FlattenedLeaderboardResult:
  """Flattened leaderboard result for analysis."""

  name: str
  task_name: str
  task_type: str
  main_score_metric: str
  main_score_value: float
  metric: str
  metric_value: float
  metric_description: str
  metric_min: int | float
  metric_max: int | float
  metric_std: float | None


def flatten_leaderboard_results(
    results: Iterator[str],
) -> List[FlattenedLeaderboardResult]:
  """Parses and flattens leaderboard results for analysis.

  Args:
    results: An iterator of JSON strings, each representing a LeaderboardResult.

  Returns:
    A list of FlattenedLeaderboardResult objects.
  """
  flattened_results = []
  for result_json in results:
    result = LeaderboardResult.from_json(result_json)
    task_metadata = result.task_metadata
    main_score_metric = task_metadata.main_score
    main_score_value = None
    for score in result.scores:
      if score.metric == main_score_metric:
        main_score_value = score.value
        break

    for score in result.scores:
      flattened_results.append(
          FlattenedLeaderboardResult(
              name=result.name,
              task_name=f'{task_metadata.name}/{result.sub_task_name}',
              task_type=task_metadata.type,
              main_score_metric=main_score_metric,
              main_score_value=main_score_value,
              metric=score.metric,
              metric_value=score.value,
              metric_description=score.description,
              metric_min=score.min,
              metric_max=score.max,
              metric_std=score.std,
          )
      )
  return flattened_results


def partition_leaderboard_results(
    results: Iterator[str],
    naming_fn: Callable[[LeaderboardResult], str],
) -> Dict[str, List[LeaderboardResult]]:
  """Partitions leaderboard results based on a naming function.

  Args:
    results: An iterator of JSON strings, each representing a LeaderboardResult.
    naming_fn: A function that takes a LeaderboardResult and returns a string
      key.

  Returns:
    A dictionary mapping keys from naming_fn to lists of LeaderboardResult
    objects.
  """
  partitions = collections.defaultdict(list)
  for result_json in results:
    result = LeaderboardResult.from_json(result_json)
    key = naming_fn(result)
    partitions[key].append(result)
  return partitions


def write_dataclasses_to_jsonl(
    results: Iterable[Any],
    f: TextIO,
) -> None:
  """Writes an iterable of dataclass objects to a file as JSONL.

  Args:
    results: An iterable of dataclass objects.
    f: An open file object to write to.
  """
  for result in results:
    json.dump(dataclasses.asdict(result), f)
    f.write('\n')
