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

"""Track primitives for structured MSEB evaluation.

Tracks group tasks by modeling paradigm (e.g. retrieval vs. generative), etc.
Sizes control the tradeoff between coverage and cost.

This module defines the primitives only. The shipped tracks, and the registry
that names them, live in `mseb.tracks`.

All track and size memberships are defined as explicit task name lists.

Usage:
  from mseb import track as track_lib
  from mseb import tracks

  # List all tracks and their sizes.
  for name, mseb_track in tracks.registry.items():
    print(name, mseb_track.describe())

  # Get task names for a track+size combination.
  task_names = tracks.get_track_by_name('retrieval').list_tasks(
      track_lib.Size.COMPACT
  )
"""

from collections.abc import Collection, Mapping, Sequence
import dataclasses
import enum
import logging

from mseb import task as task_lib
import mseb.tasks  # pylint: disable=unused-import


class Size(enum.Enum):
  DEBUG = enum.auto()
  COMPACT = enum.auto()
  FULL = enum.auto()


@dataclasses.dataclass(frozen=True)
class Track:
  """An evaluation track with tasks organized by size.

  Attributes:
    name: Short identifier (e.g. 'retrieval').
    description: Human-readable description.
    tasks_by_size: Mapping from size name to the sequence of task names at that
      size.
  """

  name: str
  description: str
  tasks_by_size: Mapping[Size, tuple[str, ...]]

  def sizes(self) -> list[Size]:
    """Returns the available sizes for this track."""
    return list(self.tasks_by_size)

  def list_tasks(
      self,
      size: Size = Size.FULL,
      select: Collection[str] | None = None,
  ) -> list[str]:
    """Returns sorted task names for the given size.

    Args:
      size: The track size.
      select: Optional task names to restrict the result to. Names outside the
        track are ignored, so this narrows a track rather than extending it. If
        None, every registered MSEB task is admitted.

    Returns:
      Sorted list of task names.

    Raises:
      ValueError: If size is not recognized.
    """
    if size not in self.tasks_by_size:
      raise ValueError(
          f'Unknown size {size!r} for track {self.name!r}.'
          f' Available: {list(self.tasks_by_size)}'
      )
    track_tasks = self.tasks_by_size[size]
    if select is None:
      select = tuple(task_lib.get_name_to_task().keys())
    select_set = frozenset(select)
    tasks = set(track_tasks) & select_set
    not_selected = set(track_tasks) - tasks
    logging.warning(
        'Track %s size %s has %d tasks, but only %d were selected, not'
        ' selected: %s',
        self.name,
        size,
        len(track_tasks),
        len(tasks),
        not_selected,
    )
    return sorted(tasks)

  def describe(self) -> str:
    """Returns a human-readable summary of the track and its sizes."""
    lines = [f'{self.name}: {self.description}']
    for size, tasks in self.tasks_by_size.items():
      lines.append(f'  {size.name.lower()}: {len(tasks)} task names')
    return '\n'.join(lines)


def add_sized_variants(task_names: Sequence[str]) -> list[str]:
  """Adds the registered sized variants of each task name.

  This lets a caller name tasks at full size and still select them under
  `--size compact`, by widening the selection to cover the sized variants.

  Args:
    task_names: Task names, typically as passed on the command line.

  Returns:
    `task_names`, plus every registered sized variant of each of them.
  """
  known_task_names = frozenset(task_lib.get_name_to_task())
  extended_task_names = list(task_names)
  for task in task_names:
    for size in set(Size) - {Size.FULL}:
      task_with_size = f'{task}{size.name.lower().capitalize()}'
      if task_with_size in known_task_names:
        extended_task_names.append(task_with_size)
  return extended_task_names
