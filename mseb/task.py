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

"""MSEB Task Class."""

import abc
import logging
from typing import Iterable, Type

from absl import flags
import apache_beam as beam
from mseb import runner as runner_lib
from mseb import types


TASK_CACHE_BASEPATH = flags.DEFINE_string(
    "task_cache_basepath",
    None,
    "Base path for the task cache.",
)

TRANSCRIPT_KEY = flags.DEFINE_string(
    "transcript_key",
    "text",
    "Key to use for the transcript in the task data.",
)


logger = logging.getLogger(__name__)


class MSEBTask(abc.ABC):
  """Abstract base class for MSEB tasks."""

  metadata: types.TaskMetadata = None

  def setup(
      self, runner: runner_lib.EncoderRunner | None = None
  ):
    """Called once to set up the task.

    This method is called before the `run` method and can be used to
    initialize any resources needed for the task, such as loading data,
    creating directories, or setting up connections to external services.

    Args:
      runner: An optional EncoderRunner class that can be used to get embeddings
        for setting up the task.
    """
    pass

  @abc.abstractmethod
  def compute_scores(
      self, embeddings: types.MultiModalEmbeddingCache
  ) -> dict[str, list[types.Score]]:
    """Evaluate the task."""

  @abc.abstractmethod
  def sounds(self) -> Iterable[types.Sound]:
    """Iterate all of the sounds in the corpus for this task."""

  def sounds_beam(self) -> beam.PCollection[types.Sound]:
    """Beam transform to iterate all of the sounds in the corpus for this task."""
    return beam.Create(list(self.sounds()))


def get_name_to_task() -> dict[str, Type[MSEBTask]]:
  name_to_task: dict[str, type[MSEBTask]] = {}
  tasks = list(MSEBTask.__subclasses__())
  while tasks:
    cls = tasks.pop()
    if cls.metadata:
      name_to_task[cls.metadata.name] = cls
    else:
      tasks.extend(cls.__subclasses__())
  return name_to_task


def get_task_by_name(task_name: str) -> Type[MSEBTask]:
  """Returns the MSEBTask class for the given task name."""
  task_cls = get_name_to_task().get(task_name)
  if task_cls is None:
    raise ValueError(
        f"Task {task_name} not found. Registered tasks: {get_name_to_task()}"
    )
  return task_cls
