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
from typing import Iterable, Type

from mseb import types


class MSEBTask(abc.ABC):
  """Abstract base class for MSEB tasks."""

  metadata: types.TaskMetadata = None

  def setup(self):
    """Called once to set up the task.

    This method is called before the `run` method and can be used to
    initialize any resources needed for the task, such as loading data,
    creating directories, or setting up connections to external services.
    """
    pass

  @abc.abstractmethod
  def compute_scores(
      self, embeddings: types.SoundEmbeddingCache
  ) -> dict[str, list[types.Score]]:
    """Evaluate the task."""

  @abc.abstractmethod
  def sounds(self) -> Iterable[types.Sound]:
    """Iterate all of the sounds in the corpus for this task."""


def get_task_list() -> list[type[MSEBTask]]:
  return list(MSEBTask.__subclasses__())


def get_name_to_task() -> dict[str, Type[MSEBTask]]:
  name_to_task: dict[str, type[MSEBTask]] = {}
  tasks = get_task_list()
  while tasks:
    cls = tasks.pop()
    if cls.metadata:
      name_to_task[cls.metadata.name] = cls
    else:
      tasks.extend(cls.__subclasses__())
  return name_to_task
