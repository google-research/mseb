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

from mseb import encoder
from mseb import types

logger = logging.getLogger(__name__)


class SoundEmbeddings(abc.ABC):

  def __init__(self, sound_encoder: encoder.SoundEncoder):
    self._encoder = sound_encoder

  @abc.abstractmethod
  def __getitem__(self, sound_id: str) -> types.SoundEmbedding:
    """Get the embeddings for a given sound id."""


class SoundEmbeddingsInMemory(SoundEmbeddings):
  """SoundEmbeddings simply stored in an in-memory dict."""

  embeddings: dict[str, types.SoundEmbedding] = {}

  def __getitem__(self, sound_id: str) -> types.SoundEmbedding:
    return self.embeddings[sound_id]


class MSEBTask(abc.ABC):
  """Abstract base class for MSEB tasks."""

  metadata: types.TaskMetadata = None

  @abc.abstractmethod
  def setup(self):
    """Called once to set up the task."""

  @abc.abstractmethod
  def run(self, embeddings: SoundEmbeddings) -> dict[str, list[types.Score]]:
    """Evaluate the task."""

  @abc.abstractmethod
  def encode(self, embeddings: SoundEmbeddings) -> SoundEmbeddings:
    """Embed task audio."""

  @abc.abstractmethod
  def evaluate(
      self, embeddings: SoundEmbeddings
  ) -> dict[str, list[types.Score]]:
    """Load data for the task."""


def get_task_list() -> list[type[MSEBTask]]:
  return list(MSEBTask.__subclasses__())


def get_name_to_task() -> dict[str, type[MSEBTask]]:
  tasks = get_task_list()
  return {cls.metadata.name: cls for cls in tasks if cls.metadata}
