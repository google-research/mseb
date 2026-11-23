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

"""Versioned and locale-aware text normalizers for MSEB."""

import sys
from typing import Protocol, runtime_checkable

from mseb import registry


@runtime_checkable
class Normalizer(Protocol):
  """Runtime-checkable structural protocol for versioned text normalizers."""

  @property
  def locale(self) -> str:
    ...

  def __call__(self, text: str) -> str:
    ...


REGISTRY: registry.Registry[Normalizer] = registry.Registry(
    'normalizers', Normalizer
)
__getattr__ = REGISTRY.bind(sys.modules[__name__])

REGISTRY.register('Whisper.V1', 'mseb.normalizers.whisper.WhisperNormalizer')
# TBD: Register 'Whisper_AA.V1' once Artificial Analysis rules are finalized.
