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

"""Tracks for MSEB."""

from mseb import track as track_lib

from . import asr
from . import generative
from . import retrieval

registry = {
    'asr': asr.ASR,
    'generative': generative.GENERATIVE,
    'retrieval': retrieval.RETRIEVAL,
}


def list_tracks() -> list[str]:
  """Returns all registered track names."""
  return sorted(registry)


def get_track_by_name(track_name: str) -> track_lib.Track:
  """Returns the track with the given name."""
  if track_name not in registry:
    raise ValueError(
        f'Unknown track {track_name!r}. Available: {list(registry.keys())}'
    )
  return registry[track_name]
