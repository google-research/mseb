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

"""ASR track."""

import types

from mseb import track
from mseb.tracks import constants

_DEBUG_ASR = tuple(
    sorted(
        set([
            'SVQEnUsSpeechTranscriptionDebug',
            'SVQFiFiSpeechTranscriptionDebug',
        ])
    )
)

_COMPACT_ASR = tuple(
    sorted(
        set([
            f'SVQ{locale}SpeechTranscriptionCompact'
            for locale in constants.SVQ_LOCALES
        ])
    )
)

_FULL_ASR = tuple(
    sorted(
        set([
            f'SVQ{locale}SpeechTranscription'
            for locale in constants.SVQ_LOCALES
        ])
    )
)

ASR = track.Track(
    name='asr',
    description='ASR tasks',
    tasks_by_size=types.MappingProxyType({
        track.Size.DEBUG: _DEBUG_ASR,
        track.Size.COMPACT: _COMPACT_ASR,
        track.Size.FULL: _FULL_ASR,
    }),
)
