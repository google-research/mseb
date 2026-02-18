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

"""MSEB Tasks."""

# pylint: disable=g-import-not-at-top

from mseb.task import get_name_to_task
from mseb.task import get_task_by_name

# The following imports are wrapped in try-except to avoid breaking environments
# where some dependencies are missing (e.g., sharded tests on GitHub).
# This ensures that tasks are registered when their dependencies are present,
# which is needed for leaderboard generation and running tasks.

try:
  from .classifications.birdset.birdset import *
except ImportError:
  pass

try:
  from .classifications.intent.speech_massive import *
except ImportError:
  pass

try:
  from .classifications.sound.fsd50k import *
except ImportError:
  pass

try:
  from .classifications.speaker_gender.speech_massive import *
except ImportError:
  pass

try:
  from .classifications.speaker_gender.svq import *
except ImportError:
  pass

try:
  from .clusterings.birdset import *
except ImportError:
  pass

try:
  from .clusterings.fsd50k import *
except ImportError:
  pass

try:
  from .clusterings.svq import *
except ImportError:
  pass

try:
  from .reasonings.span_cross_lang.svq import *
except ImportError:
  pass

try:
  from .reasonings.span_in_lang.svq import *
except ImportError:
  pass

try:
  from .rerankings.query.svq import *
except ImportError:
  pass

try:
  from .retrievals.document_cross_lang.svq import *
except ImportError:
  pass

try:
  from .retrievals.document_in_lang.svq import *
except ImportError:
  pass

try:
  from .retrievals.passage_cross_lang.svq import *
except ImportError:
  pass

try:
  from .retrievals.passage_in_lang.svq import *
except ImportError:
  pass

try:
  from .segmentations.salient_term.svq import *
except ImportError:
  pass

try:
  from .transcriptions.speech.svq import *
except ImportError:
  pass
