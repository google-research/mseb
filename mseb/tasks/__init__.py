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

"""MSEB Tasks."""

from mseb.task import get_name_to_task
from mseb.task import get_task_by_name

from .classifications.birdset.birdset import *
from .classifications.intent.speech_massive import *
from .classifications.sound.fsd50k import *
from .classifications.speaker_gender.speech_massive import *
from .classifications.speaker_gender.svq import *
from .clusterings.birdset import *
from .clusterings.fsd50k import *
from .clusterings.svq import *
from .reasonings.span_cross_lang.svq import *
from .reasonings.span_in_lang.svq import *
# TODO(tombagby): Temporary remove because of dep changes, for now unregister,
# switch them to lazy import and then re-enable.
# from .rerankings.query.svq import *
# from .retrievals.document_cross_lang.svq import *
# from .retrievals.document_in_lang.svq import *
# from .retrievals.passage_cross_lang.svq import *
# from .retrievals.passage_in_lang.svq import *
from .segmentations.salient_term.svq import *
# from .transcriptions.speech.svq import *
