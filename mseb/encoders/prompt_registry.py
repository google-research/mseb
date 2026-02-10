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

"""Registry for Encoders.

This module defines the EncoderMetadata dataclass, which holds the information
needed to instantiate and load Encoder models. It also includes definitions for
specific encoder configurations.
"""

import dataclasses
import inspect
import sys
from typing import Any, Callable
from mseb.encoders import prompt as prompt_lib
from mseb.tasks.classifications.intent import speech_massive as speech_massive_intent
from mseb.tasks.classifications.sound import fsd50k
from mseb.tasks.classifications.speaker_gender import speech_massive as speech_massive_speaker_gender


@dataclasses.dataclass(frozen=True)
class PromptMetadata:
  """Metadata for a Prompt instantiated with specific parameters."""

  name: str  # The name of the prompt.
  prompt: Callable[..., prompt_lib.Prompt]
  # Lazy evaluation of the prompt parameters so we can use flags.
  params: Callable[[], dict[str, Any]]  # Additional encoder parameters.

  def load(self) -> prompt_lib.Prompt:
    """Loads the encoder."""
    prompt = self.prompt(**self.params())  # pytype: disable=not-instantiable
    return prompt


reasoning = PromptMetadata(
    name="reasoning",
    prompt=prompt_lib.ReasoningPrompt,
    params=lambda: {},
)

intent_classification = PromptMetadata(
    name="intent_classification",
    prompt=prompt_lib.ClassificationPrompt,
    params=lambda: {
        "class_labels": list(
            speech_massive_intent.SpeechMassiveIntentClassification().class_labels()
        )
    },
)

speaker_gender_classification = PromptMetadata(
    name="speaker_gender_classification",
    prompt=prompt_lib.SpeakerClassificationPrompt,
    params=lambda: {
        "class_labels": list(
            speech_massive_speaker_gender.SpeechMassiveSpeakerGenderClassification().class_labels()
        )
    },
)

sound_classification = PromptMetadata(
    name="sound_classification",
    prompt=prompt_lib.SoundClassificationPrompt,
    params=lambda: {
        "class_labels": fsd50k.FSD50KClassification().class_labels()
    },
)

retrieval = PromptMetadata(
    name="retrieval",
    prompt=prompt_lib.RetrievalPrompt,
    params=lambda: {},
)

segmentation = PromptMetadata(
    name="segmentation",
    prompt=prompt_lib.SegmentationPrompt,
    params=lambda: {},
)

transcription = PromptMetadata(
    name="transcription",
    prompt=prompt_lib.TranscriptionPrompt,
    params=lambda: {},
)

reranking = PromptMetadata(
    name="reranking",
    prompt=prompt_lib.RerankingPrompt,
    params=lambda: {},
)

_REGISTRY: dict[str, PromptMetadata] = {}


def _register_prompts():
  """Finds and registers all PromptMetadata instances in this module."""
  current_module = sys.modules[__name__]
  for name, obj in inspect.getmembers(current_module):
    if isinstance(obj, PromptMetadata):
      if obj.name in _REGISTRY:
        raise ValueError(f"Duplicate encoder name {obj.name} found in {name}")
      _REGISTRY[obj.name] = obj


def get_prompt_metadata(name: str) -> PromptMetadata:
  """Retrieves an PromptMetadata instance by its name.

  Args:
    name: The unique name of the prompt metadata instance.

  Returns:
    The PromptMetadata instance.

  Raises:
    ValueError: if the name is not found in the registry.
  """
  if not _REGISTRY:
    _register_prompts()
  if name not in _REGISTRY:
    raise ValueError(f"PromptMetadata with name '{name}' not found.")
  return _REGISTRY[name]


# Automatically register encoders on import
_register_prompts()
