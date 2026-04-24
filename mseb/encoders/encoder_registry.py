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
from typing import Any, Callable

from absl import flags
from mseb import encoder as encoder_lib


PROMPT_NAME = flags.DEFINE_string(
    "prompt_name",
    "reasoning",
    "Name of the prompt to use for LLM-based encoders.",
)

DOCUMENT_PROMPT_NAME = flags.DEFINE_string(
    "document_prompt_name",
    "",
    "Name of the prompt to use for the document encoder in retrieval tasks.",
)


# Flags for specific encoders that were previously here are removed.
# They should be moved to the specific encoder files or registration files.

_RETRIEVAL_TOP_K = flags.DEFINE_integer(
    "retrieval_top_k",
    100,
    "The number of top k retrieved items for retrieval encoders.",
)

Encoder = encoder_lib.MultiModalEncoder


@dataclasses.dataclass(frozen=True)
class EncoderMetadata:
  """Metadata for an Encoder instantiated with specific parameters."""

  name: str  # The name of the encoder for creation and leaderboard entry.
  encoder: Callable[..., encoder_lib.MultiModalEncoder]
  # Lazy evaluation of the encoder parameters so we can use flags.
  params: Callable[[], dict[str, Any]]  # Additional encoder parameters.
  url: str | None = None  # URL for information about the encoder model.
  base_model: str | None = None  # Base model name for grouping.
  tags: tuple[str, ...] = ()

  def load(self) -> Encoder:
    """Loads the encoder."""
    encoder = self.encoder(**self.params())  # pytype: disable=not-instantiable
    return encoder


_REGISTRY: dict[str, EncoderMetadata] = {}


def register_encoder(metadata: EncoderMetadata):
  """Registers an EncoderMetadata instance.

  Args:
    metadata: The EncoderMetadata instance to register.

  Raises:
    ValueError: if an encoder with the same name is already registered.
  """
  if metadata.name in _REGISTRY:
    raise ValueError(f"Duplicate encoder name {metadata.name} found")
  _REGISTRY[metadata.name] = metadata


def get_encoder_metadata(name: str) -> EncoderMetadata:
  """Retrieves an EncoderMetadata instance by its name.

  Args:
    name: The unique name of the encoder metadata instance.

  Returns:
    The EncoderMetadata instance.

  Raises:
    ValueError: if the name is not found in the registry.
  """
  if name not in _REGISTRY:
    raise ValueError(
        f"EncoderMetadata with name '{name}' not found. "
        "Ensure the module containing the encoder is imported."
    )
  return _REGISTRY[name]
