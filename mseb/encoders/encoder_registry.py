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

"""Registry for SoundEncoders.

This module defines the EncoderMetadata dataclass, which holds the information
needed to instantiate and load SoundEncoder models. It also includes
definitions for specific encoder configurations.
"""

import dataclasses
from typing import Any, Type

from mseb import encoder as encoder_lib
from mseb.encoders import raw_encoder
from mseb.encoders import whisper_encoder


@dataclasses.dataclass(frozen=True)
class EncoderMetadata:
  """Metadata for a SoundEncoder instantiated with specific parameters."""

  name: str  # The name of the encoder for creation and leaderboard entry.
  encoder: Type[encoder_lib.SoundEncoder]  # The encoder class.
  params: dict[str, Any]  # Additional encoder parameters.

  def load(self) -> encoder_lib.SoundEncoder:
    """Loads the encoder."""
    return self.encoder(**self.params)  # pytype: disable=not-instantiable


raw_encoder_25ms_10ms = EncoderMetadata(
    name="raw_spectrogram_25ms_10ms_mean",
    encoder=raw_encoder.RawEncoder,
    params={
        "frame_length": 25,
        "frame_step": 10,
        "transform_fn": raw_encoder.spectrogram_transform,
        "pooling": "mean",
    },
)

whisper_base_speech_to_text = EncoderMetadata(
    name="whisper_base_speech_to_text",
    encoder=whisper_encoder.SpeechToTextEncoderV2,
    params=dict(model_path="base"),
)

whisper_base_pooled_last = EncoderMetadata(
    name="whisper_base_pooled_last",
    encoder=whisper_encoder.PooledAudioEncoderV2,
    params=dict(model_path="base", pooling="last"),
)

whisper_base_pooled_mean = EncoderMetadata(
    name="whisper_base_pooled_mean",
    encoder=whisper_encoder.PooledAudioEncoderV2,
    params=dict(model_path="base", pooling="mean"),
)

whisper_base_pooled_max = EncoderMetadata(
    name="whisper_base_pooled_max",
    encoder=whisper_encoder.PooledAudioEncoderV2,
    params=dict(model_path="base", pooling="max"),
)
