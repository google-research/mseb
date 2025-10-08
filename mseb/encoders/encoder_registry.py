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

"""Registry for Encoders.

This module defines the EncoderMetadata dataclass, which holds the information
needed to instantiate and load Encoder models. It also includes definitions for
specific encoder configurations.
"""

import dataclasses
import inspect
import sys
from typing import Any, Callable
from absl import flags
from mseb import encoder as encoder_lib
from mseb.encoders import gecko_encoder
from mseb.encoders import hf_sound_encoder
from mseb.encoders import normalized_text_encoder_with_prompt as text_encoder
from mseb.encoders import raw_encoder
from mseb.encoders import wav2vec_encoder
from mseb.encoders import whisper_encoder

_GECKO_MODEL_PATH = flags.DEFINE_string(
    "gecko_model_path",
    "@gecko/gecko-1b-i18n-cpu/2",
    # "@gecko/gecko-1b-i18n-tpu/2",
    "Path to Gecko model.",
)

_WHISPER_MODEL_PATH = flags.DEFINE_string(
    "whisper_model_path",
    "large-v3",
    "Path to Whisper model.",
)

Encoder = encoder_lib.MultiModalEncoder


@dataclasses.dataclass(frozen=True)
class EncoderMetadata:
  """Metadata for an Encoder instantiated with specific parameters."""

  name: str  # The name of the encoder for creation and leaderboard entry.
  encoder: Callable[..., encoder_lib.MultiModalEncoder]
  # Lazy evaluation of the encoder parameters so we can use flags.
  params: Callable[[], dict[str, Any]]  # Additional encoder parameters.

  def load(self) -> Encoder:
    """Loads the encoder."""
    encoder = self.encoder(**self.params())  # pytype: disable=not-instantiable
    return encoder


gecko_text = EncoderMetadata(
    name="gecko_text",
    encoder=text_encoder.GeckoTextEncoder,
    params=lambda: dict(model_path=_GECKO_MODEL_PATH.value),
)

gecko_transcript_truth = EncoderMetadata(
    name="gecko_transcript_truth",
    encoder=gecko_encoder.GeckoTranscriptTruthEncoder,
    params=lambda: dict(model_path=_GECKO_MODEL_PATH.value),
)

gecko_transcript_truth_or_gecko = EncoderMetadata(
    name="gecko_transcript_truth_or_gecko",
    encoder=gecko_encoder.GeckoTranscriptTruthOrGeckoEncoder,
    params=lambda: dict(gecko_model_path=_GECKO_MODEL_PATH.value),
)

gecko_transcript_truth_or_gecko_no_prompt = EncoderMetadata(
    name="gecko_transcript_truth_or_gecko_no_prompt",
    encoder=gecko_encoder.GeckoTranscriptTruthOrGeckoEncoder,
    params=lambda: dict(
        gecko_model_path=_GECKO_MODEL_PATH.value,
        query_normalizer=None,
        query_prompt_template=None,
        document_normalizer=None,
        document_prompt_template=None,
    ),
)

gecko_with_title_and_context_transcript_truth_or_gecko = EncoderMetadata(
    name="gecko_with_title_and_context_transcript_truth_or_gecko",
    encoder=gecko_encoder.GeckoWithTitleAndContextTranscriptTruthOrGeckoEncoder,
    params=lambda: dict(gecko_model_path=_GECKO_MODEL_PATH.value),
)

gecko_whisper = EncoderMetadata(
    name="gecko_whisper",
    encoder=gecko_encoder.GeckoWhisperEncoder,
    params=lambda: dict(
        whisper_model_path=_WHISPER_MODEL_PATH.value,
        gecko_model_path=_GECKO_MODEL_PATH.value,
    ),
)

gecko_whisper_or_gecko = EncoderMetadata(
    name="gecko_whisper_or_gecko",
    encoder=gecko_encoder.GeckoWhisperOrGeckoEncoder,
    params=lambda: dict(
        whisper_model_path=_WHISPER_MODEL_PATH.value,
        gecko_model_path=_GECKO_MODEL_PATH.value,
    ),
)

gecko_whisper_or_gecko_no_prompt = EncoderMetadata(
    name="gecko_whisper_or_gecko_no_prompt",
    encoder=gecko_encoder.GeckoWhisperOrGeckoEncoder,
    params=lambda: dict(
        whisper_model_path=_WHISPER_MODEL_PATH.value,
        gecko_model_path=_GECKO_MODEL_PATH.value,
        query_normalizer=None,
        query_prompt_template=None,
        document_normalizer=None,
        document_prompt_template=None,
    ),
)

gecko_with_title_and_context_whisper = EncoderMetadata(
    name="gecko_with_title_and_context_whisper",
    encoder=gecko_encoder.GeckoWithTitleAndContextWhisperEncoder,
    params=lambda: dict(
        whisper_model_path=_WHISPER_MODEL_PATH.value,
        gecko_model_path=_GECKO_MODEL_PATH.value,
    ),
)

gecko_with_title_and_context_whisper_or_gecko = EncoderMetadata(
    name="gecko_with_title_and_context_whisper_or_gecko",
    encoder=gecko_encoder.GeckoWithTitleAndContextWhisperOrGeckoEncoder,
    params=lambda: dict(
        whisper_model_path=_WHISPER_MODEL_PATH.value,
        gecko_model_path=_GECKO_MODEL_PATH.value,
    ),
)

raw_encoder_25ms_10ms = EncoderMetadata(
    name="raw_spectrogram_25ms_10ms_mean",
    encoder=raw_encoder.RawEncoder,
    params=lambda: {
        "frame_length": 25,
        "frame_step": 10,
        "transform_fn": raw_encoder.spectrogram_transform,
        "pooling": "mean",
    },
)

whisper_base_speech_to_text = EncoderMetadata(
    name="whisper_base_speech_to_text",
    encoder=whisper_encoder.SpeechToTextEncoder,
    params=lambda: dict(model_path="base"),
)

whisper_base_pooled_last = EncoderMetadata(
    name="whisper_base_pooled_last",
    encoder=whisper_encoder.PooledAudioEncoder,
    params=lambda: dict(model_path="base", pooling="last"),
)

whisper_base_pooled_mean = EncoderMetadata(
    name="whisper_base_pooled_mean",
    encoder=whisper_encoder.PooledAudioEncoder,
    params=lambda: dict(model_path=_WHISPER_MODEL_PATH.value, pooling="mean"),
)

whisper_base_pooled_max = EncoderMetadata(
    name="whisper_base_pooled_max",
    encoder=whisper_encoder.PooledAudioEncoder,
    params=lambda: dict(model_path="base", pooling="max"),
)

wav2vec2_large_960h_lv60_pooled_mean = EncoderMetadata(
    name="wav2vec2-large-960h-lv60_pooled_mean",
    encoder=wav2vec_encoder.Wav2VecEncoder,
    params=lambda: dict(
        model_path="facebook/wav2vec2-large-960h-lv60",
        transform_fn=lambda x: x,
        pooling="mean",
        device=None,
    ),
)

hubert_large_ls960_ft_pooled_mean = EncoderMetadata(
    name="hubert_large_ls960_ft_pooled_mean",
    encoder=hf_sound_encoder.HFSoundEncoder,
    params=lambda: dict(
        model_path="facebook/hubert-large-ls960-ft",
        transform_fn=lambda x: x,
        pooling="mean",
        device=None,
    ),
)



_REGISTRY: dict[str, EncoderMetadata] = {}


def _register_encoders():
  """Finds and registers all EncoderMetadata instances in this module."""
  current_module = sys.modules[__name__]
  for name, obj in inspect.getmembers(current_module):
    if isinstance(obj, EncoderMetadata):
      if obj.name in _REGISTRY:
        raise ValueError(f"Duplicate encoder name {obj.name} found in {name}")
      _REGISTRY[obj.name] = obj


def get_encoder_metadata(name: str) -> EncoderMetadata:
  """Retrieves an EncoderMetadata instance by its name.

  Args:
    name: The unique name of the encoder metadata instance.

  Returns:
    The EncoderMetadata instance.

  Raises:
    ValueError: if the name is not found in the registry.
  """
  if not _REGISTRY:
    _register_encoders()
  if name not in _REGISTRY:
    raise ValueError(f"EncoderMetadata with name '{name}' not found.")
  return _REGISTRY[name]


# Automatically register encoders on import
_register_encoders()
