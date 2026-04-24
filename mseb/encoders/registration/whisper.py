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

"""Registration for Whisper Encoders."""

from absl import flags
from mseb.encoders import encoder_registry
from mseb.encoders import whisper_encoder

_WHISPER_MODEL_PATH = flags.DEFINE_string(
    "whisper_model_path",
    "large-v3",
    "Path to Whisper model.",
)

_LANGUAGE = flags.DEFINE_string(
    "language",
    "en",
    "The two-letter ISO code for the input language.",
)

# Register encoders
encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="whisper_speech_to_text",
        encoder=whisper_encoder.SpeechToTextEncoder,
        params=lambda: dict(model_path=_WHISPER_MODEL_PATH.value),
        url="https://github.com/openai/whisper",
        base_model="whisper",
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="whisper_base_speech_to_text",
        encoder=whisper_encoder.SpeechToTextEncoder,
        params=lambda: dict(model_path="base"),
        url="https://github.com/openai/whisper",
        base_model="whisper",
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="whisper_base_pooled_last",
        encoder=whisper_encoder.PooledAudioEncoder,
        params=lambda: dict(model_path="base", pooling="last"),
        url="https://github.com/openai/whisper",
        base_model="whisper",
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="whisper_base_pooled_mean",
        encoder=whisper_encoder.PooledAudioEncoder,
        params=lambda: dict(
            model_path=_WHISPER_MODEL_PATH.value, pooling="mean"
        ),
        url="https://github.com/openai/whisper",
        base_model="whisper",
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="whisper_base_pooled_max",
        encoder=whisper_encoder.PooledAudioEncoder,
        params=lambda: dict(model_path="base", pooling="max"),
        url="https://github.com/openai/whisper",
        base_model="whisper",
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="whisper_forced_alignment",
        encoder=whisper_encoder.ForcedAlignmentEncoder,
        params=lambda: dict(
            model_path=_WHISPER_MODEL_PATH.value, language=_LANGUAGE.value
        ),
        url="https://github.com/openai/whisper",
        base_model="whisper",
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="whisper_joint",
        encoder=whisper_encoder.WhisperJointEncoder,
        params=lambda: dict(model_path=_WHISPER_MODEL_PATH.value),
        url="https://github.com/openai/whisper",
        base_model="whisper",
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="whisper_base_joint",
        encoder=whisper_encoder.WhisperJointEncoder,
        params=lambda: dict(model_path="base"),
        url="https://github.com/openai/whisper",
        base_model="whisper",
    )
)
