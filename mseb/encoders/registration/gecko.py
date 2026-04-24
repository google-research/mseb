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

"""Registration for Gecko Encoders."""

from absl import flags
from mseb.encoders import encoder_registry
from mseb.encoders import gecko_encoder
from mseb.encoders.registration.whisper import _WHISPER_MODEL_PATH

_GECKO_MODEL_PATH = flags.DEFINE_string(
    "gecko_model_path",
    "@gecko/gecko-1b-i18n-cpu/2",
    "Path to Gecko model.",
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="gecko_text",
        encoder=gecko_encoder.GeckoTextEncoder,
        params=lambda: dict(model_path=_GECKO_MODEL_PATH.value),
        url="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings",
        base_model="gecko",
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="gecko_transcript_truth",
        encoder=gecko_encoder.GeckoTranscriptTruthEncoder,
        params=lambda: dict(model_path=_GECKO_MODEL_PATH.value),
        url="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings",
        base_model="gecko",
        tags=("transcript_truth",),
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="gecko_transcript_truth_or_gecko",
        encoder=gecko_encoder.GeckoTranscriptTruthOrGeckoEncoder,
        params=lambda: dict(gecko_model_path=_GECKO_MODEL_PATH.value),
        url="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings",
        base_model="gecko",
        tags=("transcript_truth",),
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="gecko_transcript_truth_or_gecko_no_prompt",
        encoder=gecko_encoder.GeckoTranscriptTruthOrGeckoEncoder,
        params=lambda: dict(
            gecko_model_path=_GECKO_MODEL_PATH.value,
            query_normalizer=None,
            query_prompt_template=None,
            document_normalizer=None,
            document_prompt_template=None,
        ),
        url="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings",
        base_model="gecko",
        tags=("transcript_truth",),
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="gecko_with_title_and_context_transcript_truth_or_gecko",
        encoder=gecko_encoder.GeckoWithTitleAndContextTranscriptTruthOrGeckoEncoder,
        params=lambda: dict(gecko_model_path=_GECKO_MODEL_PATH.value),
        url="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings",
        base_model="gecko",
        tags=("transcript_truth",),
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="gecko_whisper",
        encoder=gecko_encoder.GeckoWhisperEncoder,
        params=lambda: dict(
            whisper_model_path=_WHISPER_MODEL_PATH.value,
            gecko_model_path=_GECKO_MODEL_PATH.value,
        ),
        url="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings",
        base_model="gecko",
        tags=("cascaded",),
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="gecko_whisper_or_gecko",
        encoder=gecko_encoder.GeckoWhisperOrGeckoEncoder,
        params=lambda: dict(
            whisper_model_path=_WHISPER_MODEL_PATH.value,
            gecko_model_path=_GECKO_MODEL_PATH.value,
        ),
        url="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings",
        base_model="gecko",
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
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
        url="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings",
        base_model="gecko",
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="gecko_with_title_and_context_whisper",
        encoder=gecko_encoder.GeckoWithTitleAndContextWhisperEncoder,
        params=lambda: dict(
            whisper_model_path=_WHISPER_MODEL_PATH.value,
            gecko_model_path=_GECKO_MODEL_PATH.value,
        ),
        url="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings",
        base_model="gecko",
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="gecko_with_title_and_context_whisper_or_gecko",
        encoder=gecko_encoder.GeckoWithTitleAndContextWhisperOrGeckoEncoder,
        params=lambda: dict(
            whisper_model_path=_WHISPER_MODEL_PATH.value,
            gecko_model_path=_GECKO_MODEL_PATH.value,
        ),
        url="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings",
        base_model="gecko",
    )
)
