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

"""Registration for LiteLLM Encoders."""

import re

from absl import flags
from mseb.encoders import encoder_registry
from mseb.encoders import litellm_embedding_encoder
from mseb.encoders import litellm_encoder
from mseb.encoders import litellm_s2t_encoder
from mseb.encoders import prompt_registry

_LITELLM_S2T_API_KEY = flags.DEFINE_string(
    "litellm_s2t_api_key",
    "",
    "API key for LiteLLM Transcriptions API.",
)

_LITELLM_S2T_MODEL_NAME = flags.DEFINE_string(
    "litellm_s2t_model_name",
    "elevenlabs/scribe_v2",
    "Name of the LiteLLM Transcriptions API model.",
)


encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="litellm_embedding",
        encoder=litellm_embedding_encoder.LiteLLMEmbeddingOrLiteLLMEmbeddingEncoder,
        params=lambda: dict(
            model_name=litellm_embedding_encoder.LITELLM_EMBEDDING_MODEL_NAME.value,
            api_key=litellm_embedding_encoder.LITELLM_EMBEDDING_API_KEY.value,
        ),
        base_model="litellm",
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="litellm_embedding_or_litellm_embedding",
        encoder=litellm_embedding_encoder.LiteLLMEmbeddingOrLiteLLMEmbeddingEncoder,
        params=lambda: dict(
            model_name=litellm_embedding_encoder.LITELLM_EMBEDDING_MODEL_NAME.value,
            api_key=litellm_embedding_encoder.LITELLM_EMBEDDING_API_KEY.value,
            prompt_for_sound=prompt_registry.get_prompt_metadata(
                encoder_registry.PROMPT_NAME.value
            ).load(),
            prompt_for_text=prompt_registry.get_prompt_metadata(
                encoder_registry.DOCUMENT_PROMPT_NAME.value
            ).load(),
            normalizer_for_text=lambda x: re.sub(
                r"\[\d+\]", "", x.lower()
            ),
        ),
        base_model="litellm",
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="litellm_embedding_transcript_truth",
        encoder=litellm_embedding_encoder.LiteLLMEmbeddingTranscriptTruthOrLiteLLMEmbeddingEncoder,
        params=lambda: dict(
            model_name=litellm_embedding_encoder.LITELLM_EMBEDDING_MODEL_NAME.value,
            api_key=litellm_embedding_encoder.LITELLM_EMBEDDING_API_KEY.value,
        ),
        base_model="litellm",
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="litellm_embedding_transcript_truth_or_litellm_embedding",
        encoder=litellm_embedding_encoder.LiteLLMEmbeddingTranscriptTruthOrLiteLLMEmbeddingEncoder,
        params=lambda: dict(
            model_name=litellm_embedding_encoder.LITELLM_EMBEDDING_MODEL_NAME.value,
            api_key=litellm_embedding_encoder.LITELLM_EMBEDDING_API_KEY.value,
            prompt_for_sound=prompt_registry.get_prompt_metadata(
                encoder_registry.PROMPT_NAME.value
            ).load(),
            prompt_for_text=prompt_registry.get_prompt_metadata(
                encoder_registry.DOCUMENT_PROMPT_NAME.value
            ).load(),
            normalizer_for_text=lambda x: re.sub(
                r"\[\d+\]", "", x.lower()
            ),
        ),
        base_model="litellm",
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="litellm_speech_to_text",
        encoder=litellm_s2t_encoder.LiteLLMSpeechToTextEncoder,
        params=lambda: dict(
            model_name=_LITELLM_S2T_MODEL_NAME.value,
            api_key=_LITELLM_S2T_API_KEY.value,
        ),
        base_model="litellm",
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="litellm_with_title_and_context",
        encoder=litellm_encoder.LiteLLMWithTitleAndContextEncoder,
        params=lambda: dict(
            model_name=litellm_encoder.LITELLM_MODEL_NAME.value,
            api_key=litellm_encoder.LITELLM_API_KEY.value,
            prompt=prompt_registry.get_prompt_metadata(
                encoder_registry.PROMPT_NAME.value
            ).load(),
        ),
        url="https://ai.google.dev/gemini-api/docs",
        base_model="litellm",
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="litellm_with_title_and_context_transcript_truth",
        encoder=litellm_encoder.LiteLLMWithTitleAndContextTranscriptTruthEncoder,
        params=lambda: dict(
            model_name=litellm_encoder.LITELLM_MODEL_NAME.value,
            api_key=litellm_encoder.LITELLM_API_KEY.value,
            prompt=prompt_registry.get_prompt_metadata(
                encoder_registry.PROMPT_NAME.value
            ).load(),
        ),
        url="https://ai.google.dev/gemini-api",
        base_model="litellm",
        tags=("transcript_truth",),
    )
)
