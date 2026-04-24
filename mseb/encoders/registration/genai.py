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

"""Registration for GenAI Encoders."""

from mseb.encoders import encoder_registry
from mseb.encoders import genai_embedding_encoder
from mseb.encoders import genai_llm_encoder
from mseb.encoders import prompt_registry

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="genai_llm",
        encoder=genai_llm_encoder.GenaiLLMEncoder,
        params=lambda: dict(
            model_path=genai_llm_encoder.GENAI_LLM_ENCODER_MODEL_PATH.value,
            api_key=genai_llm_encoder.GENAI_LLM_ENCODER_GEMINI_API_KEY.value,
            prompt=prompt_registry.get_prompt_metadata(
                encoder_registry.PROMPT_NAME.value
            ).load(),
        ),
        url="https://ai.google.dev/gemma",
        base_model="gemma",
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="genai_llm_transcript_truth",
        encoder=genai_llm_encoder.GenaiLLMTranscriptTruthEncoder,
        params=lambda: dict(
            model_path=genai_llm_encoder.GENAI_LLM_ENCODER_MODEL_PATH.value,
            api_key=genai_llm_encoder.GENAI_LLM_ENCODER_GEMINI_API_KEY.value,
            prompt=prompt_registry.get_prompt_metadata(
                encoder_registry.PROMPT_NAME.value
            ).load(),
        ),
        url="https://ai.google.dev/gemma",
        base_model="gemma",
        tags=("transcript_truth",),
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="genai_embedding",
        encoder=genai_embedding_encoder.GenaiEmbeddingEncoder,
        params=lambda: dict(
            model_path=genai_embedding_encoder.GENAI_EMBEDDING_ENCODER_MODEL_PATH.value,
            api_key=genai_embedding_encoder.GENAI_EMBEDDING_ENCODER_GEMINI_API_KEY.value,
            prompt=prompt_registry.get_prompt_metadata(
                encoder_registry.PROMPT_NAME.value
            ).load(),
        ),
        url="https://ai.google.dev/gemma",
        base_model="gemma",
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="genai_embedding_or_genai_embedding",
        encoder=genai_embedding_encoder.GenaiEmbeddingOrGenaiEmbeddingEncoder,
        params=lambda: dict(
            model_path=genai_embedding_encoder.GENAI_EMBEDDING_ENCODER_MODEL_PATH.value,
            api_key=genai_embedding_encoder.GENAI_EMBEDDING_ENCODER_GEMINI_API_KEY.value,
            query_prompt=prompt_registry.get_prompt_metadata(
                encoder_registry.PROMPT_NAME.value
            ).load(),
            document_prompt=prompt_registry.get_prompt_metadata(
                encoder_registry.DOCUMENT_PROMPT_NAME.value
            ).load(),
        ),
        url="https://ai.google.dev/gemma",
        base_model="gemma",
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="genai_embedding_transcript_truth",
        encoder=genai_embedding_encoder.GenaiEmbeddingTranscriptTruthEncoder,
        params=lambda: dict(
            model_path=genai_embedding_encoder.GENAI_EMBEDDING_ENCODER_MODEL_PATH.value,
            api_key=genai_embedding_encoder.GENAI_EMBEDDING_ENCODER_GEMINI_API_KEY.value,
            prompt=prompt_registry.get_prompt_metadata(
                encoder_registry.PROMPT_NAME.value
            ).load(),
        ),
        url="https://ai.google.dev/gemma",
        base_model="gemma",
        tags=("transcript_truth",),
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="genai_embedding_transcript_truth_or_genai_embedding",
        encoder=genai_embedding_encoder.GenaiEmbeddingTranscriptTruthOrGenaiEmbeddingEncoder,
        params=lambda: dict(
            model_path=genai_embedding_encoder.GENAI_EMBEDDING_ENCODER_MODEL_PATH.value,
            api_key=genai_embedding_encoder.GENAI_EMBEDDING_ENCODER_GEMINI_API_KEY.value,
            query_prompt=prompt_registry.get_prompt_metadata(
                encoder_registry.PROMPT_NAME.value
            ).load(),
            document_prompt=prompt_registry.get_prompt_metadata(
                encoder_registry.DOCUMENT_PROMPT_NAME.value
            ).load(),
        ),
        url="https://ai.google.dev/gemma",
        base_model="gemma",
    )
)
