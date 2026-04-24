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

"""Registration for OpenAI Encoders."""

from absl import flags
from mseb.encoders import encoder_registry
from mseb.encoders import gemini_embedding_encoder
from mseb.encoders import openai_llm_encoder
from mseb.encoders import openai_s2t_encoder
from mseb.encoders import prompt_registry
from mseb.encoders.registration.gemini import _GEMINI_EMBEDDING_MODEL_PATH
from mseb.encoders.registration.gemini import _GEMINI_EMBEDDING_TASK_TYPE
from mseb.encoders.registration.whisper import _WHISPER_MODEL_PATH

_OPENAI_API_KEY = flags.DEFINE_string(
    "openai_api_key",
    "",
    "API key for OpenAI API.",
)

_OPENAI_SERVER_URL = flags.DEFINE_string(
    "openai_server_url",
    "https://generativelanguage.googleapis.com/v1beta/openai/",
    "URL of OpenAI API server.",
)

_OPENAI_MODEL_NAME = flags.DEFINE_string(
    "openai_model_name",
    "gemini-2.5-flash-lite",
    "Name of the OpenAI API model.",
)

_OPENAI_S2T_API_KEY = flags.DEFINE_string(
    "openai_s2t_api_key",
    "",
    "API key for OpenAI Audio Transcriptions API.",
)

_OPENAI_S2T_MODEL_NAME = flags.DEFINE_string(
    "openai_s2t_model_name",
    "gpt-4o-transcribe",
    "Name of the OpenAI Audio Transcriptions API model.",
)

_OPENAI_S2T_SERVER_URL = flags.DEFINE_string(
    "openai_s2t_server_url",
    "https://api.openai.com/v1",
    "URL of OpenAI S2T API server.",
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="openai_speech_to_text",
        encoder=openai_s2t_encoder.OpenAISpeechToTextEncoder,
        params=lambda: dict(
            model_name=_OPENAI_S2T_MODEL_NAME.value,
            api_key=_OPENAI_S2T_API_KEY.value,
            server_url=_OPENAI_S2T_SERVER_URL.value,
        ),
        url="https://platform.openai.com/docs/guides/speech-to-text",
        base_model="openai",
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="openai_llm_with_title_and_context",
        encoder=openai_llm_encoder.OpenAILLMWithTitleAndContextEncoder,
        params=lambda: dict(
            model_name=_OPENAI_MODEL_NAME.value,
            server_url=_OPENAI_SERVER_URL.value,
            api_key=_OPENAI_API_KEY.value,
            prompt=prompt_registry.get_prompt_metadata(
                encoder_registry.PROMPT_NAME.value
            ).load(),
        ),
        url="https://ai.google.dev/gemini-api/docs/openai",
        base_model="gemini",
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="openai_llm_with_title_and_context_transcript_truth",
        encoder=openai_llm_encoder.OpenAILLMWithTitleAndContextTranscriptTruthEncoder,
        params=lambda: dict(
            model_name=_OPENAI_MODEL_NAME.value,
            server_url=_OPENAI_SERVER_URL.value,
            api_key=_OPENAI_API_KEY.value,
            prompt=prompt_registry.get_prompt_metadata(
                encoder_registry.PROMPT_NAME.value
            ).load(),
        ),
        url="https://ai.google.dev/gemini-api/docs/openai",
        base_model="gemini",
        tags=("transcript_truth",),
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="openai_llm_with_title_and_context_whisper",
        encoder=openai_llm_encoder.OpenAILLMWithTitleAndContextWhisperEncoder,
        params=lambda: dict(
            whisper_model_path=_WHISPER_MODEL_PATH.value,
            llm_model_name=_OPENAI_MODEL_NAME.value,
            llm_server_url=_OPENAI_SERVER_URL.value,
            llm_api_key=_OPENAI_API_KEY.value,
            prompt=prompt_registry.get_prompt_metadata(
                encoder_registry.PROMPT_NAME.value
            ).load(),
        ),
        url="https://ai.google.dev/gemini-api/docs/openai",
        base_model="gemini",
        tags=("cascaded",),
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="openai_llm_rag_gemini_embedding_transcript_truth",
        encoder=openai_llm_encoder.RagOpenAILLMWithTitleAndContextTranscriptTruthEncoder,
        params=lambda: dict(
            model_name=_OPENAI_MODEL_NAME.value,
            server_url=_OPENAI_SERVER_URL.value,
            api_key=_OPENAI_API_KEY.value,
            rag_encoder=gemini_embedding_encoder.GeminiEmbeddingTextEncoder(
                model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
                normalizer=None,
                prompt_template="task: search result | query: {text}",
                task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
            ),
        ),
        url="https://ai.google.dev/gemini-api/docs/openai",
        base_model="gemini",
        tags=("transcript_truth",),
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="openai_llm_rag_gemini_embedding_whisper",
        encoder=openai_llm_encoder.RagOpenAILLMWithTitleAndContextWhisperEncoder,
        params=lambda: dict(
            whisper_model_path=_WHISPER_MODEL_PATH.value,
            llm_model_name=_OPENAI_MODEL_NAME.value,
            llm_server_url=_OPENAI_SERVER_URL.value,
            llm_api_key=_OPENAI_API_KEY.value,
            rag_encoder=gemini_embedding_encoder.GeminiEmbeddingTextEncoder(
                model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
                normalizer=None,
                prompt_template="task: search result | query: {text}",
                task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
            ),
        ),
        url="https://ai.google.dev/gemini-api/docs/openai",
        base_model="gemini",
        tags=("cascaded",),
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="openai_llm_rag_gemini_embedding",
        encoder=openai_llm_encoder.RagOpenAILLMWithTitleAndContextEncoder,
        params=lambda: dict(
            model_name=_OPENAI_MODEL_NAME.value,
            server_url=_OPENAI_SERVER_URL.value,
            api_key=_OPENAI_API_KEY.value,
            rag_encoder=gemini_embedding_encoder.GeminiEmbeddingTextEncoder(
                model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
                normalizer=None,
                prompt_template="task: search result | query: {text}",
                task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
            ),
        ),
        url="https://ai.google.dev/gemini-api/docs/openai",
        base_model="gemini",
    )
)
