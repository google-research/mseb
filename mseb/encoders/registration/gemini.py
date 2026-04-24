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

"""Registration for Gemini Encoders."""

from absl import flags
from mseb.encoders import encoder_registry
from mseb.encoders import gemini_embedding_encoder
from mseb.encoders.registration.whisper import _WHISPER_MODEL_PATH

_GEMINI_EMBEDDING_MODEL_PATH = flags.DEFINE_string(
    "gemini_embedding_model_path",
    "gemini-embedding-001",
    "Path to Gemini embedding model.",
)

_GEMINI_EMBEDDING_TASK_TYPE = flags.DEFINE_string(
    "gemini_embedding_task_type",
    None,
    "Task type for Gemini embedding model. One of: None, RETRIEVAL_DOCUMENT,"
    " RETRIEVAL_QUERY, SEMANTIC_SIMILARITY, CLASSIFICATION, CLUSTERING",
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="gemini_embedding_text",
        encoder=gemini_embedding_encoder.GeminiEmbeddingTextEncoder,
        params=lambda: dict(
            model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
            task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
        ),
        url="https://ai.google.dev/gemini-api/docs/embeddings",
        base_model="gemini_embedding",
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="gemini_embedding_transcript_truth",
        encoder=gemini_embedding_encoder.GeminiEmbeddingTranscriptTruthEncoder,
        params=lambda: dict(
            model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
            task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
        ),
        url="https://ai.google.dev/gemini-api/docs/embeddings",
        base_model="gemini_embedding",
        tags=("transcript_truth",),
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="gemini_embedding_transcript_truth_or_gemini_embedding",
        encoder=gemini_embedding_encoder.GeminiEmbeddingTranscriptTruthOrGeminiEmbeddingEncoder,
        params=lambda: dict(
            model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
            query_task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
            document_task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
        ),
        url="https://ai.google.dev/gemini-api/docs/embeddings",
        base_model="gemini_embedding",
        tags=("transcript_truth",),
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="gemini_embedding_transcript_truth_or_gemini_embedding_no_prompt",
        encoder=gemini_embedding_encoder.GeminiEmbeddingTranscriptTruthOrGeminiEmbeddingEncoder,
        params=lambda: dict(
            model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
            query_normalizer=None,
            query_prompt_template=None,
            query_task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
            document_normalizer=None,
            document_prompt_template=None,
            document_task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
        ),
        url="https://ai.google.dev/gemini-api/docs/embeddings",
        base_model="gemini_embedding",
        tags=("transcript_truth",),
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="gemini_embedding_with_title_and_context_transcript_truth_or_gemini_embedding",
        encoder=gemini_embedding_encoder.GeminiEmbeddingWithTitleAndContextTranscriptTruthOrGeminiEmbeddingEncoder,
        params=lambda: dict(
            model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
            query_task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
            document_task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
        ),
        url="https://ai.google.dev/gemini-api/docs/embeddings",
        base_model="gemini_embedding",
        tags=("transcript_truth",),
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="gemini_embedding_whisper",
        encoder=gemini_embedding_encoder.GeminiEmbeddingWhisperEncoder,
        params=lambda: dict(
            whisper_model_path=_WHISPER_MODEL_PATH.value,
            gemini_embedding_model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
            gemini_embedding_task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
        ),
        url="https://ai.google.dev/gemini-api/docs/embeddings",
        base_model="gemini_embedding",
        tags=("cascaded",),
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="gemini_embedding_whisper_or_gemini_embedding",
        encoder=gemini_embedding_encoder.GeminiEmbeddingWhisperOrGeminiEmbeddingEncoder,
        params=lambda: dict(
            whisper_model_path=_WHISPER_MODEL_PATH.value,
            gemini_embedding_model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
            query_task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
            document_task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
        ),
        url="https://ai.google.dev/gemini-api/docs/embeddings",
        base_model="gemini_embedding",
        tags=("cascaded",),
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="gemini_embedding_whisper_or_gemini_embedding_no_prompt",
        encoder=gemini_embedding_encoder.GeminiEmbeddingWhisperOrGeminiEmbeddingEncoder,
        params=lambda: dict(
            whisper_model_path=_WHISPER_MODEL_PATH.value,
            gemini_embedding_model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
            query_normalizer=None,
            query_prompt_template=None,
            query_task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
            document_normalizer=None,
            document_prompt_template=None,
            document_task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
        ),
        url="https://ai.google.dev/gemini-api/docs/embeddings",
        base_model="gemini_embedding",
        tags=("cascaded",),
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="gemini_embedding_with_title_and_context_whisper",
        encoder=gemini_embedding_encoder.GeminiEmbeddingWithTitleAndContextWhisperEncoder,
        params=lambda: dict(
            whisper_model_path=_WHISPER_MODEL_PATH.value,
            gemini_embedding_model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
            gemini_embedding_task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
        ),
        url="https://ai.google.dev/gemini-api/docs/embeddings",
        base_model="gemini_embedding",
        tags=("cascaded",),
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="gemini_embedding_with_title_and_context_whisper_or_gemini_embedding",
        encoder=gemini_embedding_encoder.GeminiEmbeddingWithTitleAndContextWhisperOrGeminiEmbeddingEncoder,
        params=lambda: dict(
            whisper_model_path=_WHISPER_MODEL_PATH.value,
            gemini_embedding_model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
            query_task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
            document_task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
        ),
        url="https://ai.google.dev/gemini-api/docs/embeddings",
        base_model="gemini_embedding",
        tags=("cascaded",),
    )
)
