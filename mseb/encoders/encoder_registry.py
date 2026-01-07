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
from mseb.encoders import clap_encoder
from mseb.encoders import gecko_encoder
from mseb.encoders import gemini_embedding_encoder
from mseb.encoders import gemma_encoder
from mseb.encoders import hf_llm_encoder
from mseb.encoders import hf_sound_encoder
from mseb.encoders import openai_llm_encoder
from mseb.encoders import prompt_registry
from mseb.encoders import raw_encoder
from mseb.encoders import segmentation_encoder
from mseb.encoders import text_encoder_with_prompt as prompt_encoder
from mseb.encoders import wav2vec_encoder
from mseb.encoders import whisper_encoder

_PROMPT_NAME = flags.DEFINE_string(
    "prompt_name",
    "reasoning",
    "Name of the prompt to use for LLM-based encoders.",
)

_GECKO_MODEL_PATH = flags.DEFINE_string(
    "gecko_model_path",
    "@gecko/gecko-1b-i18n-cpu/2",
    # "@gecko/gecko-1b-i18n-tpu/2",
    "Path to Gecko model.",
)

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

_GEMMA_URL = flags.DEFINE_string(
    "gemma_url",
    "google/gemma-3-27b-it",
    "URL of Evergreen server serving the Gemma model.",
)

_HF_LLM_MODEL_PATH = flags.DEFINE_string(
    "hf_llm_model_path",
    "google/gemma-3n-E2B-it",
    "Path to HF LLM model.",
)

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

_IDF_TABLE_PATH = flags.DEFINE_string(
    "idf_table_path",
    "idf_table_path",
    "The path to idf table.",
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
    encoder=prompt_encoder.GeckoTextEncoder,
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

whisper_speech_to_text = EncoderMetadata(
    name="whisper_speech_to_text",
    encoder=whisper_encoder.SpeechToTextEncoder,
    params=lambda: dict(model_path=_WHISPER_MODEL_PATH.value),
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

whisper_forced_alignment = EncoderMetadata(
    name="whisper_forced_alignment",
    encoder=whisper_encoder.ForcedAlignmentEncoder,
    params=lambda: dict(
        model_path=_WHISPER_MODEL_PATH.value,
        language=_LANGUAGE.value
    )
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

gemini_embedding_text = EncoderMetadata(
    name="gemini_embedding_text",
    encoder=gemini_embedding_encoder.GeminiEmbeddingTextEncoder,
    params=lambda: dict(
        model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
        task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
    ),
)

gemini_embedding_transcript_truth = EncoderMetadata(
    name="gemini_embedding_transcript_truth",
    encoder=gemini_embedding_encoder.GeminiEmbeddingTranscriptTruthEncoder,
    params=lambda: dict(
        model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
        task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
    ),
)

gemini_embedding_transcript_truth_or_gemini_embedding = EncoderMetadata(
    name="gemini_embedding_transcript_truth_or_gemini_embedding",
    encoder=gemini_embedding_encoder.GeminiEmbeddingTranscriptTruthOrGeminiEmbeddingEncoder,
    params=lambda: dict(
        model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
        query_task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
        document_task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
    ),
)

gemini_embedding_transcript_truth_or_gemini_embedding_no_prompt = EncoderMetadata(
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
)

gemini_embedding_with_title_and_context_transcript_truth_or_gemini_embedding = EncoderMetadata(
    name="gemini_embedding_with_title_and_context_transcript_truth_or_gemini_embedding",
    encoder=gemini_embedding_encoder.GeminiEmbeddingWithTitleAndContextTranscriptTruthOrGeminiEmbeddingEncoder,
    params=lambda: dict(
        model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
        query_task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
        document_task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
    ),
)

gemini_embedding_whisper = EncoderMetadata(
    name="gemini_embedding_whisper",
    encoder=gemini_embedding_encoder.GeminiEmbeddingWhisperEncoder,
    params=lambda: dict(
        whisper_model_path=_WHISPER_MODEL_PATH.value,
        gemini_embedding_model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
        gemini_embedding_task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
    ),
)

gemini_embedding_whisper_or_gemini_embedding = EncoderMetadata(
    name="gemini_embedding_whisper_or_gemini_embedding",
    encoder=gemini_embedding_encoder.GeminiEmbeddingWhisperOrGeminiEmbeddingEncoder,
    params=lambda: dict(
        whisper_model_path=_WHISPER_MODEL_PATH.value,
        gemini_embedding_model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
        query_task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
        document_task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
    ),
)

gemini_embedding_whisper_or_gemini_embedding_no_prompt = EncoderMetadata(
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
)

gemini_embedding_with_title_and_context_whisper = EncoderMetadata(
    name="gemini_embedding_with_title_and_context_whisper",
    encoder=gemini_embedding_encoder.GeminiEmbeddingWithTitleAndContextWhisperEncoder,
    params=lambda: dict(
        whisper_model_path=_WHISPER_MODEL_PATH.value,
        gemini_embedding_model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
        gemini_embedding_task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
    ),
)

gemini_embedding_with_title_and_context_whisper_or_gemini_embedding = EncoderMetadata(
    name="gemini_embedding_with_title_and_context_whisper_or_gemini_embedding",
    encoder=gemini_embedding_encoder.GeminiEmbeddingWithTitleAndContextWhisperOrGeminiEmbeddingEncoder,
    params=lambda: dict(
        whisper_model_path=_WHISPER_MODEL_PATH.value,
        gemini_embedding_model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
        query_task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
        document_task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
    ),
)

gemma_with_title_and_context_transcript_truth = EncoderMetadata(
    name="gemma_with_title_and_context_transcript_truth",
    encoder=gemma_encoder.GemmaWithTitleAndContextTranscriptTruthEncoder,
    params=lambda: dict(
        model_path=_GEMMA_URL.value,
        prompt=prompt_registry.get_prompt_metadata(_PROMPT_NAME.value).load(),
    ),
)

gemma_with_title_and_context_whisper = EncoderMetadata(
    name="gemma_with_title_and_context_whisper",
    encoder=gemma_encoder.GemmaWithTitleAndContextWhisperEncoder,
    params=lambda: dict(
        whisper_model_path=_WHISPER_MODEL_PATH.value,
        gemma_model_path=_GEMMA_URL.value,
        prompt=prompt_registry.get_prompt_metadata(_PROMPT_NAME.value).load(),
    ),
)

gemma_with_title_and_context = EncoderMetadata(
    name="gemma_with_title_and_context",
    encoder=gemma_encoder.GemmaWithTitleAndContextEncoder,
    params=lambda: dict(
        model_path=_GEMMA_URL.value,
        prompt=prompt_registry.get_prompt_metadata(_PROMPT_NAME.value).load(),
    ),
)

retrieval_gemini_embedding_transcript_truth = EncoderMetadata(
    name="retrieval_gemini_embedding_transcript_truth",
    encoder=gemini_embedding_encoder.RetrievalGeminiEmbeddingTranscriptTruthEncoder,
    params=lambda: dict(
        model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
        task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
        top_k=100,
    ),
)

retrieval_gemini_embedding_whisper = EncoderMetadata(
    name="retrieval_gemini_embedding_whisper",
    encoder=gemini_embedding_encoder.RetrievalGeminiEmbeddingWhisperEncoder,
    params=lambda: dict(
        whisper_model_path=_WHISPER_MODEL_PATH.value,
        gemini_embedding_model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
        gemini_embedding_task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
        top_k=100,
    ),
)

gemma_rag_gemini_embedding_transcript_truth = EncoderMetadata(
    name="gemma_rag_gemini_embedding_transcript_truth",
    encoder=gemma_encoder.RagGemmaWithTitleAndContextTranscriptTruthEncoder,
    params=lambda: dict(
        model_path=_GEMMA_URL.value,
        rag_encoder=gemini_embedding_encoder.GeminiEmbeddingTextEncoder(
            model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
            normalizer=None,
            prompt_template="task: search result | query: {text}",
            task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
        ),
    ),
)

gemma_rag_gemini_embedding_whisper = EncoderMetadata(
    name="gemma_rag_gemini_embedding_whisper",
    encoder=gemma_encoder.RagGemmaWithTitleAndContextWhisperEncoder,
    params=lambda: dict(
        whisper_model_path=_WHISPER_MODEL_PATH.value,
        gemma_model_path=_GEMMA_URL.value,
        rag_encoder=gemini_embedding_encoder.GeminiEmbeddingTextEncoder(
            model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
            normalizer=None,
            prompt_template="task: search result | query: {text}",
            task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
        ),
    ),
)

gemma_rag_gemini_embedding = EncoderMetadata(
    name="gemma_rag_gemini_embedding",
    encoder=gemma_encoder.RagGemmaWithTitleAndContextEncoder,
    params=lambda: dict(
        model_path=_GEMMA_URL.value,
        rag_encoder=gemini_embedding_encoder.GeminiEmbeddingTextEncoder(
            model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
            normalizer=None,
            prompt_template="task: search result | query: {text}",
            task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
        ),
    ),
)


hf_llm_with_title_and_context = EncoderMetadata(
    name="hf_llm_with_title_and_context",
    encoder=hf_llm_encoder.HFLLMWithTitleAndContextEncoder,
    params=lambda: dict(
        model_path=_HF_LLM_MODEL_PATH.value,
        prompt=prompt_registry.get_prompt_metadata(_PROMPT_NAME.value).load(),
    ),
)

hf_llm_with_title_and_context_transcript_truth = EncoderMetadata(
    name="hf_llm_with_title_and_context_transcript_truth",
    encoder=hf_llm_encoder.HFLLMWithTitleAndContextTranscriptTruthEncoder,
    params=lambda: dict(
        model_path=_HF_LLM_MODEL_PATH.value,
        prompt=prompt_registry.get_prompt_metadata(_PROMPT_NAME.value).load(),
    ),
)

hf_llm_with_title_and_context_whisper = EncoderMetadata(
    name="hf_llm_with_title_and_context_whisper",
    encoder=hf_llm_encoder.HFLLMWithTitleAndContextWhisperEncoder,
    params=lambda: dict(
        whisper_model_path=_WHISPER_MODEL_PATH.value,
        model_path=_HF_LLM_MODEL_PATH.value,
        prompt=prompt_registry.get_prompt_metadata(_PROMPT_NAME.value).load(),
    ),
)

hf_llm_rag_gemini_embedding_transcript_truth = EncoderMetadata(
    name="hf_llm_rag_gemini_embedding_transcript_truth",
    encoder=hf_llm_encoder.RagHFLLMWithTitleAndContextTranscriptTruthEncoder,
    params=lambda: dict(
        model_path=_HF_LLM_MODEL_PATH.value,
        rag_encoder=gemini_embedding_encoder.GeminiEmbeddingTextEncoder(
            model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
            normalizer=None,
            prompt_template="task: search result | query: {text}",
            task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
        ),
    ),
)

hf_llm_rag_gemini_embedding_whisper = EncoderMetadata(
    name="hf_llm_rag_gemini_embedding_whisper",
    encoder=hf_llm_encoder.RagHFLLMWithTitleAndContextWhisperEncoder,
    params=lambda: dict(
        whisper_model_path=_WHISPER_MODEL_PATH.value,
        hf_llm_model_path=_HF_LLM_MODEL_PATH.value,
        rag_encoder=gemini_embedding_encoder.GeminiEmbeddingTextEncoder(
            model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
            normalizer=None,
            prompt_template="task: search result | query: {text}",
            task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
        ),
    ),
)

hf_llm_rag_gemini_embedding = EncoderMetadata(
    name="hf_llm_rag_gemini_embedding",
    encoder=hf_llm_encoder.RagHFLLMWithTitleAndContextEncoder,
    params=lambda: dict(
        model_path=_HF_LLM_MODEL_PATH.value,
        rag_encoder=gemini_embedding_encoder.GeminiEmbeddingTextEncoder(
            model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
            normalizer=None,
            prompt_template="task: search result | query: {text}",
            task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
        ),
    ),
)

openai_llm_with_title_and_context = EncoderMetadata(
    name="openai_llm_with_title_and_context",
    encoder=openai_llm_encoder.OpenAILLMWithTitleAndContextEncoder,
    params=lambda: dict(
        model_name=_OPENAI_MODEL_NAME.value,
        server_url=_OPENAI_SERVER_URL.value,
        api_key=_OPENAI_API_KEY.value,
        prompt=prompt_registry.get_prompt_metadata(_PROMPT_NAME.value).load(),
    ),
)

openai_llm_with_title_and_context_transcript_truth = EncoderMetadata(
    name="openai_llm_with_title_and_context_transcript_truth",
    encoder=openai_llm_encoder.OpenAILLMWithTitleAndContextTranscriptTruthEncoder,
    params=lambda: dict(
        model_name=_OPENAI_MODEL_NAME.value,
        server_url=_OPENAI_SERVER_URL.value,
        api_key=_OPENAI_API_KEY.value,
        prompt=prompt_registry.get_prompt_metadata(_PROMPT_NAME.value).load(),
    ),
)

openai_llm_with_title_and_context_whisper = EncoderMetadata(
    name="openai_llm_with_title_and_context_whisper",
    encoder=openai_llm_encoder.OpenAILLMWithTitleAndContextWhisperEncoder,
    params=lambda: dict(
        whisper_model_path=_WHISPER_MODEL_PATH.value,
        llm_model_name=_OPENAI_MODEL_NAME.value,
        llm_server_url=_OPENAI_SERVER_URL.value,
        llm_api_key=_OPENAI_API_KEY.value,
        prompt=prompt_registry.get_prompt_metadata(_PROMPT_NAME.value).load(),
    ),
)

openai_llm_rag_gemini_embedding_transcript_truth = EncoderMetadata(
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
)

openai_llm_rag_gemini_embedding_whisper = EncoderMetadata(
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
)

openai_llm_rag_gemini_embedding = EncoderMetadata(
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
)

# Segmentation encoders:
whisper_base_asr_saliency = EncoderMetadata(
    name="whisper_base_asr_saliency",
    encoder=segmentation_encoder.create_asr_saliency_cascade,
    params=lambda: dict(
        whisper_model_path="base",
        language=_LANGUAGE.value,
        top_k=3,
        idf_table_path=_IDF_TABLE_PATH.value
    )
)

whisper_large_asr_saliency = EncoderMetadata(
    name="whisper_large_asr_saliency",
    encoder=segmentation_encoder.create_asr_saliency_cascade,
    params=lambda: dict(
        whisper_model_path=_WHISPER_MODEL_PATH.value,
        language=_LANGUAGE.value,
        top_k=3,
        idf_table_path=_IDF_TABLE_PATH.value
    )
)

# Classification encoders:
laion_clap_encoder = EncoderMetadata(
    name="laion_clap_encoder",
    encoder=clap_encoder.ClapEncoder,
    params=lambda: dict(
        model_path="laion/clap-htsat-unfused",
    )
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
