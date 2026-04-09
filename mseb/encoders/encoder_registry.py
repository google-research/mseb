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
import inspect
import sys
from typing import Any, Callable

from absl import flags
from mseb import encoder as encoder_lib
from mseb.encoders import clap_encoder
from mseb.encoders import encodec_encoder
from mseb.encoders import gecko_encoder
from mseb.encoders import gemini_embedding_encoder
from mseb.encoders import gemma3n_encoder
from mseb.encoders import genai_embedding_encoder
from mseb.encoders import genai_llm_encoder
from mseb.encoders import hf_llm_encoder
from mseb.encoders import hf_sound_encoder
from mseb.encoders import litellm_embedding_encoder
from mseb.encoders import litellm_encoder
from mseb.encoders import litellm_s2t_encoder
from mseb.encoders import openai_llm_encoder
from mseb.encoders import openai_s2t_encoder
from mseb.encoders import prompt_registry
from mseb.encoders import raw_encoder
from mseb.encoders import segmentation_encoder
from mseb.encoders import wav2vec_encoder
from mseb.encoders import whisper_encoder

_PROMPT_NAME = flags.DEFINE_string(
    "prompt_name",
    "reasoning",
    "Name of the prompt to use for LLM-based encoders.",
)

_CLAP_MODEL_PATH = flags.DEFINE_string(
    "clap_model_path",
    "laion/clap-htsat-unfused",
    "Path to CLAP model.",
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

_LITELLM_EMBEDDING_API_KEY = flags.DEFINE_string(
    "litellm_embedding_api_key",
    "",
    "API key for LiteLLM Embedding API.",
)

_LITELLM_EMBEDDING_MODEL_NAME = flags.DEFINE_string(
    "litellm_embedding_model_name",
    "bedrock/amazon.nova-2-multimodal-embeddings-v1:0",
    "Name of the LiteLLM Embedding API model.",
)

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

_WHISPER_MODEL_PATH = flags.DEFINE_string(
    "whisper_model_path",
    "large-v3",
    "Path to Whisper model.",
)

_ENCODEC_MODEL_PATH = flags.DEFINE_string(
    "encodec_model_path",
    "facebook/encodec_24khz",
    "Path to ENCODEC model.",
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


gecko_text = EncoderMetadata(
    name="gecko_text",
    encoder=gecko_encoder.GeckoTextEncoder,
    params=lambda: dict(model_path=_GECKO_MODEL_PATH.value),
    url="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings",
    base_model="gecko",
)

gecko_transcript_truth = EncoderMetadata(
    name="gecko_transcript_truth",
    encoder=gecko_encoder.GeckoTranscriptTruthEncoder,
    params=lambda: dict(model_path=_GECKO_MODEL_PATH.value),
    url="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings",
    base_model="gecko",
    tags=("transcript_truth",),
)

gecko_transcript_truth_or_gecko = EncoderMetadata(
    name="gecko_transcript_truth_or_gecko",
    encoder=gecko_encoder.GeckoTranscriptTruthOrGeckoEncoder,
    params=lambda: dict(gecko_model_path=_GECKO_MODEL_PATH.value),
    url="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings",
    base_model="gecko",
    tags=("transcript_truth",),
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
    url="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings",
    base_model="gecko",
    tags=("transcript_truth",),
)

gecko_with_title_and_context_transcript_truth_or_gecko = EncoderMetadata(
    name="gecko_with_title_and_context_transcript_truth_or_gecko",
    encoder=gecko_encoder.GeckoWithTitleAndContextTranscriptTruthOrGeckoEncoder,
    params=lambda: dict(gecko_model_path=_GECKO_MODEL_PATH.value),
    url="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings",
    base_model="gecko",
    tags=("transcript_truth",),
)

gecko_whisper = EncoderMetadata(
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

gecko_whisper_or_gecko = EncoderMetadata(
    name="gecko_whisper_or_gecko",
    encoder=gecko_encoder.GeckoWhisperOrGeckoEncoder,
    params=lambda: dict(
        whisper_model_path=_WHISPER_MODEL_PATH.value,
        gecko_model_path=_GECKO_MODEL_PATH.value,
    ),
    url="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings",
    base_model="gecko",
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
    url="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings",
    base_model="gecko",
)

gecko_with_title_and_context_whisper = EncoderMetadata(
    name="gecko_with_title_and_context_whisper",
    encoder=gecko_encoder.GeckoWithTitleAndContextWhisperEncoder,
    params=lambda: dict(
        whisper_model_path=_WHISPER_MODEL_PATH.value,
        gecko_model_path=_GECKO_MODEL_PATH.value,
    ),
    url="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings",
    base_model="gecko",
)

gecko_with_title_and_context_whisper_or_gecko = EncoderMetadata(
    name="gecko_with_title_and_context_whisper_or_gecko",
    encoder=gecko_encoder.GeckoWithTitleAndContextWhisperOrGeckoEncoder,
    params=lambda: dict(
        whisper_model_path=_WHISPER_MODEL_PATH.value,
        gecko_model_path=_GECKO_MODEL_PATH.value,
    ),
    url="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings",
    base_model="gecko",
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
    url="https://en.wikipedia.org/wiki/Spectrogram",
    base_model="raw_spectrogram_25ms_10ms_mean",
)

raw_encoder_25ms_10ms_wo_pooling = EncoderMetadata(
    name="raw_spectrogram_25ms_10ms_wo_pooling",
    encoder=raw_encoder.RawEncoder,
    params=lambda: {
        "frame_length": 400,
        "frame_step": 160,
        "transform_fn": raw_encoder.spectrogram_transform,
        "pooling": None,
    },
    url="https://en.wikipedia.org/wiki/Spectrogram",
    base_model="raw_spectrogram_25ms_10ms_wo_pooling",
)

litellm_embedding = EncoderMetadata(
    name="litellm_embedding",
    encoder=litellm_embedding_encoder.LiteLLMEmbeddingEncoder,
    params=lambda: dict(
        model_name=_LITELLM_EMBEDDING_MODEL_NAME.value,
        api_key=_LITELLM_EMBEDDING_API_KEY.value,
    ),
    base_model="litellm",
)

litellm_speech_to_text = EncoderMetadata(
    name="litellm_speech_to_text",
    encoder=litellm_s2t_encoder.LiteLLMSpeechToTextEncoder,
    params=lambda: dict(
        model_name=_LITELLM_S2T_MODEL_NAME.value,
        api_key=_LITELLM_S2T_API_KEY.value,
    ),
    base_model="litellm",
)

openai_speech_to_text = EncoderMetadata(
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

whisper_speech_to_text = EncoderMetadata(
    name="whisper_speech_to_text",
    encoder=whisper_encoder.SpeechToTextEncoder,
    params=lambda: dict(model_path=_WHISPER_MODEL_PATH.value),
    url="https://github.com/openai/whisper",
    base_model="whisper",
)

whisper_base_speech_to_text = EncoderMetadata(
    name="whisper_base_speech_to_text",
    encoder=whisper_encoder.SpeechToTextEncoder,
    params=lambda: dict(model_path="base"),
    url="https://github.com/openai/whisper",
    base_model="whisper",
)

whisper_base_pooled_last = EncoderMetadata(
    name="whisper_base_pooled_last",
    encoder=whisper_encoder.PooledAudioEncoder,
    params=lambda: dict(model_path="base", pooling="last"),
    url="https://github.com/openai/whisper",
    base_model="whisper",
)

whisper_base_pooled_mean = EncoderMetadata(
    name="whisper_base_pooled_mean",
    encoder=whisper_encoder.PooledAudioEncoder,
    params=lambda: dict(model_path=_WHISPER_MODEL_PATH.value, pooling="mean"),
    url="https://github.com/openai/whisper",
    base_model="whisper",
)

whisper_base_pooled_max = EncoderMetadata(
    name="whisper_base_pooled_max",
    encoder=whisper_encoder.PooledAudioEncoder,
    params=lambda: dict(model_path="base", pooling="max"),
    url="https://github.com/openai/whisper",
    base_model="whisper",
)

whisper_forced_alignment = EncoderMetadata(
    name="whisper_forced_alignment",
    encoder=whisper_encoder.ForcedAlignmentEncoder,
    params=lambda: dict(
        model_path=_WHISPER_MODEL_PATH.value, language=_LANGUAGE.value
    ),
    url="https://github.com/openai/whisper",
    base_model="whisper",
)

whisper_joint = EncoderMetadata(
    name="whisper_joint",
    encoder=whisper_encoder.WhisperJointEncoder,
    params=lambda: dict(model_path=_WHISPER_MODEL_PATH.value),
    url="https://github.com/openai/whisper",
    base_model="whisper",
)

whisper_base_joint = EncoderMetadata(
    name="whisper_base_joint",
    encoder=whisper_encoder.WhisperJointEncoder,
    params=lambda: dict(model_path="base"),
    url="https://github.com/openai/whisper",
    base_model="whisper",
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
    url="https://huggingface.co/facebook/wav2vec2-large-960h-lv60",
    base_model="wav2vec2",
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
    url="https://huggingface.co/facebook/hubert-large-ls960-ft",
    base_model="hubert",
)

gemini_embedding_text = EncoderMetadata(
    name="gemini_embedding_text",
    encoder=gemini_embedding_encoder.GeminiEmbeddingTextEncoder,
    params=lambda: dict(
        model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
        task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
    ),
    url="https://ai.google.dev/gemini-api/docs/embeddings",
    base_model="gemini_embedding",
)

gemini_embedding_transcript_truth = EncoderMetadata(
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

gemini_embedding_transcript_truth_or_gemini_embedding = EncoderMetadata(
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
    url="https://ai.google.dev/gemini-api/docs/embeddings",
    base_model="gemini_embedding",
    tags=("transcript_truth",),
)

gemini_embedding_with_title_and_context_transcript_truth_or_gemini_embedding = EncoderMetadata(
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

gemini_embedding_whisper = EncoderMetadata(
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

gemini_embedding_whisper_or_gemini_embedding = EncoderMetadata(
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
    url="https://ai.google.dev/gemini-api/docs/embeddings",
    base_model="gemini_embedding",
    tags=("cascaded",),
)

gemini_embedding_with_title_and_context_whisper = EncoderMetadata(
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

gemini_embedding_with_title_and_context_whisper_or_gemini_embedding = EncoderMetadata(
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

genai_llm = EncoderMetadata(
    name="genai_llm",
    encoder=genai_llm_encoder.GenaiLLMEncoder,
    params=lambda: dict(
        model_path=genai_llm_encoder.GENAI_LLM_ENCODER_MODEL_PATH.value,
        api_key=genai_llm_encoder.GENAI_LLM_ENCODER_GEMINI_API_KEY.value,
        prompt=prompt_registry.get_prompt_metadata(_PROMPT_NAME.value).load(),
    ),
    url="https://ai.google.dev/gemma",
    base_model="gemma",
)

genai_llm_transcript_truth = EncoderMetadata(
    name="genai_llm_transcript_truth",
    encoder=genai_llm_encoder.GenaiLLMTranscriptTruthEncoder,
    params=lambda: dict(
        model_path=genai_llm_encoder.GENAI_LLM_ENCODER_MODEL_PATH.value,
        api_key=genai_llm_encoder.GENAI_LLM_ENCODER_GEMINI_API_KEY.value,
        prompt=prompt_registry.get_prompt_metadata(_PROMPT_NAME.value).load(),
    ),
    url="https://ai.google.dev/gemma",
    base_model="gemma",
    tags=("transcript_truth",),
)

genai_embedding = EncoderMetadata(
    name="genai_embedding",
    encoder=genai_embedding_encoder.GenaiEmbeddingEncoder,
    params=lambda: dict(
        model_path=genai_embedding_encoder.GENAI_EMBEDDING_ENCODER_MODEL_PATH.value,
        api_key=genai_embedding_encoder.GENAI_EMBEDDING_ENCODER_GEMINI_API_KEY.value,
        prompt=prompt_registry.get_prompt_metadata(_PROMPT_NAME.value).load(),
    ),
    url="https://ai.google.dev/gemma",
    base_model="gemma",
)

genai_embedding_transcript_truth = EncoderMetadata(
    name="genai_embedding_transcript_truth",
    encoder=genai_embedding_encoder.GenaiEmbeddingTranscriptTruthEncoder,
    params=lambda: dict(
        model_path=genai_embedding_encoder.GENAI_EMBEDDING_ENCODER_MODEL_PATH.value,
        api_key=genai_embedding_encoder.GENAI_EMBEDDING_ENCODER_GEMINI_API_KEY.value,
        prompt=prompt_registry.get_prompt_metadata(_PROMPT_NAME.value).load(),
    ),
    url="https://ai.google.dev/gemma",
    base_model="gemma",
    tags=("transcript_truth",),
)

litellm_with_title_and_context = EncoderMetadata(
    name="litellm_with_title_and_context",
    encoder=litellm_encoder.LiteLLMWithTitleAndContextEncoder,
    params=lambda: dict(
        model_name=litellm_encoder.LITELLM_MODEL_NAME.value,
        api_key=litellm_encoder.LITELLM_API_KEY.value,
        prompt=prompt_registry.get_prompt_metadata(_PROMPT_NAME.value).load(),
    ),
    url="https://ai.google.dev/gemini-api/docs",
    base_model="litellm",
)

litellm_with_title_and_context_transcript_truth = EncoderMetadata(
    name="litellm_with_title_and_context_transcript_truth",
    encoder=litellm_encoder.LiteLLMWithTitleAndContextTranscriptTruthEncoder,
    params=lambda: dict(
        model_name=litellm_encoder.LITELLM_MODEL_NAME.value,
        api_key=litellm_encoder.LITELLM_API_KEY.value,
        prompt=prompt_registry.get_prompt_metadata(_PROMPT_NAME.value).load(),
    ),
    url="https://ai.google.dev/gemini-api",
    base_model="litellm",
    tags=("transcript_truth",),
)

retrieval_gemini_embedding_transcript_truth = EncoderMetadata(
    name="retrieval_gemini_embedding_transcript_truth",
    encoder=gemini_embedding_encoder.RetrievalGeminiEmbeddingTranscriptTruthEncoder,
    params=lambda: dict(
        model_path=_GEMINI_EMBEDDING_MODEL_PATH.value,
        task_type=_GEMINI_EMBEDDING_TASK_TYPE.value,
        top_k=_RETRIEVAL_TOP_K.value,
    ),
    base_model="gemini_embedding",
    tags=("transcript_truth",),
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
    base_model="gemini_embedding",
    tags=("cascaded",),
)


hf_llm_with_title_and_context = EncoderMetadata(
    name="hf_llm_with_title_and_context",
    encoder=hf_llm_encoder.HFLLMWithTitleAndContextEncoder,
    params=lambda: dict(
        model_path=_HF_LLM_MODEL_PATH.value,
        prompt=prompt_registry.get_prompt_metadata(_PROMPT_NAME.value).load(),
    ),
    url="https://huggingface.co/google/gemma-3n-E2B-it",
    base_model="gemma",
)

hf_llm_with_title_and_context_transcript_truth = EncoderMetadata(
    name="hf_llm_with_title_and_context_transcript_truth",
    encoder=hf_llm_encoder.HFLLMWithTitleAndContextTranscriptTruthEncoder,
    params=lambda: dict(
        model_path=_HF_LLM_MODEL_PATH.value,
        prompt=prompt_registry.get_prompt_metadata(_PROMPT_NAME.value).load(),
    ),
    url="https://huggingface.co/google/gemma-3n-E2B-it",
    base_model="gemma",
    tags=("transcript_truth",),
)

hf_llm_with_title_and_context_whisper = EncoderMetadata(
    name="hf_llm_with_title_and_context_whisper",
    encoder=hf_llm_encoder.HFLLMWithTitleAndContextWhisperEncoder,
    params=lambda: dict(
        whisper_model_path=_WHISPER_MODEL_PATH.value,
        model_path=_HF_LLM_MODEL_PATH.value,
        prompt=prompt_registry.get_prompt_metadata(_PROMPT_NAME.value).load(),
    ),
    url="https://huggingface.co/google/gemma-3n-E2B-it",
    base_model="gemma",
    tags=("cascaded",),
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
    url="https://huggingface.co/google/gemma-3n-E2B-it",
    base_model="gemma",
    tags=("transcript_truth",),
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
    url="https://huggingface.co/google/gemma-3n-E2B-it",
    base_model="gemma",
    tags=("cascaded",),
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
    url="https://huggingface.co/google/gemma-3n-E2B-it",
    base_model="gemma",
)


gemma3n_audio_e2b_it = EncoderMetadata(
    name="gemma3n_audio_e2b_it",
    encoder=gemma3n_encoder.Gemma3nEncoder,
    params=lambda: dict(
        model_path=_HF_LLM_MODEL_PATH.value,
    ),
    url="https://huggingface.co/google/gemma-3n-E2B-it",
    base_model="gemma3n",
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
    url="https://ai.google.dev/gemini-api/docs/openai",
    base_model="gemini",
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
    url="https://ai.google.dev/gemini-api/docs/openai",
    base_model="gemini",
    tags=("transcript_truth",),
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
    url="https://ai.google.dev/gemini-api/docs/openai",
    base_model="gemini",
    tags=("cascaded",),
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
    url="https://ai.google.dev/gemini-api/docs/openai",
    base_model="gemini",
    tags=("transcript_truth",),
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
    url="https://ai.google.dev/gemini-api/docs/openai",
    base_model="gemini",
    tags=("cascaded",),
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
    url="https://ai.google.dev/gemini-api/docs/openai",
    base_model="gemini",
)

# Segmentation encoders:
whisper_base_asr_saliency = EncoderMetadata(
    name="whisper_base_asr_saliency",
    encoder=segmentation_encoder.create_asr_saliency_cascade,
    params=lambda: dict(
        whisper_model_path="base",
        language=_LANGUAGE.value,
        top_k=3,
        idf_table_path=_IDF_TABLE_PATH.value,
    ),
    url="https://github.com/openai/whisper",
    base_model="whisper",
)

whisper_large_asr_saliency = EncoderMetadata(
    name="whisper_large_asr_saliency",
    encoder=segmentation_encoder.create_asr_saliency_cascade,
    params=lambda: dict(
        whisper_model_path=_WHISPER_MODEL_PATH.value,
        language=_LANGUAGE.value,
        top_k=3,
        idf_table_path=_IDF_TABLE_PATH.value,
    ),
    url="https://github.com/openai/whisper",
    base_model="whisper",
)

# Classification encoders:
laion_clap_encoder = EncoderMetadata(
    name="laion_clap_encoder",
    encoder=clap_encoder.ClapEncoder,
    params=lambda: dict(
        model_path=_CLAP_MODEL_PATH.value,
    ),
    url="https://huggingface.co/laion/clap-htsat-unfused",
    base_model="clap",
)

encodec_encoder_24khz = EncoderMetadata(
    name="encodec_encoder_24khz",
    encoder=encodec_encoder.EncodecJointEncoder,
    params=lambda: dict(
        model_path=_ENCODEC_MODEL_PATH.value,
    ),
    url="https://huggingface.co/facebook/encodec_24khz",
    base_model="encodec",
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
