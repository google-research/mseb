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

"""MSEB Encoder for EmbeddingGemma 2, supporting both audio and text."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TYPE_CHECKING, cast

from mseb import encoder
from mseb import types

import numpy as np
import torch

if TYPE_CHECKING:
  import sentence_transformers  # pylint: disable=g-import-not-at-top

DEFAULT_MODEL_PATH = 'google/embeddinggemma-2'
TARGET_SAMPLE_RATE = 16000


def _validate_model_kwargs(model_kwargs: Mapping[str, Any] | None) -> None:
  """Validates that model_kwargs does not request unsupported float16 dtype."""
  if not model_kwargs:
    return
  for key in ('dtype', 'torch_dtype'):
    if key not in model_kwargs:
      continue
    val = model_kwargs[key]
    if val in ('float16', 'fp16', torch.float16, np.float16):
      raise ValueError(
          'EmbeddingGemma 2 does not support float16. Please use float32 or '
          'bfloat16.'
      )


class _EmbeddingGemma2ModelLoader:
  """Loads and caches a shared EmbeddingGemma 2 SentenceTransformer model."""

  def __init__(
      self,
      model_path: str = DEFAULT_MODEL_PATH,
      device: str | None = None,
      model_kwargs: Mapping[str, Any] | None = None,
  ):
    _validate_model_kwargs(model_kwargs)
    self.model_path = model_path
    self.device = device
    self.model_kwargs = dict(model_kwargs) if model_kwargs is not None else None
    self._model: sentence_transformers.SentenceTransformer | None = None

  def load(self) -> sentence_transformers.SentenceTransformer:
    """Returns the cached SentenceTransformer model, loading it on first call."""
    if self._model is None:
      from mseb.encoders import embeddinggemma2_modeling  # pylint: disable=g-import-not-at-top,unused-import
      import sentence_transformers as st  # pylint: disable=g-import-not-at-top

      model_path = self.model_path
      self._model = st.SentenceTransformer(
          model_path, device=self.device, model_kwargs=self.model_kwargs
      )
    return self._model


class _EmbeddingGemma2SoundEncoder(encoder.MultiModalEncoder):
  """Encodes Sound inputs into embeddings using EmbeddingGemma 2."""

  def __init__(
      self,
      model_loader: _EmbeddingGemma2ModelLoader,
      normalize_embeddings: bool = True,
      truncate_dim: int | None = None,
  ):
    super().__init__()
    self._model_loader = model_loader
    self.model_path = model_loader.model_path
    self.normalize_embeddings = normalize_embeddings
    self.truncate_dim = truncate_dim
    self._model: sentence_transformers.SentenceTransformer | None = None
    self._target_sr: int = TARGET_SAMPLE_RATE

  def _setup(self) -> None:
    self._model = self._model_loader.load()
    if self._model:
      processor = getattr(self._model[0], 'processor', None)
      feature_extractor = getattr(processor, 'feature_extractor', None)
      self._target_sr = int(
          getattr(feature_extractor, 'sampling_rate', TARGET_SAMPLE_RATE)
      )

  def _check_input_types(self, batch: Sequence[types.MultiModalObject]) -> None:
    if not all(isinstance(x, types.Sound) for x in batch):
      raise ValueError(
          '_EmbeddingGemma2SoundEncoder only supports a batch of Sound inputs.'
      )

  def _encode(
      self, batch: Sequence[types.MultiModalObject]
  ) -> Sequence[types.SoundEmbedding]:
    if self._model is None:
      raise RuntimeError('Encoder is not set up. Please call .setup() first.')
    sound_batch = cast(Sequence[types.Sound], batch)
    audio_inputs = [
        {
            'audio': {
                'array': np.asarray(
                    encoder.resample_sound(
                        sound_item, target_sr=self._target_sr
                    ).waveform,
                    dtype=np.float32,
                ),
                'sampling_rate': self._target_sr,
            }
        }
        for sound_item in sound_batch
    ]
    audio_embeds = np.asarray(
        self._model.encode(
            cast(Any, audio_inputs),
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            truncate_dim=self.truncate_dim,
        ),
        dtype=np.float32,
    )
    return [
        types.SoundEmbedding(
            embedding=cast(Any, audio_embeds[i][np.newaxis, :]),
            timestamps=cast(
                Any,
                np.array([[
                    0.0,
                    sound_item.context.length / sound_item.context.sample_rate,
                ]]),
            ),
            context=sound_item.context,
        )
        for i, sound_item in enumerate(sound_batch)
    ]


class _EmbeddingGemma2TextEncoder(encoder.MultiModalEncoder):
  """Encodes Text inputs into embeddings using EmbeddingGemma 2."""

  def __init__(
      self,
      model_loader: _EmbeddingGemma2ModelLoader,
      normalizer: Callable[[str], str] | None = None,
      prompt: str | None = None,
      prompt_name: str | None = None,
      prompt_template: str | None = None,
      normalize_embeddings: bool = True,
      truncate_dim: int | None = None,
  ):
    super().__init__()
    self._model_loader = model_loader
    self.model_path = model_loader.model_path
    self.normalizer = normalizer
    self.prompt = prompt
    self.prompt_name = prompt_name
    self.prompt_template = prompt_template
    self.normalize_embeddings = normalize_embeddings
    self.truncate_dim = truncate_dim
    self._model: sentence_transformers.SentenceTransformer | None = None

  def _setup(self) -> None:
    self._model = self._model_loader.load()

  def _check_input_types(self, batch: Sequence[types.MultiModalObject]) -> None:
    if not all(isinstance(x, types.Text) for x in batch):
      raise ValueError(
          '_EmbeddingGemma2TextEncoder only supports a batch of Text inputs.'
      )

  def _normalize_optional(self, value: str | None) -> str | None:
    if value is None:
      return None
    return self.normalizer(value) if self.normalizer else value

  def _format_text(self, item: types.Text) -> str:
    text = self.normalizer(item.text) if self.normalizer else item.text
    title = (
        self._normalize_optional(item.title_text)
        if isinstance(item, types.TextWithTitleAndContext)
        else None
    )
    context = (
        self._normalize_optional(item.context_text)
        if isinstance(item, types.TextWithTitleAndContext)
        else None
    )
    resolved_title = title if title is not None else 'none'
    if self.prompt_template is not None:
      return self.prompt_template.format(
          text=text,
          title=resolved_title,
          context=context if context is not None else 'none',
      )
    if title is None and context is None:
      return text
    if context is None:
      return f'title: {resolved_title} | text: {text}'
    return f'title: {resolved_title} | context: {context} | text: {text}'

  def _encode(
      self, batch: Sequence[types.MultiModalObject]
  ) -> Sequence[types.TextEmbedding]:
    if self._model is None:
      raise RuntimeError('Encoder is not set up. Please call .setup() first.')
    text_batch = cast(Sequence[types.Text], batch)
    texts = [self._format_text(item) for item in text_batch]
    encode_kwargs: dict[str, Any] = {
        'convert_to_numpy': True,
        'normalize_embeddings': self.normalize_embeddings,
        'truncate_dim': self.truncate_dim,
    }
    if self.prompt is not None:
      encode_kwargs['prompt'] = self.prompt
    if self.prompt_name is not None:
      encode_kwargs['prompt_name'] = self.prompt_name
    text_embeds = np.asarray(
        self._model.encode(texts, **encode_kwargs),
        dtype=np.float32,
    )
    return [
        types.TextEmbedding(
            embedding=cast(Any, text_embeds[i][np.newaxis, :]),
            spans=cast(Any, np.array([[0, len(text_item.text)]])),
            context=text_item.context,
        )
        for i, text_item in enumerate(text_batch)
    ]


def EmbeddingGemma2Encoder(  # pylint: disable=invalid-name
    model_path: str = DEFAULT_MODEL_PATH,
    device: str | None = None,
    normalizer: Callable[[str], str] | None = None,
    prompt: str | None = None,
    prompt_name: str | None = None,
    prompt_template: str | None = None,
    normalize_embeddings: bool = True,
    truncate_dim: int | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
) -> encoder.CollectionEncoder:
  """Creates a multi-modal EmbeddingGemma 2 encoder for audio and text.

  Args:
    model_path: The Hugging Face model ID or local checkpoint directory.
    device: The device to load the model onto ("cuda", "cpu", or None).
    normalizer: Optional callable to normalize input text before encoding.
    prompt: Optional task prompt prefix to prepend to text inputs.
    prompt_name: Optional predefined prompt name in the SentenceTransformer
      configuration (e.g. "query", "document", "Retrieval-query").
    prompt_template: Optional format template for text inputs (with `{text}`,
      `{title}`, `{context}` placeholders).
    normalize_embeddings: Whether to L2-normalize output embeddings.
    truncate_dim: Optional Matryoshka output embedding dimension.
    model_kwargs: Optional keyword arguments forwarded to the model loader
      (`float16` is not supported).

  Returns:
    An `encoder.CollectionEncoder` supporting `types.Sound` and `types.Text`.
  """
  loader = _EmbeddingGemma2ModelLoader(
      model_path=model_path, device=device, model_kwargs=model_kwargs
  )
  sound_encoder = _EmbeddingGemma2SoundEncoder(
      model_loader=loader,
      normalize_embeddings=normalize_embeddings,
      truncate_dim=truncate_dim,
  )
  text_encoder = _EmbeddingGemma2TextEncoder(
      model_loader=loader,
      normalizer=normalizer,
      prompt=prompt,
      prompt_name=prompt_name,
      prompt_template=prompt_template,
      normalize_embeddings=normalize_embeddings,
      truncate_dim=truncate_dim,
  )
  return encoder.CollectionEncoder({
      types.Sound: sound_encoder,
      types.Text: text_encoder,
  })
