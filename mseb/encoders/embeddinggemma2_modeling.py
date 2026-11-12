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

"""HuggingFace model, config, and processor definitions for EmbeddingGemma 2.

This module has two clearly separated sections:

1. **Architecture shim (`transformers >= 5.16, < 5.20.2`)**:
   Upstream `transformers >= 5.20.2` includes
   `transformers.models.embedding_gemma2` natively. For
   `transformers >= 5.16, < 5.20.2` (which already includes `gemma4`), this
   section defines:
   - `EmbeddingGemma2TextConfig` and `EmbeddingGemma2Config`: configure
     bidirectional attention (`use_bidirectional_attention = 'all'`,
     `is_causal = False`), an un-halved `sliding_window` radius (`512` tokens
     per side), `default` RoPE on `full_attention` layers, `embedding_dim =
     768`,
     and `vision_config = None` (since MSEB uses only audio and text).
   - `EmbeddingGemma2TextModel` and `EmbeddingGemma2Model`: disable token-ID
     per-layer inputs (`embed_tokens_per_layer = None`) while keeping learned
     PLE projections, add `embedding_projection` (`hidden_size ->
     embedding_dim`),
     map checkpoint PLE weight names (`_PLE_KEY_MAPPING`), and ignore unused
     `vision_tower.` / `embed_vision.` weights.
   - `EmbeddingGemma2VideoProcessor`, `EmbeddingGemma2Processor`, and
     `register_embeddinggemma2_with_transformers`: name-only processor
     subclasses
     and `Auto*` registrations required so `AutoProcessor` /
     `SentenceTransformer`
     can resolve `"processor_class": "EmbeddingGemma2Processor"` and
     `"video_processor_type": "EmbeddingGemma2VideoProcessor"` from
     `processor_config.json`, and so `Gemma4Model.__init__` can resolve
     `AutoModel.from_config(config.text_config)`.
   TODO(mseb-dev): Delete the architecture shim and use
   `transformers.models.embedding_gemma2` once `transformers >= 5.20.2`.

2. **Sharded checkpoint loader (`SHARDS_INDEX_NAME`, `_load_sharded_state_dict`,
   `_is_vision_key`)**:
   Public Hugging Face checkpoints (`google/embeddinggemma-2`) store a standard
   unsharded `model.safetensors`, for which `_load_sharded_state_dict` returns
   `None`. Checkpoints checked into `testdata/embeddinggemma2` are split into
   `<= 100 MB` shards (`shard-00001-of-00017.safetensors`, ...) indexed by
   `shards.index.json` (with `weight_map` and `split_tensors`) to comply with
   GitHub's 100 MB per-file limit; `_load_sharded_state_dict` reassembles those
   non-vision shards into an in-memory `state_dict`.

"""

import json
import os
import re
from typing import Any, cast

import safetensors.torch
import torch
from torch import nn
import transformers
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
from transformers.models.gemma4.configuration_gemma4 import PreTrainedConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4Model
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel
from transformers.models.gemma4.processing_gemma4 import Gemma4Processor
from transformers.models.gemma4.video_processing_gemma4 import Gemma4VideoProcessor

SHARDS_INDEX_NAME = 'shards.index.json'

_PLE_KEY_MAPPING = {
    r'\bple\.': '',
    r'\.ple_block\.': '.',
}

_VISION_KEY_RE = re.compile(r'(^|\.)(vision_tower|embed_vision)\.')


def _is_vision_key(key: str) -> bool:
  """Returns True if `key` belongs to the unused vision tower or embedder."""
  return bool(_VISION_KEY_RE.search(key))


# ==============================================================================
# Section 1: Architecture shim for transformers >= 5.16, < 5.20.2
# ==============================================================================


class EmbeddingGemma2TextConfig(Gemma4TextConfig):
  """Configuration for the EmbeddingGemma 2 text backbone."""

  model_type = 'embedding_gemma2_text'
  hidden_size: int = 512
  intermediate_size: int = 2048
  num_hidden_layers: int = 24
  num_attention_heads: int = 4
  num_key_value_heads: int = 2
  max_position_embeddings: int = 262_144
  hidden_size_per_layer_input: int = 512
  vocab_size_per_layer_input: int = 1
  embedding_dim: int = 768
  use_cache: bool = False

  def __post_init__(self, **kwargs):
    # Real checkpoints include `rope_parameters`, `layer_types`, and
    # `per_layer_config` in `config.json`. When building from scratch (e.g. in
    # unit tests), upstream `EmbeddingGemma2TextConfig` differs from
    # `Gemma4TextConfig` in two defaults:
    # 1. `full_attention` uses `rope_type='default'` (not `'proportional'`).
    # 2. `num_global_key_value_heads` defaults to 1 on `full_attention` layers
    #    without gating on `attention_k_eq_v` (which remains False so
    #    `Gemma4TextAttention` instantiates separate `v_proj` weights).
    if self.rope_parameters is None:
      self.rope_parameters = {
          'sliding_attention': {'rope_type': 'default', 'rope_theta': 10_000.0},
          'full_attention': {'rope_type': 'default', 'rope_theta': 1_000_000.0},
      }
    kwargs.setdefault('num_global_key_value_heads', 1)

    # Upstream `EmbeddingGemma2TextConfig` (transformers >= 5.20.2,
    # `configuration_embedding_gemma2.py`) subclasses `PreTrainedConfig`
    # directly and stores `sliding_window=512` as the inclusive per-side radius
    # passed to `masking_utils.sliding_window_bidirectional_overlay`
    # (`abs(q_idx - kv_idx) <= sliding_window`, i.e. 512 tokens on each side).
    # `Gemma4TextConfig.__post_init__` (`configuration_gemma4.py:186-188`)
    # halves `sliding_window` (`W // 2 + 1`) when
    # `use_bidirectional_attention == 'all'` and `Gemma4TextConfig.to_dict`
    # (`configuration_gemma4.py:232-233`) doubles it back (`(W - 1) * 2`);
    # preserve the un-halved `sliding_window` across both methods.
    sliding_window = self.sliding_window
    self.attention_k_eq_v = True
    super().__post_init__(**kwargs)
    self.attention_k_eq_v = False
    self.sliding_window = sliding_window
    self.use_bidirectional_attention = 'all'
    self.is_causal = False

  def to_dict(self) -> dict[str, Any]:
    output = super().to_dict()
    output['sliding_window'] = self.sliding_window
    return output


class EmbeddingGemma2Config(Gemma4Config):
  """Multimodal configuration for EmbeddingGemma 2."""

  model_type = 'embedding_gemma2'
  sub_configs = {
      'text_config': EmbeddingGemma2TextConfig,
      'vision_config': Gemma4VisionConfig,
      'audio_config': Gemma4AudioConfig,
  }
  text_config: EmbeddingGemma2TextConfig | dict[str, Any] | None = None
  vision_config: Gemma4VisionConfig | dict[str, Any] | None = None
  audio_config: Gemma4AudioConfig | dict[str, Any] | None = None

  def __post_init__(self, **kwargs):
    if self.text_config is None:
      self.text_config = EmbeddingGemma2TextConfig()
    elif isinstance(self.text_config, dict):
      self.text_config = EmbeddingGemma2TextConfig(**self.text_config)
    # MSEB uses only audio and text; dropping vision_config here avoids
    # instantiating the 170M-parameter vision tower and keeps model.config
    # consistent with the instantiated submodules.
    self.vision_config = None
    super().__post_init__(**kwargs)


class EmbeddingGemma2TextModel(Gemma4TextModel):
  """The EmbeddingGemma 2 text backbone built on Gemma4TextModel."""

  config_class = EmbeddingGemma2TextConfig
  config: EmbeddingGemma2TextConfig
  embed_tokens_per_layer: nn.Module | None

  def __init__(self, config: EmbeddingGemma2TextConfig):
    super().__init__(config)
    self.embed_tokens_per_layer = None
    self.embedding_projection = nn.Linear(
        config.hidden_size, config.embedding_dim, bias=False
    )
    # `super().__init__(config)` calls `post_init()` before
    # `embedding_projection` is attached; call `post_init()` again so
    # `_init_weights` initializes `embedding_projection`.
    self.post_init()

  def get_per_layer_inputs(
      self,
      input_ids: torch.Tensor | None,
      inputs_embeds: torch.Tensor | None,
  ) -> torch.Tensor | None:
    del input_ids, inputs_embeds
    return None

  def forward(self, *args: Any, **kwargs: Any) -> BaseModelOutputWithPast:
    outputs = super().forward(*args, **kwargs)
    outputs.last_hidden_state = self.embedding_projection(
        outputs.last_hidden_state
    )
    return outputs


class EmbeddingGemma2Model(Gemma4Model):
  """The multimodal EmbeddingGemma 2 model built on Gemma4Model."""

  config_class = EmbeddingGemma2Config
  config: EmbeddingGemma2Config
  _keys_to_ignore_on_load_unexpected = [_VISION_KEY_RE.pattern]

  @classmethod
  def from_pretrained(
      cls,
      pretrained_model_name_or_path: str | os.PathLike[str] | None,
      *model_args: Any,
      **kwargs: Any,
  ) -> Any:
    kwargs.setdefault('key_mapping', _PLE_KEY_MAPPING)
    # Sharded checkpoint plumbing (used only when loading a directory indexed by
    # `shards.index.json`, such as `testdata/embeddinggemma2`):
    # `PreTrainedModel.from_pretrained` in transformers 5.16.1
    # (`modeling_utils.py:4210`) raises `ValueError` if both
    # `pretrained_model_name_or_path` and `state_dict` are non-None, so we pass
    # `None` and restore `name_or_path` afterwards.
    sharded_dir: str | None = None
    if (
        pretrained_model_name_or_path is not None
        and kwargs.get('state_dict') is None
    ):
      model_dir = os.fspath(pretrained_model_name_or_path)
      sharded_state_dict = _load_sharded_state_dict(model_dir)
      if sharded_state_dict is not None:
        sharded_dir = model_dir
        if not isinstance(kwargs.get('config'), PreTrainedConfig):
          kwargs['config'] = EmbeddingGemma2Config.from_pretrained(model_dir)
        kwargs['state_dict'] = sharded_state_dict
        pretrained_model_name_or_path = None

    caller_wants_loading_info = bool(kwargs.get('output_loading_info', False))
    kwargs['output_loading_info'] = True
    model, loading_info = cast(
        tuple[EmbeddingGemma2Model, dict[str, Any]],
        super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        ),
    )
    if loading_info.get('missing_keys'):
      raise RuntimeError(
          'Missing keys when loading EmbeddingGemma2Model: '
          f'{loading_info["missing_keys"]}'
      )
    if sharded_dir is not None:
      model.name_or_path = sharded_dir
      model.config.name_or_path = sharded_dir
    return (model, loading_info) if caller_wants_loading_info else model


# Name-only processor subclasses: `processor_config.json` specifies
# `"processor_class": "EmbeddingGemma2Processor"` and
# `"video_processor_type": "EmbeddingGemma2VideoProcessor"`. SentenceTransformer
# resolves both by class name via `AutoProcessor` / `AutoVideoProcessor`, and
# `_EmbeddingGemma2SoundEncoder` reads
# `processor.feature_extractor.sampling_rate`.
class EmbeddingGemma2VideoProcessor(Gemma4VideoProcessor):
  """Video processor registered under the class name in `processor_config.json`."""


class EmbeddingGemma2Processor(Gemma4Processor):
  """Processor registered under the class name in `processor_config.json`."""


def register_embeddinggemma2_with_transformers() -> None:
  """Registers EmbeddingGemma 2 classes with HuggingFace Auto classes."""
  transformers.AutoConfig.register(
      'embedding_gemma2_text', EmbeddingGemma2TextConfig, exist_ok=True
  )
  transformers.AutoConfig.register(
      'embedding_gemma2', EmbeddingGemma2Config, exist_ok=True
  )
  # `Gemma4Model.__init__` builds `self.language_model` via
  # `AutoModel.from_config(config.text_config)`, so registering
  # `EmbeddingGemma2TextModel` is required even when instantiating
  # `EmbeddingGemma2Model` directly without SentenceTransformer.
  transformers.AutoModel.register(
      EmbeddingGemma2TextConfig, EmbeddingGemma2TextModel, exist_ok=True
  )
  transformers.AutoModel.register(
      EmbeddingGemma2Config, EmbeddingGemma2Model, exist_ok=True
  )
  transformers.AutoProcessor.register(
      EmbeddingGemma2Config, EmbeddingGemma2Processor, exist_ok=True
  )
  transformers.AutoVideoProcessor.register(
      EmbeddingGemma2Config, EmbeddingGemma2VideoProcessor, exist_ok=True
  )


register_embeddinggemma2_with_transformers()


# ==============================================================================
# Section 2: Sharded checkpoint loader (`shards.index.json` + `split_tensors`)
# ==============================================================================


def _load_sharded_state_dict(
    pretrained_model_name_or_path: str | os.PathLike[str] | None,
) -> dict[str, torch.Tensor] | None:
  """Loads a `<100MB` checkpoint directory indexed by `shards.index.json`.

  Returns `None` if `pretrained_model_name_or_path` is not a local directory or
  already contains a standard `model.safetensors` /
  `model.safetensors.index.json` checkpoint.

  Args:
    pretrained_model_name_or_path: Model identifier or local directory path.

  Returns:
    A reassembled non-vision `state_dict`, or `None` if the directory does not
    use `shards.index.json`.
  """
  if pretrained_model_name_or_path is None:
    return None
  model_dir = os.fspath(pretrained_model_name_or_path)
  if not os.path.isdir(model_dir):
    return None
  if os.path.exists(os.path.join(model_dir, 'model.safetensors')) or (
      os.path.exists(os.path.join(model_dir, 'model.safetensors.index.json'))
  ):
    return None
  index_path = os.path.join(model_dir, SHARDS_INDEX_NAME)
  if not os.path.exists(index_path):
    return None

  with open(index_path, 'r', encoding='utf-8') as f:
    index = json.load(f)
  needed_shards = sorted({
      shard_file
      for key, shard_file in index['weight_map'].items()
      if not _is_vision_key(key)
  })
  state_dict: dict[str, torch.Tensor] = {}
  for shard_file in needed_shards:
    shard_path = os.path.join(model_dir, shard_file)
    for k, v in safetensors.torch.load_file(shard_path).items():
      if not _is_vision_key(k):
        state_dict[k] = v
  for name, split in index.get('split_tensors', {}).items():
    if _is_vision_key(name):
      continue
    parts = [state_dict.pop(p) for p in split['parts']]
    state_dict[name] = torch.cat(parts).reshape(split['shape'])
  return state_dict
