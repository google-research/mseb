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

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
import sys
import tempfile
from typing import Any, TYPE_CHECKING, cast
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from mseb import types
from mseb.encoders import embeddinggemma2_encoder
import numpy as np
import pytest
import safetensors.torch
import torch

_MODULES_AT_ENCODER_IMPORT = frozenset(sys.modules.keys())

pytest.importorskip('sentence_transformers')

if TYPE_CHECKING:
  from mseb.encoders import embeddinggemma2_modeling  # pylint: disable=g-import-not-at-top


def _to_checkpoint_ple_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
  """Renames model state_dict keys to the on-disk checkpoint PLE key names."""
  ple_sd: dict[str, torch.Tensor] = {}
  for k, v in state_dict.items():
    new_k = k.replace(
        'language_model.per_layer_model_projection.',
        'language_model.ple.per_layer_model_projection.',
    ).replace(
        'language_model.per_layer_projection_norm.',
        'language_model.ple.per_layer_projection_norm.',
    )
    for attr in (
        'per_layer_input_gate',
        'per_layer_projection',
        'post_per_layer_input_norm',
    ):
      new_k = new_k.replace(f'.{attr}.', f'.ple_block.{attr}.')
    ple_sd[new_k] = v
  return ple_sd


class _TinyFeatureExtractor:
  """Minimal stand-in providing `sampling_rate`."""

  def __init__(self, sampling_rate: int = 16000):
    self.sampling_rate = sampling_rate


class _TinyProcessor:
  """Minimal stand-in providing `feature_extractor`."""

  def __init__(self, sampling_rate: int = 16000):
    self.feature_extractor = _TinyFeatureExtractor(sampling_rate=sampling_rate)


class _TinyTransformerModule:
  """Minimal first module of SentenceTransformer exposing `auto_model` and `processor`."""

  def __init__(
      self,
      auto_model: embeddinggemma2_modeling.EmbeddingGemma2Model,
      sampling_rate: int = 16000,
  ):
    self.auto_model = auto_model
    self.processor = _TinyProcessor(sampling_rate=sampling_rate)


class _TinySentenceTransformer:
  """Minimal SentenceTransformer stand-in backed by a real tiny EmbeddingGemma2Model."""

  def __init__(
      self,
      model: embeddinggemma2_modeling.EmbeddingGemma2Model,
      sampling_rate: int = 16000,
  ):
    self._first_module = _TinyTransformerModule(
        auto_model=model, sampling_rate=sampling_rate
    )
    self.last_encode_kwargs: dict[str, Any] = {}
    self.last_inputs: Any = None

  def __getitem__(self, idx: int) -> Any:
    if idx == 0:
      return self._first_module
    raise IndexError(idx)

  def encode(
      self,
      inputs: Any,
      convert_to_numpy: bool = True,
      normalize_embeddings: bool = True,
      truncate_dim: int | None = None,
      **kwargs: Any,
  ) -> np.ndarray | torch.Tensor:
    del convert_to_numpy
    from mseb.encoders import embeddinggemma2_modeling  # pylint: disable=g-import-not-at-top

    self.last_inputs = inputs
    self.last_encode_kwargs = dict(
        normalize_embeddings=normalize_embeddings,
        truncate_dim=truncate_dim,
        **kwargs,
    )
    model = self._first_module.auto_model
    text_config = model.config.text_config
    audio_config = model.config.audio_config
    assert isinstance(
        text_config, embeddinggemma2_modeling.EmbeddingGemma2TextConfig
    )
    assert isinstance(audio_config, embeddinggemma2_modeling.Gemma4AudioConfig)
    vocab_size = text_config.vocab_size
    rows: list[torch.Tensor] = []
    with torch.no_grad():
      for item in inputs:
        if isinstance(item, dict) and 'audio' in item:
          arr = np.asarray(item['audio']['array'], dtype=np.float32)
          seed_val = int(np.abs(arr[:64]).sum() * 1000) + len(arr) + 7
          gen = torch.Generator().manual_seed(seed_val)
          num_freq_bins = audio_config.subsampling_conv_channels[0]
          # 8 mel frames -> 2 stride-2 conv layers -> 2 audio soft tokens.
          input_features = torch.randn((1, 8, num_freq_bins), generator=gen)
          input_features_mask = torch.ones((1, 8), dtype=torch.bool)
          ids = torch.tensor(
              [[
                  model.config.boa_token_id,
                  model.config.audio_token_id,
                  model.config.audio_token_id,
                  model.config.eoa_token_index,
              ]],
              dtype=torch.long,
          )
          mask = torch.ones_like(ids)
          mm_token_type_ids = torch.zeros_like(ids)
          hidden = model(
              input_ids=ids,
              attention_mask=mask,
              input_features=input_features,
              input_features_mask=input_features_mask,
              mm_token_type_ids=mm_token_type_ids,
          ).last_hidden_state
        else:
          text_bytes = str(item).encode('utf-8')
          seed_val = sum((i + 1) * b for i, b in enumerate(text_bytes)) + 13
          ids = torch.tensor(
              [[(seed_val + j * 31) % (vocab_size - 20) + 5 for j in range(6)]],
              dtype=torch.long,
          )
          mask = torch.ones_like(ids)
          hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
        pooled = hidden.mean(dim=1)[0]
        if truncate_dim is not None:
          pooled = pooled[:truncate_dim]
        if normalize_embeddings:
          pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        rows.append(pooled)
    stacked = torch.stack(rows, dim=0).to(torch.float32)
    return stacked.cpu().numpy()


@pytest.mark.optional
class EmbeddingGemma2EncoderTest(parameterized.TestCase):

  tiny_config: embeddinggemma2_modeling.EmbeddingGemma2Config
  tiny_model: embeddinggemma2_modeling.EmbeddingGemma2Model
  tiny_st: _TinySentenceTransformer

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    from mseb.encoders import embeddinggemma2_modeling  # pylint: disable=g-import-not-at-top

    torch.manual_seed(0)
    text_config = dict(
        vocab_size=280,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=16,
        num_global_key_value_heads=1,
        sliding_window=4,
        hidden_size_per_layer_input=16,
        embedding_dim=16,
    )
    vision_config = dict(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        patch_size=4,
        pooling_kernel_size=2,
    )
    audio_config = dict(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        subsampling_conv_channels=(16, 16),
    )
    cls.tiny_config = embeddinggemma2_modeling.EmbeddingGemma2Config(
        text_config=text_config,
        vision_config=vision_config,
        audio_config=audio_config,
        boi_token_id=261,
        eoi_token_id=262,
        image_token_id=263,
        video_token_id=264,
        boa_token_id=265,
        eoa_token_index=266,
        audio_token_id=267,
    )
    cls.tiny_model = embeddinggemma2_modeling.EmbeddingGemma2Model(
        cls.tiny_config
    ).eval()
    cls.tiny_st = _TinySentenceTransformer(cls.tiny_model)

  def setUp(self):
    super().setUp()
    self.model_path = 'dummy/embeddinggemma2'

  def test_registration_import_does_not_load_heavy_modules(self):
    for forbidden in (
        'mseb.encoders.embeddinggemma2_modeling',
        'transformers.modeling_utils',
        'torch._dynamo',
        'triton',
    ):
      self.assertNotIn(forbidden, _MODULES_AT_ENCODER_IMPORT)
    with mock.patch.dict(sys.modules):
      sys.modules.pop('mseb.encoders.embeddinggemma2_modeling', None)
      enc = embeddinggemma2_encoder.EmbeddingGemma2Encoder(
          model_path=self.model_path
      )
      self.assertIsNotNone(enc)
      self.assertNotIn('mseb.encoders.embeddinggemma2_modeling', sys.modules)

  def test_loads_complete_model_from_shards_index_and_split_tensors(self):
    from mseb.encoders import embeddinggemma2_modeling  # pylint: disable=g-import-not-at-top

    orig_sd = self.tiny_model.state_dict()
    ple_sd = _to_checkpoint_ple_state_dict(orig_sd)
    with tempfile.TemporaryDirectory() as tmp_dir:
      self.tiny_config.save_pretrained(tmp_dir)
      embed_key = 'language_model.embed_tokens.weight'
      embed_tensor = ple_sd[embed_key]
      flat = embed_tensor.reshape(-1)
      part0, part1 = [p.clone() for p in torch.tensor_split(flat, 2)]

      shard1 = {
          f'{embed_key}::part0': part0,
          **{
              k: v
              for k, v in ple_sd.items()
              if k != embed_key and k.startswith('language_model.')
          },
      }
      shard2 = {
          f'{embed_key}::part1': part1,
          **{
              k: v
              for k, v in ple_sd.items()
              if not k.startswith('language_model.')
          },
      }
      shard3 = {
          'vision_tower.unused.weight': torch.zeros((2, 2)),
          'model.vision_tower.unused.weight': torch.zeros((2, 2)),
          'embed_vision.unused.weight': torch.zeros((2, 2)),
      }
      shard1_name = 'shard-00001-of-00003.safetensors'
      shard2_name = 'shard-00002-of-00003.safetensors'
      shard3_name = 'shard-00003-of-00003.safetensors'
      safetensors.torch.save_file(
          shard1, os.path.join(tmp_dir, shard1_name), metadata={'format': 'pt'}
      )
      safetensors.torch.save_file(
          shard2, os.path.join(tmp_dir, shard2_name), metadata={'format': 'pt'}
      )
      safetensors.torch.save_file(
          shard3, os.path.join(tmp_dir, shard3_name), metadata={'format': 'pt'}
      )
      weight_map = {k: shard1_name for k in shard1}
      weight_map.update({k: shard2_name for k in shard2})
      weight_map.update({k: shard3_name for k in shard3})
      index = {
          'metadata': {'source': 'model.safetensors'},
          'weight_map': weight_map,
          'split_tensors': {
              embed_key: {
                  'parts': [f'{embed_key}::part0', f'{embed_key}::part1'],
                  'shape': list(embed_tensor.shape),
              }
          },
      }
      with open(
          os.path.join(tmp_dir, embeddinggemma2_modeling.SHARDS_INDEX_NAME),
          'w',
          encoding='utf-8',
      ) as f:
        json.dump(index, f)

      loaded = embeddinggemma2_modeling.EmbeddingGemma2Model.from_pretrained(
          tmp_dir
      ).eval()
      self.assertEqual(loaded.name_or_path, tmp_dir)
      self.assertIsNone(loaded.config.vision_config)
      self.assertIsNone(loaded.vision_tower)
      loaded_sd = loaded.state_dict()
      self.assertEqual(set(loaded_sd.keys()), set(orig_sd.keys()))
      for k, v in orig_sd.items():
        self.assertTrue(torch.equal(loaded_sd[k], v), k)

  def test_from_pretrained_raises_on_missing_keys(self):
    from mseb.encoders import embeddinggemma2_modeling  # pylint: disable=g-import-not-at-top

    incomplete_sd = _to_checkpoint_ple_state_dict(self.tiny_model.state_dict())
    incomplete_sd.pop('language_model.embedding_projection.weight')
    with tempfile.TemporaryDirectory() as tmp_dir:
      self.tiny_config.save_pretrained(tmp_dir)
      safetensors.torch.save_file(
          incomplete_sd,
          os.path.join(tmp_dir, 'model.safetensors'),
          metadata={'format': 'pt'},
      )
      with self.assertRaisesRegex(
          RuntimeError,
          r'Missing keys when loading EmbeddingGemma2Model.*'
          r'language_model\.embedding_projection\.weight',
      ):
        embeddinggemma2_modeling.EmbeddingGemma2Model.from_pretrained(tmp_dir)

  def test_unhalved_bidirectional_sliding_window_and_ple_mapping(self):
    from mseb.encoders import embeddinggemma2_modeling  # pylint: disable=g-import-not-at-top
    from transformers import masking_utils  # pylint: disable=g-import-not-at-top

    text_cfg = self.tiny_config.text_config
    assert isinstance(
        text_cfg, embeddinggemma2_modeling.EmbeddingGemma2TextConfig
    )
    # `sliding_window=4` must remain 4 (not halved to `4 // 2 + 1 = 3` as in
    # `Gemma4TextConfig`), and survive `to_dict()` round-trip without doubling.
    self.assertEqual(text_cfg.sliding_window, 4)
    reloaded_cfg = embeddinggemma2_modeling.EmbeddingGemma2TextConfig(
        **text_cfg.to_dict()
    )
    self.assertEqual(reloaded_cfg.sliding_window, 4)

    # Verify the bidirectional sliding-window mask width on a 10-token sequence
    # (`seq_len=10 > sliding_window=4`): query 0 attends to keys 0..4
    # (`|0 - kv| <= 4`, including distance 4 > W // 2 + 1 = 3), and masks out
    # keys 5..9 (`|0 - kv| >= 5 > 4`).
    dummy_embeds = torch.zeros((1, 10, text_cfg.hidden_size))
    attention_mask = torch.ones((1, 10), dtype=torch.long)
    position_ids = torch.arange(10).unsqueeze(0)
    sliding_mask = masking_utils.create_sliding_window_causal_mask(
        config=text_cfg,
        inputs_embeds=dummy_embeds,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=position_ids,
    )
    assert isinstance(sliding_mask, torch.Tensor)
    if sliding_mask.dtype == torch.bool:
      row0_allowed = [bool(x) for x in sliding_mask[0, 0, 0].tolist()]
    else:
      row0_allowed = [float(x) == 0.0 for x in sliding_mask[0, 0, 0].tolist()]
    self.assertEqual(
        row0_allowed,
        [True, True, True, True, True, False, False, False, False, False],
    )

    # Verify in the actual model that layer 0 (`sliding_attention`, W=4) at
    # token 0 changes when token 4 (distance 4 > W//2 + 1 = 3) is modified,
    # and does NOT change when token 5 (distance 5 > W = 4) is modified.
    base_ids = torch.tensor(
        [[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]], dtype=torch.long
    )
    ids_dist4 = base_ids.clone()
    ids_dist4[0, 4] = 99
    ids_dist5 = base_ids.clone()
    ids_dist5[0, 5] = 99
    layer0_outputs: list[torch.Tensor] = []
    handle = self.tiny_model.language_model.layers[0].register_forward_hook(
        lambda _mod, _inp, out: layer0_outputs.append(out.detach().clone())
    )
    try:
      with torch.no_grad():
        self.tiny_model(input_ids=base_ids, attention_mask=attention_mask)
        self.tiny_model(input_ids=ids_dist4, attention_mask=attention_mask)
        self.tiny_model(input_ids=ids_dist5, attention_mask=attention_mask)
    finally:
      handle.remove()

    self.assertLen(layer0_outputs, 3)
    out_base = layer0_outputs[0]
    out_dist4 = layer0_outputs[1]
    out_dist5 = layer0_outputs[2]
    self.assertFalse(torch.allclose(out_base[:, 0, :], out_dist4[:, 0, :]))
    torch.testing.assert_close(out_base[:, 0, :], out_dist5[:, 0, :])

  def test_loads_via_real_sentence_transformer_and_auto_processor(self):
    from mseb.encoders import embeddinggemma2_modeling  # pylint: disable=g-import-not-at-top
    from transformers.models.gemma.tokenization_gemma import GemmaTokenizer  # pylint: disable=g-import-not-at-top
    from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor  # pylint: disable=g-import-not-at-top
    from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor  # pylint: disable=g-import-not-at-top

    vocab: dict[str, int] = {
        '<pad>': 0,
        '<eos>': 1,
        '<bos>': 2,
        '<unk>': 3,
        '<mask>': 4,
    }
    for b in range(256):
      vocab[f'<0x{b:02X}>'] = 5 + b
    special_tokens = {
        'boi_token': ('<|image>', 261),
        'eoi_token': ('<image|>', 262),
        'image_token': ('<|image|>', 263),
        'video_token': ('<|video|>', 264),
        'boa_token': ('<|audio>', 265),
        'eoa_token': ('<audio|>', 266),
        'audio_token': ('<|audio|>', 267),
    }
    for tok_str, tok_id in special_tokens.values():
      vocab[tok_str] = tok_id
    for idx in range(268, 280):
      vocab[f'<unused{idx}>'] = idx

    with tempfile.TemporaryDirectory() as tmp_dir:
      self.tiny_config.save_pretrained(tmp_dir)
      ple_sd = _to_checkpoint_ple_state_dict(self.tiny_model.state_dict())
      shard_name = 'shard-00001-of-00001.safetensors'
      safetensors.torch.save_file(
          ple_sd,
          os.path.join(tmp_dir, shard_name),
          metadata={'format': 'pt'},
      )
      with open(
          os.path.join(tmp_dir, embeddinggemma2_modeling.SHARDS_INDEX_NAME),
          'w',
          encoding='utf-8',
      ) as f:
        json.dump(
            {
                'metadata': {'source': 'model.safetensors'},
                'weight_map': {k: shard_name for k in ple_sd},
                'split_tensors': {},
            },
            f,
        )

      tokenizer = GemmaTokenizer(
          vocab=vocab,
          merges=[],
          extra_special_tokens=[t for t, _ in special_tokens.values()],
          model_specific_special_tokens={
              k: t for k, (t, _) in special_tokens.items()
          },
          **{k: t for k, (t, _) in special_tokens.items()},
      )
      processor = embeddinggemma2_modeling.EmbeddingGemma2Processor(
          feature_extractor=Gemma4AudioFeatureExtractor(
              feature_size=16,
              sampling_rate=16000,
              padding_value=0.0,
              return_attention_mask=True,
          ),
          image_processor=Gemma4ImageProcessor(),
          tokenizer=tokenizer,
          video_processor=embeddinggemma2_modeling.EmbeddingGemma2VideoProcessor(),
          chat_template=(
              '{% for m in messages %}{% for c in m["content"] %}'
              '{% if c["type"] == "audio" %}<|audio|>'
              '{% elif c["type"] == "text" %}{{ c["text"] }}{% endif %}'
              '{% endfor %}{% endfor %}'
          ),
      )
      processor.save_pretrained(tmp_dir)
      # Include the extra video_processor keys from the real checkpoint's
      # `processor_config.json` (`add_timestamps`, `max_frames`,
      # `overflow_strategy`) to verify `EmbeddingGemma2VideoProcessor` accepts
      # them without error.
      proc_cfg_path = os.path.join(tmp_dir, 'processor_config.json')
      with open(proc_cfg_path, 'r', encoding='utf-8') as f:
        proc_cfg = json.load(f)
      proc_cfg.setdefault('video_processor', {}).update({
          'add_timestamps': False,
          'fps': 1,
          'max_frames': 32,
          'overflow_strategy': 'uniform',
          'video_processor_type': 'EmbeddingGemma2VideoProcessor',
      })
      with open(proc_cfg_path, 'w', encoding='utf-8') as f:
        json.dump(proc_cfg, f)

      with open(
          os.path.join(tmp_dir, 'modules.json'), 'w', encoding='utf-8'
      ) as f:
        json.dump(
            [
                {
                    'idx': 0,
                    'name': '0',
                    'path': '',
                    'type': (
                        'sentence_transformers.base.modules.transformer.Transformer'
                    ),
                },
                {
                    'idx': 1,
                    'name': '1',
                    'path': '1_Pooling',
                    'type': (
                        'sentence_transformers.sentence_transformer.modules.pooling.Pooling'
                    ),
                },
                {
                    'idx': 2,
                    'name': '2',
                    'path': '2_Normalize',
                    'type': (
                        'sentence_transformers.base.modules.normalize.Normalize'
                    ),
                },
            ],
            f,
        )
      os.makedirs(os.path.join(tmp_dir, '1_Pooling'), exist_ok=True)
      with open(
          os.path.join(tmp_dir, '1_Pooling', 'config.json'),
          'w',
          encoding='utf-8',
      ) as f:
        json.dump(
            {
                'embedding_dimension': 16,
                'pooling_mode': 'mean',
                'include_prompt': True,
            },
            f,
        )
      os.makedirs(os.path.join(tmp_dir, '2_Normalize'), exist_ok=True)
      with open(
          os.path.join(tmp_dir, '2_Normalize', 'config.json'),
          'w',
          encoding='utf-8',
      ) as f:
        json.dump({}, f)
      with open(
          os.path.join(tmp_dir, 'sentence_bert_config.json'),
          'w',
          encoding='utf-8',
      ) as f:
        json.dump(
            {
                'transformer_task': 'feature-extraction',
                'modality_config': {
                    'text': {
                        'method': 'forward',
                        'method_output_name': 'last_hidden_state',
                    },
                    'audio': {
                        'method': 'forward',
                        'method_output_name': 'last_hidden_state',
                    },
                    'message': {
                        'method': 'forward',
                        'method_output_name': 'last_hidden_state',
                        'format': 'structured',
                    },
                },
                'module_output_name': 'token_embeddings',
            },
            f,
        )
      with open(
          os.path.join(tmp_dir, 'config_sentence_transformers.json'),
          'w',
          encoding='utf-8',
      ) as f:
        json.dump(
            {
                'model_type': 'SentenceTransformer',
                'prompts': {'query': 'task: search result | query: '},
                'similarity_fn_name': 'cosine',
            },
            f,
        )

      enc = embeddinggemma2_encoder.EmbeddingGemma2Encoder(
          model_path=tmp_dir, device='cpu', truncate_dim=8
      )
      enc.setup()
      sound = types.Sound(
          waveform=cast(
              Any, np.sin(np.linspace(0, 4 * np.pi, 16000, dtype=np.float32))
          ),
          context=types.SoundContextParams(
              id='s_real', sample_rate=16000, length=16000
          ),
      )
      text = types.Text(
          text='hello world', context=types.TextContextParams(id='t_real')
      )
      sound_out = cast(Sequence[types.SoundEmbedding], enc.encode([sound]))
      text_out = cast(Sequence[types.TextEmbedding], enc.encode([text]))
      self.assertEqual(sound_out[0].embedding.shape, (1, 8))
      self.assertEqual(text_out[0].embedding.shape, (1, 8))
      np.testing.assert_allclose(
          np.linalg.norm(sound_out[0].embedding[0]), 1.0, atol=1e-2
      )
      np.testing.assert_allclose(
          np.linalg.norm(text_out[0].embedding[0]), 1.0, atol=1e-2
      )

  def test_hermetic_joint_audio_and_text_encoding_same_target(self):
    enc = embeddinggemma2_encoder.EmbeddingGemma2Encoder(
        model_path=self.model_path,
        device='cpu',
    )
    with mock.patch.object(
        embeddinggemma2_encoder._EmbeddingGemma2ModelLoader,
        'load',
        return_value=cast(Any, self.tiny_st),
    ):
      enc.setup()

    sound_1 = types.Sound(
        waveform=cast(
            Any,
            np.cos(np.linspace(0, 8 * np.pi, 48000, dtype=np.float32)),
        ),
        context=types.SoundContextParams(
            id='sound_0',
            sample_rate=48000,
            length=48000,
            language='en',
        ),
    )
    sound_2 = types.SoundWithTitleAndContext(
        waveform=cast(
            Any,
            np.sin(np.linspace(0, 4 * np.pi, 16000, dtype=np.float32)),
        ),
        title_text='Sine wave',
        context_text='Synthetic audio',
        context=types.SoundContextParams(
            id='sound_sine',
            sample_rate=16000,
            length=16000,
        ),
    )
    sound_outputs = cast(
        Sequence[types.SoundEmbedding], enc.encode([sound_1, sound_2])
    )
    self.assertLen(sound_outputs, 2)
    for orig, out in zip([sound_1, sound_2], sound_outputs):
      self.assertIsInstance(out, types.SoundEmbedding)
      self.assertEqual(out.context, orig.context)
      self.assertEqual(out.embedding.shape, (1, 16))
      self.assertEqual(out.embedding.dtype, np.float32)
      np.testing.assert_allclose(
          np.linalg.norm(out.embedding[0]), 1.0, atol=1e-2
      )
      expected_duration = orig.context.length / orig.context.sample_rate
      np.testing.assert_allclose(
          out.timestamps, np.array([[0.0, expected_duration]])
      )

    text_1 = types.Text(
        text='What is the tallest mountain?',
        context=types.TextContextParams(id='text_1'),
    )
    text_2 = types.TextWithTitleAndContext(
        text='Mount Everest is the highest mountain above sea level.',
        title_text='Mount Everest',
        context_text='Geography',
        context=types.TextContextParams(id='text_2'),
    )
    text_outputs = cast(
        Sequence[types.TextEmbedding], enc.encode([text_1, text_2])
    )
    self.assertLen(text_outputs, 2)
    for orig, out in zip([text_1, text_2], text_outputs):
      self.assertIsInstance(out, types.TextEmbedding)
      self.assertEqual(out.context, orig.context)
      self.assertEqual(out.embedding.shape, (1, 16))
      self.assertEqual(out.embedding.dtype, np.float32)
      np.testing.assert_allclose(
          np.linalg.norm(out.embedding[0]), 1.0, atol=1e-2
      )
      np.testing.assert_array_equal(out.spans, np.array([[0, len(orig.text)]]))

    self.assertFalse(
        np.allclose(sound_outputs[0].embedding, sound_outputs[1].embedding)
    )
    self.assertFalse(
        np.allclose(text_outputs[0].embedding, text_outputs[1].embedding)
    )
    similarity = float(
        np.matmul(sound_outputs[0].embedding, text_outputs[0].embedding.T)[0, 0]
    )
    self.assertBetween(similarity, -1.01, 1.01)

  def test_truncate_dim_and_prompts_hermetic(self):
    enc = embeddinggemma2_encoder.EmbeddingGemma2Encoder(
        model_path=self.model_path,
        device='cpu',
        prompt_name='query',
        truncate_dim=8,
    )
    with mock.patch.object(
        embeddinggemma2_encoder._EmbeddingGemma2ModelLoader,
        'load',
        return_value=cast(Any, self.tiny_st),
    ):
      enc.setup()

    sound = types.Sound(
        waveform=cast(Any, np.zeros(8000, dtype=np.float32)),
        context=types.SoundContextParams(
            id='s1', sample_rate=16000, length=8000
        ),
    )
    text = types.Text(
        text='search query',
        context=types.TextContextParams(id='t1'),
    )

    sound_out = cast(Sequence[types.SoundEmbedding], enc.encode([sound]))
    text_out = cast(Sequence[types.TextEmbedding], enc.encode([text]))
    self.assertEqual(
        self.tiny_st.last_encode_kwargs.get('prompt_name'), 'query'
    )
    self.assertEqual(self.tiny_st.last_encode_kwargs.get('truncate_dim'), 8)

    self.assertEqual(sound_out[0].embedding.shape, (1, 8))
    self.assertEqual(text_out[0].embedding.shape, (1, 8))
    np.testing.assert_allclose(
        np.linalg.norm(sound_out[0].embedding[0]), 1.0, atol=1e-2
    )
    np.testing.assert_allclose(
        np.linalg.norm(text_out[0].embedding[0]), 1.0, atol=1e-2
    )

  def test_prompt_template_and_normalizer(self):
    enc = embeddinggemma2_encoder.EmbeddingGemma2Encoder(
        model_path=self.model_path,
        device='cpu',
        normalizer=str.lower,
        prompt_template='title: {title} | text: {text}',
    )
    with mock.patch.object(
        embeddinggemma2_encoder._EmbeddingGemma2ModelLoader,
        'load',
        return_value=cast(Any, self.tiny_st),
    ):
      enc.setup()

    doc = types.TextWithTitleAndContext(
        text='UPPERCASE BODY',
        title_text='UPPERCASE TITLE',
        context=types.TextContextParams(id='doc_1'),
    )
    outputs = cast(Sequence[types.TextEmbedding], enc.encode([doc]))
    self.assertEqual(
        self.tiny_st.last_inputs,
        ['title: uppercase title | text: uppercase body'],
    )
    self.assertLen(outputs, 1)
    self.assertIsInstance(outputs[0], types.TextEmbedding)
    np.testing.assert_array_equal(
        outputs[0].spans, np.array([[0, len(doc.text)]])
    )

  def test_sub_encoder_input_type_validation(self):
    loader = embeddinggemma2_encoder._EmbeddingGemma2ModelLoader(
        model_path=self.model_path, device='cpu'
    )
    sound_enc = embeddinggemma2_encoder._EmbeddingGemma2SoundEncoder(
        model_loader=loader
    )
    text_enc = embeddinggemma2_encoder._EmbeddingGemma2TextEncoder(
        model_loader=loader
    )
    text = types.Text(
        text='hello', context=types.TextContextParams(id='t_wrong')
    )
    sound = types.Sound(
        waveform=cast(Any, np.zeros(1600, dtype=np.float32)),
        context=types.SoundContextParams(
            id='s_wrong', sample_rate=16000, length=1600
        ),
    )
    with self.assertRaisesRegex(ValueError, 'only supports a batch of Sound'):
      sound_enc._check_input_types([text])
    with self.assertRaisesRegex(ValueError, 'only supports a batch of Text'):
      text_enc._check_input_types([sound])

  @parameterized.parameters(
      ({'dtype': 'float16'},),
      ({'dtype': 'fp16'},),
      ({'torch_dtype': torch.float16},),
      ({'dtype': np.float16},),
  )
  def test_rejects_float16_dtype(self, bad_kwargs):
    with self.assertRaisesRegex(ValueError, 'does not support float16'):
      embeddinggemma2_encoder.EmbeddingGemma2Encoder(
          model_path=self.model_path,
          model_kwargs=bad_kwargs,
      )

  def test_encode_before_setup_raises(self):
    enc = embeddinggemma2_encoder.EmbeddingGemma2Encoder(
        model_path=self.model_path
    )
    text = types.Text(
        text='hello', context=types.TextContextParams(id='t_uninit')
    )
    with self.assertRaises(RuntimeError):
      enc.encode([text])


if __name__ == '__main__':
  absltest.main()
