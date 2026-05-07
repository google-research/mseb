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

"""Registration for Gemma JAX encoders."""

from absl import flags
from gemma import gm
from kauldron import kd
from mseb.encoders import encoder_registry
from mseb.encoders import gemma_jax_encoder
from mseb.encoders import prompt_registry

_GEMMA4_E2B_IT_JAX_CHECKPOINT_PATH = flags.DEFINE_string(
    "gemma4_e2b_it_jax_checkpoint_path",
    gm.ckpts.CheckpointPath.GEMMA4_E2B_IT,
    "Path to Gemma4 E2B IT JAX/FLAX checkpoint.",
)

_GEMMA4_E4B_IT_JAX_CHECKPOINT_PATH = flags.DEFINE_string(
    "gemma4_e4b_it_jax_checkpoint_path",
    gm.ckpts.CheckpointPath.GEMMA4_E4B_IT,
    "Path to Gemma4 E4B IT JAX/FLAX checkpoint.",
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        "gemma4_e2b_jax",
        encoder=gemma_jax_encoder.GemmaJaxWithTitleAndContextEncoder,
        params=lambda: dict(
            model=gm.nn.Gemma4_E2B(text_only=False),
            checkpoint_path=_GEMMA4_E2B_IT_JAX_CHECKPOINT_PATH.value,
            prompt=prompt_registry.get_prompt_metadata(
                encoder_registry.PROMPT_NAME.value
            ).load(),
            # go/gemma/sharding
            sharding=kd.sharding.FSDPSharding(),
        ),
        base_model="gemma4",
        url="https://github.com/google-deepmind/gemma",
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        "gemma4_e4b_jax",
        encoder=gemma_jax_encoder.GemmaJaxWithTitleAndContextEncoder,
        params=lambda: dict(
            model=gm.nn.Gemma4_E4B(text_only=False),
            checkpoint_path=_GEMMA4_E4B_IT_JAX_CHECKPOINT_PATH.value,
            prompt=prompt_registry.get_prompt_metadata(
                encoder_registry.PROMPT_NAME.value
            ).load(),
            # go/gemma/sharding
            sharding=kd.sharding.FSDPSharding(),
        ),
        base_model="gemma4",
        url="https://github.com/google-deepmind/gemma",
    )
)
