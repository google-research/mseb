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

"""Registration for Gemma3n Encoders."""

from absl import flags
from mseb.encoders import encoder_registry
from mseb.encoders import gemma3n_encoder

_HF_LLM_MODEL_PATH = flags.DEFINE_string(
    "hf_llm_model_path",
    "google/gemma-3n-E2B-it",
    "Path to HF LLM model.",
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="gemma3n_audio_e2b_it",
        encoder=gemma3n_encoder.Gemma3nEncoder,
        params=lambda: dict(
            model_path=_HF_LLM_MODEL_PATH.value,
        ),
        url="https://huggingface.co/google/gemma-3n-E2B-it",
        base_model="gemma3n",
    )
)
