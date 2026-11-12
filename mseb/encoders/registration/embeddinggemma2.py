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

"""Registration for EmbeddingGemma 2 Encoders.

"""

from absl import flags
from mseb.encoders import embeddinggemma2_encoder
from mseb.encoders import encoder_registry

_EMBEDDINGGEMMA2_MODEL_PATH = flags.DEFINE_string(
    "embeddinggemma2_model_path",
    embeddinggemma2_encoder.DEFAULT_MODEL_PATH,
    "Path to EmbeddingGemma 2 model.",
)


encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="embeddinggemma2",
        encoder=embeddinggemma2_encoder.EmbeddingGemma2Encoder,
        params=lambda: dict(
            model_path=_EMBEDDINGGEMMA2_MODEL_PATH.value,
        ),
        url="https://huggingface.co/google/embeddinggemma-2",
        base_model="embeddinggemma-2",
    )
)
