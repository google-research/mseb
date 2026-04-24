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

"""Registration for Encodec Encoders."""

from absl import flags
from mseb.encoders import encodec_encoder
from mseb.encoders import encoder_registry

_ENCODEC_MODEL_PATH = flags.DEFINE_string(
    "encodec_model_path",
    "facebook/encodec_24khz",
    "Path to ENCODEC model.",
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="encodec_encoder_24khz",
        encoder=encodec_encoder.EncodecJointEncoder,
        params=lambda: dict(
            model_path=_ENCODEC_MODEL_PATH.value,
        ),
        url="https://huggingface.co/facebook/encodec_24khz",
        base_model="encodec",
    )
)
