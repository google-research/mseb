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

"""Registration for Segmentation Encoders."""

from absl import flags
from mseb.encoders import encoder_registry
from mseb.encoders import segmentation_encoder
from mseb.encoders.registration.whisper import _LANGUAGE
from mseb.encoders.registration.whisper import _WHISPER_MODEL_PATH

_IDF_TABLE_PATH = flags.DEFINE_string(
    "idf_table_path",
    "idf_table_path",
    "The path to idf table.",
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
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
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
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
)
