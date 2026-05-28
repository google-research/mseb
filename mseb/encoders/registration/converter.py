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

"""Registration for Converters."""

from mseb.encoders import converter
from mseb.encoders import encoder_registry

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="sound_embedding_to_text",
        encoder=converter.SoundEmbeddingToTextConverter,
        params=lambda: dict,
    )
)
