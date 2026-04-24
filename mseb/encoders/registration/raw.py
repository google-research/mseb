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

"""Registration for Raw Encoders."""

from mseb.encoders import encoder_registry
from mseb.encoders import raw_encoder

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
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
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
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
)
