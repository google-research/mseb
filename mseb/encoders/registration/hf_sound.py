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

"""Registration for HuggingFace Sound Encoders."""

from mseb.encoders import encoder_registry
from mseb.encoders import hf_sound_encoder
from mseb.encoders import wav2vec_encoder

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="wav2vec2-large-960h-lv60_pooled_mean",
        encoder=wav2vec_encoder.Wav2VecEncoder,
        params=lambda: dict(
            model_path="facebook/wav2vec2-large-960h-lv60",
            transform_fn=lambda x: x,
            pooling="mean",
            device=None,
        ),
        url="https://huggingface.co/facebook/wav2vec2-large-960h-lv60",
        base_model="wav2vec2",
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="hubert_large_ls960_ft_pooled_mean",
        encoder=hf_sound_encoder.HFSoundEncoder,
        params=lambda: dict(
            model_path="facebook/hubert-large-ls960-ft",
            transform_fn=lambda x: x,
            pooling="mean",
            device=None,
        ),
        url="https://huggingface.co/facebook/hubert-large-ls960-ft",
        base_model="hubert",
    )
)
