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

"""Registration for ElevenLabs Encoders (official SDK).

The LiteLLM-based `elevenlabs_scribe_v2*` encoders in `litellm.py` are kept
unchanged; the encoders here use the official SDK, which is required for the
`keyterms` contextual-biasing parameter.
"""

from absl import flags
from mseb.encoders import elevenlabs_s2t_encoder
from mseb.encoders import encoder_registry

_ELEVENLABS_API_KEY = flags.DEFINE_string(
    "elevenlabs_api_key",
    "",
    "API key for the ElevenLabs API. If empty, the SDK falls back to the"
    " ELEVENLABS_API_KEY environment variable.",
)

_ELEVENLABS_S2T_MODEL_ID = flags.DEFINE_string(
    "elevenlabs_s2t_model_id",
    "scribe_v2",
    "ElevenLabs Speech-to-Text model id.",
)

_ELEVENLABS_URL = "https://elevenlabs.io/docs/capabilities/speech-to-text"
_ELEVENLABS_KEYTERMS_URL = (
    "https://elevenlabs.io/docs/capabilities/speech-to-text#keyterm-prompting"
)


def _params(no_verbatim: bool, use_keyterms: bool):
  return lambda: dict(
      api_key=_ELEVENLABS_API_KEY.value or None,
      model_id=_ELEVENLABS_S2T_MODEL_ID.value,
      no_verbatim=no_verbatim,
      use_keyterms=use_keyterms,
      # SVQ-style single-speaker queries: no speaker labels or event tags, as
      # in the LiteLLM-based `elevenlabs_scribe_v2*` registrations.
      tag_audio_events=False,
      diarize=False,
  )


# Clean / verbatim transcription through the SDK, without context. Should
# match the LiteLLM-based `elevenlabs_scribe_v2` / `_verbatim` encoders.
encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="elevenlabs_sdk_scribe_v2",
        encoder=elevenlabs_s2t_encoder.ElevenLabsSpeechToTextEncoder,
        params=_params(no_verbatim=True, use_keyterms=False),
        url=_ELEVENLABS_URL,
        base_model="elevenlabs",
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="elevenlabs_sdk_scribe_v2_verbatim",
        encoder=elevenlabs_s2t_encoder.ElevenLabsSpeechToTextEncoder,
        params=_params(no_verbatim=False, use_keyterms=False),
        url=_ELEVENLABS_URL,
        base_model="elevenlabs",
    )
)

# Same, but forwarding the contextual-biasing terms of
# `SoundWithTitleAndContext` inputs (e.g. the Contextual Earnings-22 local and
# global context tasks) as `keyterms`. Plain `Sound` inputs are transcribed
# without keyterms.
encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="elevenlabs_scribe_v2_contextual",
        encoder=elevenlabs_s2t_encoder.ElevenLabsSpeechToTextEncoder,
        params=_params(no_verbatim=True, use_keyterms=True),
        url=_ELEVENLABS_KEYTERMS_URL,
        base_model="elevenlabs",
        tags=("contextual",),
    )
)

encoder_registry.register_encoder(
    encoder_registry.EncoderMetadata(
        name="elevenlabs_scribe_v2_verbatim_contextual",
        encoder=elevenlabs_s2t_encoder.ElevenLabsSpeechToTextEncoder,
        params=_params(no_verbatim=False, use_keyterms=True),
        url=_ELEVENLABS_KEYTERMS_URL,
        base_model="elevenlabs",
        tags=("contextual",),
    )
)
