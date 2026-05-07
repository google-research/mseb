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

"""Gemma encoder implementation using the open-source Gemma JAX library."""

from typing import Callable, Mapping, Optional, Tuple

from etils import epath
from gemma import gm
from kauldron import kd
import librosa
from mseb import encoder
from mseb import utils
from mseb.encoders import converter
from mseb.encoders import prompt as prompt_lib
from mseb.encoders import text_encoder_with_prompt as prompt_encoder
import numpy as np


class GemmaJaxEncoder(prompt_encoder.TextEncoderWithPrompt):
  """Encoder using the Gemma JAX library."""

  def __init__(
      self,
      model: gm.nn.TransformerLike,
      checkpoint_path: epath.PathLike,
      normalizer: Callable[[str], str] | None = None,
      prompt: prompt_lib.Prompt = prompt_lib.DefaultPrompt(),
      task_prompts: Mapping[str, prompt_lib.Prompt] | None = None,
      audio_sample_rate: int = 16000,
      sharding: kd.sharding.ShardingTree | None = None,
  ):
    super().__init__(
        normalizer=normalizer,
        prompt=prompt,
        task_prompts=task_prompts,
    )
    self.model = model
    self.checkpoint_path = checkpoint_path
    self.audio_sample_rate = audio_sample_rate
    self.sampler = None
    self.sharding = sharding

  def _setup(self):
    if self.sampler is None:
      params = gm.ckpts.load_params(
          self.checkpoint_path,
          text_only=False,
          sharding=self.sharding,
      )
      self.sampler = gm.text.ChatSampler(
          model=self.model,
          params=params,
          multi_turn=False,
          audio_sample_rate=self.audio_sample_rate,
      )

      self.prompt_encode_fn = lambda prompts: np.array([
          GemmaJaxEncoder._get_response(
              request_prompt=prompt, sampler=self.sampler
          )
          for prompt in prompts
      ])

  @staticmethod
  def _get_response(
      request_prompt: Tuple[str, Optional[bytes]],
      *,
      sampler: gm.text.ChatSampler,
  ) -> str:
    text_prompt = request_prompt[0]
    waveform = None
    if request_prompt[1] is not None:
      waveform, sample_rate = utils.wav_bytes_to_waveform(request_prompt[1])
      waveform = librosa.resample(
          waveform, orig_sr=sample_rate, target_sr=sampler.audio_sample_rate
      )
      text_prompt += "<|audio|>"
    response = sampler.chat(
        prompt=text_prompt,
        audio=[waveform] if waveform is not None else None,
    )
    return response.removesuffix("<turn|>").strip()


def GemmaJaxWithTitleAndContextEncoder(
    model: gm.nn.TransformerLike,
    checkpoint_path: epath.PathLike,
    normalizer: Callable[[str], str] | None = None,
    prompt: prompt_lib.Prompt = prompt_lib.DefaultPrompt(),
    task_prompts: Mapping[str, prompt_lib.Prompt] | None = None,
    audio_sample_rate: int = 16000,
    sharding: kd.sharding.ShardingTree | None = None,
) -> encoder.CascadeEncoder:
  """E2E HF LLM encoder with title and context."""

  return encoder.CascadeEncoder(
      encoders=[
          GemmaJaxEncoder(
              model=model,
              checkpoint_path=checkpoint_path,
              normalizer=normalizer,
              prompt=prompt,
              task_prompts=task_prompts,
              audio_sample_rate=audio_sample_rate,
              sharding=sharding,
          ),
          converter.SoundEmbeddingToTextConverter(),
          converter.TextToTextPredictionConverter(),
      ]
  )
