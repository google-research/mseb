# Copyright 2025 The MSEB Authors.
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

"""MSEB Encoder for the CLAP model, supporting both audio and text."""

from __future__ import annotations

from typing import cast, Optional, Sequence

from mseb import encoder
from mseb import types
import numpy as np
import torch
import transformers


class _CLAPAudioEncoder(encoder.MultiModalEncoder):
  """Internal class to encode audio using the CLAP model's audio tower."""

  def __init__(self, model_path: str, device: Optional[str] = None):
    super().__init__()
    self.model_path = model_path
    self.model: transformers.ClapModel | None = None
    self.processor: transformers.ClapProcessor | None = None
    self.device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

  def _setup(self):
    self.processor = transformers.ClapProcessor.from_pretrained(self.model_path)
    self.model = transformers.ClapModel.from_pretrained(self.model_path)
    self.model.eval()
    self.model.to(self.device)
    print(f"CLAP audio encoder loaded on device: {self.device}")

  def _check_input_types(self,
                         batch: Sequence[types.MultiModalObject]
                         ) -> None:
    if not all(isinstance(x, types.Sound) for x in batch):
      raise ValueError(
          "_CLAPAudioEncoder only supports a batch of Sound inputs."
      )

  def _encode(self,
              batch: Sequence[types.MultiModalObject]
              ) -> Sequence[types.SoundEmbedding]:
    if not self.model or not self.processor:
      raise RuntimeError(
          "Encoder is not set up. Please call .setup() first."
      )

    sound_batch = cast(Sequence[types.Sound], batch)
    target_sr = self.processor.feature_extractor.sampling_rate
    resampled_sound_batch = [
        encoder.resample_sound(sound_item, target_sr=target_sr)
        for sound_item in sound_batch
    ]
    waveforms = [item.waveform for item in resampled_sound_batch]

    inputs = self.processor(
        audios=waveforms,
        sampling_rate=target_sr,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(self.device) for k, v in inputs.items()}

    with torch.no_grad():
      audio_embeds = self.model.get_audio_features(**inputs)
      audio_embeds = audio_embeds.to("cpu").numpy()

    output_embeddings = []
    for i, sound_item in enumerate(sound_batch):
      embedding = audio_embeds[i][np.newaxis, :]
      end_time = sound_item.context.length / sound_item.context.sample_rate
      timestamps = np.array([[0.0, end_time]])
      output_embeddings.append(
          types.SoundEmbedding(
              embedding=embedding,
              timestamps=timestamps,
              context=sound_item.context
          )
      )
    return output_embeddings


class _CLAPTextEncoder(encoder.MultiModalEncoder):
  """Internal class to encode text using the CLAP model's text tower."""

  def __init__(self, model_path: str, device: Optional[str] = None):
    super().__init__()
    self.model_path = model_path
    self.model: transformers.ClapModel | None = None
    self.processor: transformers.ClapProcessor | None = None
    self.device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

  def _setup(self):
    self.processor = transformers.ClapProcessor.from_pretrained(self.model_path)
    self.model = transformers.ClapModel.from_pretrained(self.model_path)
    self.model.eval()
    self.model.to(self.device)
    print(f"CLAP text encoder loaded on device: {self.device}")

  def _check_input_types(self,
                         batch: Sequence[types.MultiModalObject]
                         ) -> None:
    if not all(isinstance(x, types.Text) for x in batch):
      raise ValueError(
          "_CLAPTextEncoder only supports a batch of Text inputs."
      )

  def _encode(self,
              batch: Sequence[types.MultiModalObject]
              ) -> Sequence[types.TextEmbedding]:
    if not self.model or not self.processor:
      raise RuntimeError(
          "Encoder is not set up. Please call .setup() first."
      )

    text_batch = cast(Sequence[types.Text], batch)
    texts = [item.text for item in text_batch]
    inputs = self.processor(
        text=texts,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(self.device) for k, v in inputs.items()}

    with torch.no_grad():
      text_embeds = self.model.get_text_features(**inputs)
      text_embeds = text_embeds.to("cpu").numpy()

    output_embeddings = []
    for i, text_item in enumerate(text_batch):
      embedding = text_embeds[i][np.newaxis, :]
      spans = np.array([[0, len(text_item.text)]])
      output_embeddings.append(
          types.TextEmbedding(
              embedding=embedding,
              spans=spans,
              context=text_item.context
          )
      )
    return output_embeddings


def ClapEncoder(
    model_path: str = "laion/clap-htsat-unfused",
    device: Optional[str] = None
) -> encoder.CollectionEncoder:
  """Factory function to create a fully configured multi-modal CLAP encoder.

  This function builds a CollectionEncoder that contains both the audio and text
  towers of the CLAP model, dispatched based on input type.

  Args:
    model_path: The Hugging Face model path.
    device: The device to load the models onto ("cuda", "cpu", or None).

  Returns:
    An initialized CollectionEncoder ready for setup and use.
  """
  return encoder.CollectionEncoder({
      types.Sound: _CLAPAudioEncoder(model_path=model_path, device=device),
      types.Text: _CLAPTextEncoder(model_path=model_path, device=device),
  })
