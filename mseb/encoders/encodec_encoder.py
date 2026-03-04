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

"""MSEB Joint Encoder for Encodec using the Transformers library.

It extracts synchronized representations from three distinct stages of
the compression pipeline: the raw acoustic projection, the continuous
quantized bottleneck, and the discrete unit indices.
"""

from __future__ import annotations

from typing import cast, Sequence

from absl import logging
from mseb import encoder
from mseb import types
import numpy as np
import torch
import transformers


class EncodecJointEncoder(encoder.MultiModalEncoder):
  """Joint Encoder producing projected, quantized, and discrete representations.

  This encoder is designed for robustness and rate-distortion analysis by
  providing three synchronized layers from a single forward pass:

  1. 'projected_latents': The raw continuous output of the convolutional encoder
     before any quantization.
  2. 'quantized_latents': The continuous vectors after being mapped to the
     nearest codebook entries via Residual Vector Quantization (RVQ).
  3. 'quantized_codes': The discrete RVQ indices stringified for symbolic
     stability metrics like Unit Edit Distance (UED).

  In standard Encodec, the convolutional encoder downsamples directly to the
  bottleneck frequency (typically 75Hz for the 24kHz model). Therefore, all
  three representations naturally share the same temporal resolution.
  """

  def __init__(
      self,
      model_path: str = "facebook/encodec_24khz",
      device: str | None = None,
  ):
    """Initializes the EncodecJointEncoder.

    Args:
      model_path: The Hugging Face model path for the Encodec model.
      device: Hardware device for inference (e.g., 'cuda', 'cpu').
          If None, auto-detects the best available device.
    """
    super().__init__()
    self.model_path = model_path
    self.model: transformers.EncodecModel | None = None
    self.processor: transformers.EncodecFeatureExtractor | None = None
    self.device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
    self.device = torch.device(self.device_str)

  def _setup(self):
    """Loads models and prepares encoders.

    Configures the Encodec model with safetensors disabled for internal
    compatibility and sets the model to evaluation mode.

    Raises:
      RuntimeError: If the model or processor cannot be loaded from the path.
    """
    self.processor = transformers.EncodecFeatureExtractor.from_pretrained(
        self.model_path
    )
    # Default to True for security and performance in open source.
    use_safe = True

    self.model = transformers.EncodecModel.from_pretrained(
        self.model_path,
        use_safetensors=use_safe
    )
    self.model.eval()
    self.model.to(self.device)
    logging.info("Encodec model loaded on %s", self.device)

  def _check_input_types(self, batch: Sequence[types.MultiModalObject]) -> None:
    """Validates that all inputs are Sound objects.

    Args:
      batch: A sequence of objects to validate.

    Raises:
      ValueError: If any object in the batch is not of type types.Sound.
    """
    if not all(isinstance(x, types.Sound) for x in batch):
      raise ValueError("EncodecJointEncoder only supports Sound inputs.")

  def _encode(
      self, batch: Sequence[types.MultiModalObject]
  ) -> Sequence[types.SoundEmbeddingCollection]:
    """Encodes a batch of sounds into synchronized multi-layer collections.

    This method performs an optimized single forward pass to extract three
    layers. It ensures perfect temporal synchronization by truncating any
    minor padding discrepancies between the continuous and quantized states.

    Args:
      batch: A sequence of Sound objects.

    Returns:
      A sequence of SoundEmbeddingCollection objects containing
      'projected_latents', 'quantized_latents', and 'quantized_codes'.

    Raises:
      RuntimeError: If the encoder is not set up (setup() was not called).
    """
    if not self.model or not self.processor:
      raise RuntimeError("Encoder is not set up. Please call .setup() first.")

    sound_batch = cast(Sequence[types.Sound], batch)
    target_sr = self.processor.sampling_rate

    # Resample all sounds to model native sample rate.
    resampled_waveforms = [
        encoder.resample_sound(s, target_sr=target_sr).waveform
        for s in sound_batch
    ]

    inputs = self.processor(
        raw_audio=resampled_waveforms,
        sampling_rate=target_sr,
        return_tensors="pt",
    ).to(self.device)

    with torch.no_grad():
      # 1. Capture Raw Encoder Output
      # Result shape: [Batch, Hidden_Dim, Seq_Len]
      projected_latents = self.model.encoder(inputs["input_values"])

      # 2. Capture Discrete Codes safely via the standard API
      model_outputs = self.model.encode(**inputs)
      codes = model_outputs.audio_codes

    # Move projected to CPU
    projected_np = projected_latents.cpu().numpy()

    output_collections = []
    for i in range(len(sound_batch)):
      original_context = sound_batch[i].context
      # PROJECTED
      p_emb_raw = projected_np[i].T
      # CODES
      # Safely extract codes for exactly this batch item regardless
      # of HF's 4D shape quirks
      if codes.ndim == 4:
        if codes.shape[0] == len(sound_batch):
          item_codes = codes[i]
        else:
          item_codes = codes[:, i]
      elif codes.ndim == 3:
        item_codes = codes[i]
      else:
        item_codes = codes

      # Flatten to strictly [Num_Codebooks, Seq_Len]
      if item_codes.ndim == 3:
        item_codes = item_codes[0]

      d_codes_raw = item_codes.cpu().numpy().T

      # QUANTIZED
      # Safely decode just this item using the official .decode() method
      # This bypasses the missing .forward() error and prevents reshaping
      # math bugs.
      # [1, Num_Codebooks, Seq_Len]
      item_codes_batch = item_codes.unsqueeze(0)
      with torch.no_grad():
        item_quantized = self.model.quantizer.decode(item_codes_batch)

      q_emb_raw = item_quantized[0].cpu().numpy().T

      logging.info(
          "Item %d raw frame lengths - Projected: %d, Quantized: %d, Codes: %d",
          i, p_emb_raw.shape[0], q_emb_raw.shape[0], d_codes_raw.shape[0]
      )

      # Ensure perfect synchronization
      min_frames = min(
          p_emb_raw.shape[0],
          q_emb_raw.shape[0],
          d_codes_raw.shape[0]
      )

      p_emb = p_emb_raw[:min_frames]
      q_emb = q_emb_raw[:min_frames]
      d_codes_raw = d_codes_raw[:min_frames]

      # Generate timestamps corresponding to the synchronized
      # bottleneck resolution.
      wav_dur = len(sound_batch[i].waveform) / target_sr
      step = wav_dur / min_frames
      starts = np.arange(min_frames) * step
      timestamps = np.stack([starts, starts + step], axis=1)

      output_collections.append(
          types.SoundEmbeddingCollection(
              embeddings={
                  "projected_latents": types.SoundEmbedding(
                      p_emb, timestamps, original_context),
                  "quantized_latents": types.SoundEmbedding(
                      q_emb, timestamps, original_context),
                  "quantized_codes": types.SoundEmbedding(
                      np.array(["_".join(map(str, row))
                                for row in d_codes_raw],
                               dtype=object),
                      timestamps, original_context
                  ),
              },
              context=original_context,
          )
      )

    return output_collections
