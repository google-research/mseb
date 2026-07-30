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

"""Custom decoding task for Whisper, returning hypotheses and other results."""

import dataclasses
from typing import Any

from mseb import types
from mseb.encoders import whisper_encoder
import numpy as np
import torch
import whisper


@dataclasses.dataclass(slots=True)
class HypothesesDecodingResult:
  """Container for hypotheses decoding task results.

  Attributes:
    tokens: The token sequences for each hypothesis.
    sum_logprobs: Sum of log probabilities for each hypothesis.
    no_speech_probs: No-speech probability for each audio sample.
    hypotheses: Decoded text hypotheses as strings for each audio sample.
  """

  tokens: Any
  sum_logprobs: Any
  no_speech_probs: Any
  hypotheses: list[list[str]]

  def __post_init__(self):
    if len(self.hypotheses) != len(self.sum_logprobs):
      raise ValueError(
          "Length of hypotheses and sum_logprobs must match, got"
          f" {len(self.hypotheses)} vs {len(self.sum_logprobs)}."
      )
    for i, (hyp_list, logprob_list) in enumerate(
        zip(self.hypotheses, self.sum_logprobs)
    ):
      if len(hyp_list) != len(logprob_list):
        raise ValueError(
            f"Sample {i}: number of hypotheses ({len(hyp_list)}) does not match"
            f" number of logprobs ({len(logprob_list)})."
        )


class HypothesesDecodingTask(whisper.decoding.DecodingTask):
  """A custom decoding task for Whisper.

  Returns the hypotheses as strings along with other decoding results.
  """

  @torch.no_grad()
  def run_hypotheses(self, mel: torch.Tensor) -> HypothesesDecodingResult:
    """Runs the decoding task.

    Args:
      mel: The Mel spectrogram of the audio.

    Returns:
      A HypothesesDecodingResult containing hypotheses as strings along with
      other decoding results.
    """
    self.decoder.reset()
    tokenizer: whisper.tokenizer.Tokenizer = self.tokenizer
    n_audio: int = mel.shape[0]

    audio_features: torch.Tensor = self._get_audio_features(
        mel
    )  # encoder forward pass
    tokens: torch.Tensor = torch.tensor([self.initial_tokens]).repeat(
        n_audio, 1
    )

    assert (
        self.options.task == "transcribe" or self.options.task == "translate"
    )
    self._detect_language(audio_features, tokens)

    # repeat text tensors by the group size, for beam search or best-of-n
    # sampling
    tokens = tokens.repeat_interleave(self.n_group, dim=0).to(
        audio_features.device
    )

    # call the main sampling loop
    tokens, sum_logprobs, no_speech_probs = self._main_loop(
        audio_features, tokens
    )

    # reshape the tensors to have (n_audio,  n_group) as the first two
    # dimensions
    audio_features = audio_features[:: self.n_group]
    no_speech_probs = no_speech_probs[:: self.n_group]
    assert audio_features.shape[0] == len(no_speech_probs) == n_audio

    tokens = tokens.reshape(n_audio, self.n_group, -1)
    sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

    # get the final candidates for each group, and slice between the first
    # sampled token and EOT
    tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
    tokens: list[list[torch.Tensor]] = [
        [t[self.sample_begin : (t == tokenizer.eot).nonzero()[0, 0]] for t in s]  # pytype: disable=attribute-error
        for s in tokens
    ]
    # Remove special tokens from the text tokens.
    text_tokens = [
        [[u for u in t if u < tokenizer.eot] for t in s] for s in tokens
    ]
    return HypothesesDecodingResult(
        tokens=tokens,
        sum_logprobs=sum_logprobs,
        no_speech_probs=no_speech_probs,
        hypotheses=[
            [tokenizer.decode(t).strip() for t in s] for s in text_tokens
        ],
    )


class WhisperHypothesesEncoder(whisper_encoder.Whisper):
  """Encodes speech into a collection of Whisper hypotheses."""

  def __init__(
      self,
      model_path: str,
      num_hypotheses: int,
      device: str | None = None,
      temperature: float = 0.0,
      word_timestamps: bool = False,
      task: str = "transcribe",
  ):
    super().__init__(model_path, device=device)
    self.num_hypotheses = num_hypotheses
    self.temperature = temperature
    self.word_timestamps = word_timestamps
    self._tokenizer = None
    self.task = task

  def _setup(self):
    super()._setup()
    if self.word_timestamps:
      assert self.model is not None, "Model is not loaded."
      self._tokenizer = whisper.tokenizer.get_tokenizer(
          self.model.is_multilingual,
          num_languages=self.model.num_languages,
          language=None,
          task=self.task,
      )

  def _encode_sound(
      self,
      waveform: np.ndarray,
      params: types.SoundContextParams,
  ) -> types.SoundEmbeddingCollection:
    """Encodes speech into a collection of Whisper hypotheses."""
    assert self.model is not None

    mel = whisper.audio.log_mel_spectrogram(
        waveform,
        self.model.dims.n_mels,
        padding=whisper.audio.N_SAMPLES,
        device=self.model.device,
    )
    num_frames = mel.shape[-1] - whisper.audio.N_FRAMES
    mel = whisper.audio.pad_or_trim(mel, whisper.audio.N_FRAMES)
    mel = mel.unsqueeze(0)

    task = HypothesesDecodingTask(
        self.model,
        whisper.decoding.DecodingOptions(
            temperature=self.temperature,
            best_of=self.num_hypotheses if self.temperature > 0 else None,
            beam_size=self.num_hypotheses if self.temperature == 0 else None,
            task=self.task,
        ),
    )
    results = task.run_hypotheses(mel)

    hyp_to_score = {}
    for i in range(len(results.hypotheses[0])):
      hyp = results.hypotheses[0][i]
      score = results.sum_logprobs[0][i] if self.temperature == 0 else 0.0
      if hyp not in hyp_to_score:
        hyp_to_score[hyp] = score
      else:
        hyp_to_score[hyp] = np.logaddexp(hyp_to_score[hyp], score)

    embeddings = dict()
    if not self.word_timestamps:
      for hyp, score in hyp_to_score.items():
        embeddings[f"hypothesis_{len(embeddings)}"] = types.SoundEmbedding(
            embedding=np.array([hyp], dtype=object),
            timestamps=np.array(
                [[params.waveform_start_second, params.waveform_end_second]],
                dtype=float,
            ),
            scores=np.array([score], dtype=float),
            context=params,
        )
    else:
      # TODO(allauzen): Investigate how to compute timestamps without performing
      # forced alignment.
      assert self.task == "transcribe", (
          "Forced alignment is only supported for transcription tasks."
      )
      assert self._tokenizer is not None
      mel = mel.squeeze(0)
      for hyp, score in hyp_to_score.items():
        tokens = self._tokenizer.encode(hyp)
        alignment = whisper.timing.find_alignment(
            self.model, self._tokenizer, tokens, mel, num_frames
        )
        n_words = len(alignment)
        timestamps = np.empty((n_words, 2), dtype=float)
        words = np.empty((n_words), dtype=object)
        for j, word_timing in enumerate(alignment):
          timestamps[j, :] = [word_timing.start, word_timing.end]
          words[j] = word_timing.word
        embeddings[f"hypothesis_{len(embeddings)}"] = types.SoundEmbedding(
            embedding=words,
            timestamps=timestamps,
            scores=np.array([score], dtype=float),
            context=params,
        )

    return types.SoundEmbeddingCollection(
        embeddings=embeddings,
        context=params,
    )
