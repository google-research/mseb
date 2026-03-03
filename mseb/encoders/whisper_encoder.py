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

"""Whisper Encoders."""

import abc
from typing import Callable, Optional, Sequence

from absl import logging
from fvcore import nn
from mseb import encoder
from mseb import types
import numpy as np
import torch
import whisper


class Whisper(encoder.MultiModalEncoder):
  """A base class for encoding speech with Whisper model."""

  def __init__(
      self,
      model_path: str,
      device: str | None = None,
  ):
    super().__init__()
    self.model_path = model_path
    self.model = None
    self.device = device

  def _setup(self):
    """Loads the Whisper model."""
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = self.device if self.device else default_device
    logging.info(
        'Loading Whisper model from %s on device %s', self.model_path, device
    )
    self.model = whisper.load_model(self.model_path, device=device)

  def _check_input_types(self, batch: Sequence[types.MultiModalObject]) -> None:
    if not all(isinstance(x, types.Sound) for x in batch):
      raise ValueError('Whisper only supports a batch of all Sound inputs.')

  @abc.abstractmethod
  def _encode_sound(
      self,
      waveform: np.ndarray,
      params: types.SoundContextParams,
  ) -> types.SoundEmbedding:
    """Encodes speech using Whisper model.

    The embedding can be the transcription output or some activations extracted
    from Whisper model.

    Args:
      waveform: A one-dimensional NumPy array of floating-point numbers,
        representing the audio waveform. This array must be sampled at the same
        rate as whisper.audio.SAMPLE_RATE
      params: A `SoundContextParams` object containing metadata and context
        about the sound, such as its sample rate.

    Returns:
      A SoundEmbedding object.
    """
    ...

  def _encode(
      self,
      batch: Sequence[types.MultiModalObject],
  ) -> Sequence[types.SoundEmbedding]:
    """Encodes a batch of sound sources into embeddings and timestamps.

    Args:
      batch: A sequence of types.Sound objects to encode.

    Returns:
      A list of tuples, one for each input, each tuple containing:
        - waveform_embeddings (np.ndarray): A 2D array of shape
          (n, embedding_dim).
        - embedding_timestamps (np.ndarray): A 2D array of shape (m, 2),
          where each row is an [start, end] pair indicating a segment by
          sound waveform index.
          There are two common cases for the relation between embeddings (n)
          and timestamps (m):
            - Frame-Aligned (m == n): The i-th timestamp corresponds
              directly to the i-th embedding vector.
            - Utterance-Level (m == 1): A single timestamp pair represents
              the start and end of the entire audio segment from which the
              embeddings were extracted.
    """
    sound_batch: list[types.Sound] = []
    for example in batch:
      assert isinstance(example, types.Sound)
      sound_batch.append(example)
    outputs = []
    for sound in sound_batch:
      sound = encoder.resample_sound(sound, whisper.audio.SAMPLE_RATE)
      outputs.append(self._encode_sound(sound.waveform, sound.context))
    return outputs


class SpeechToTextEncoder(Whisper):
  """Represents speech with its transcription derived by Whisper model."""

  def __init__(
      self,
      model_path: str,
      device: str | None = None,
      temperature: float = 0.0,
      word_timestamps: bool = False,
  ):
    super().__init__(model_path, device=device)
    self.temperature = temperature
    self.word_timestamps = word_timestamps

  def _encode_sound(
      self,
      waveform: np.ndarray,
      params: types.SoundContextParams,
  ) -> types.SoundEmbedding:
    """Encodes speech to text using the Whisper model.

    The output structure of the embeddings and timestamps arrays
    depends on the word_timestamps parameter:

    1.  Sentence-level output (if word_timestamps is False):
        * timestamps: A numpy.ndarray of shape [n_segments, 2],
            where each inner element is
            [segment_start_time_seconds, segment_end_time_seconds].
        * embeddings: A numpy.ndarray of shape [n_segments] containing
            the transcribed text for each segment (strings).

    2.  Word-level output (if word_timestamps is True):
        * timestamps: A numpy.ndarray of shape [n_words, 2],
            where each inner element is
            [word_start_time_seconds, word_end_time_seconds].
        * embeddings: A numpy.ndarray of shape [n_words] containing
            the individual words (strings).

    Args:
      waveform: A one-dimensional NumPy array of floating-point numbers,
        representing the audio waveform. This array must be sampled at the same
        rate as whisper.audio.SAMPLE_RATE
      params: A `SoundContextParams` object containing metadata and context
        about the sound, such as its sample rate.

    Returns:
      A tuple (embeddings, timestamps).
      embeddings: A numpy.ndarray containing the transcribed text or words.
                  The shape is [n_segments] or [n_words] depending
                  on word_timestamps. Dtype will be object for strings.
      timestamps: A numpy.ndarray containing the start and end times.
                  The shape is [n_segments, 2] or [n_words, 2] depending
                  on word_timestamps. Dtype will be float.
    """
    assert self.model is not None
    recognition_result = self.model.transcribe(
        waveform.astype(np.float32),
        language=params.language.split('_')[0] if params.language else None,
        temperature=self.temperature,
        word_timestamps=self.word_timestamps,
    )
    if self.word_timestamps:
      timestamps_list = []
      embeddings_list = []
      for segment in recognition_result['segments']:
        for word in segment['words']:
          timestamps_list.append([word['start'], word['end']])
          embeddings_list.append(word['word'])
      timestamps = np.array(timestamps_list, dtype=float)
      embeddings = np.array(embeddings_list, dtype=object)
    else:
      timestamp_start, timestamp_end = 0, 0
      texts = []
      for i, segment in enumerate(recognition_result['segments']):
        if i == 0:
          timestamp_start = segment['start']
        timestamp_end = segment['end']
        texts.append(segment['text'])
      timestamps = np.array([[timestamp_start, timestamp_end]], dtype=float)
      embeddings = np.array([''.join(texts)], dtype=object)
    return types.SoundEmbedding(
        embedding=embeddings, timestamps=timestamps, context=params
    )


class ForcedAlignmentEncoder(Whisper):
  """Embeds by forced-alignment of speech and text by Whisper model."""

  def __init__(
      self,
      model_path: str,
      device: str | None = None,
      language: Optional[str] = None,
  ):
    """Initializes the Whisper encoder.

    Args:
      model_path: The path to the Whisper model.
      device: The device to use for the Whisper model.
      language: Whisper language.
    """
    super().__init__(model_path, device=device)
    self.language = language

  def _setup(self):
    """Loads the Whisper model and the tokenizer."""
    super()._setup()
    self.tokenizer = whisper.tokenizer.get_tokenizer(
        self.model.is_multilingual,
        num_languages=self.model.num_languages,
        language=self.language,
        task='transcript',
    )

  def _encode_sound(
      self,
      waveform: np.ndarray,
      params: types.SoundContextParams,
  ) -> types.SoundEmbedding:
    """Encodes speech and text using forced alignment with the Whisper model.

    Args:
      waveform: A one-dimensional NumPy array of floating-point numbers,
        representing the audio waveform. This array must be sampled at
        whisper.audio.SAMPLE_RATE.
      params: A `SoundContextParams` object containing metadata and context
        about the sound, such as its sample rate.

    Returns:
      A tuple (words, timestamps).
      words: A NumPy array containing the transcribed words corresponding to
             the timestamps. Shape is [n_words].
      timestamps: A NumPy array containing the start and end times for each
                  word. Shape is [n_words, 2], where each row is
                  [start_time, end_time].

    Raise:
      ValueError: If params does not include text or tokenizer can not
                  tokenize text.
    """
    assert self.model is not None
    mel = whisper.audio.log_mel_spectrogram(
        waveform, self.model.dims.n_mels, padding=whisper.audio.N_SAMPLES
    )
    num_frames = mel.shape[-1] - whisper.audio.N_FRAMES
    mel = whisper.audio.pad_or_trim(mel, whisper.audio.N_FRAMES)
    if not params.text:
      logging.warning('Context text is empty. No alignment will be performed.')
      return types.SoundEmbedding(
          embedding=np.empty((0), dtype=object),
          timestamps=np.empty((0, 2), dtype=float),
          context=params,
      )
    tokens = self.tokenizer.encode(params.text)
    if not tokens:
      logging.warning(
          'No tokens generated from context text. Ensure text is valid.'
      )
      return types.SoundEmbedding(
          embedding=np.empty((0), dtype=object),
          timestamps=np.empty((0, 2), dtype=float),
          context=params,
      )
    alignment = whisper.timing.find_alignment(
        self.model, self.tokenizer, tokens, mel, num_frames
    )
    n_words = len(alignment)
    timestamps = np.empty((n_words, 2), dtype=float)
    words = np.empty((n_words), dtype=object)
    for i, word_timing in enumerate(alignment):
      timestamps[i, :] = [word_timing.start, word_timing.end]
      words[i] = word_timing.word

    return types.SoundEmbedding(
        embedding=words, timestamps=timestamps, context=params
    )


class PooledAudioEncoder(Whisper):
  """Embeds by pooling Whisper audio encoder activations."""

  def __init__(
      self,
      model_path: str,
      device: str | None = None,
      pooling: str | None = None,
  ):
    """Initializes the Whisper encoder.

    Args:
      model_path: The path to the Whisper model.
      device: The device to use for the Whisper model;
      pooling: The type of pooling to apply to the encoder activations.
        Supported options: 'last', 'mean', 'max'. Defaults to None.
    """
    super().__init__(model_path, device=device)
    self._flops_cache = None
    self.pool_fn: Callable[[np.ndarray], np.ndarray]
    if pooling is None:
      self.pool_fn = lambda x: x
    elif pooling == 'last':
      self.pool_fn = lambda x: x[-1][None, :]
    elif pooling == 'mean':
      self.pool_fn = lambda x: np.mean(x, axis=0, keepdims=True)
    elif pooling == 'max':
      self.pool_fn = lambda x: np.max(x, axis=0, keepdims=True)
    else:
      raise ValueError(
          f'Unsupported pooling type: {pooling}. '
          'Expected one of last, mean, max.'
      )
    # In whisper.audio: N_SAMPLES_PER_TOKEN = 2 * HOP_LENGTH
    self.encoder_stride = 2

  def _encode_sound(
      self,
      waveform: np.ndarray,
      params: types.SoundContextParams,
  ) -> types.SoundEmbedding:
    """Encodes speech into a pooled embedding of Whisper encoder activations.

    Args:
      waveform: A one-dimensional NumPy array of floating-point numbers,
        representing the audio waveform. This array must be sampled at
        whisper.audio.SAMPLE_RATE.
      params: Encoder input context parameters. This parameter is part of the
        abstract _encode interface defined in the parent class Whisper, but it
        is not directly utilized by this encoder.

    Returns:
      A tuple (embedding, timestamp).
      embedding: A NumPy array of shape (1, D) if pooling is not None and (n, D)
                 if pooling is None. Here D is the dimension of the Whisper
                 model's audio encoder output.
      timestamp: A NumPy array with a single row [start, end], representing
                 the start and end times of the input audio. start is always 0,
                 and end is the total duration of the waveform in seconds.
                 Shape will be (1, 2).
    """
    assert self.model is not None
    mel, num_frames = self.get_mel_inputs(waveform)
    with torch.no_grad():
      embeddings = self.model.embed_audio(mel)
    embeddings = embeddings.to('cpu').detach().numpy().squeeze(0)
    num_embeddings = num_frames // self.encoder_stride
    audio_duration_seconds = len(waveform) / whisper.audio.SAMPLE_RATE
    timestamp = np.array([[0, audio_duration_seconds]])
    return types.SoundEmbedding(
        embedding=self.pool_fn(embeddings[:num_embeddings, :]),
        timestamps=timestamp,
        context=params,
    )

  def get_mel_inputs(self, waveform: np.ndarray):
    assert self.model is not None
    mel = whisper.audio.log_mel_spectrogram(
        waveform, self.model.dims.n_mels, padding=whisper.audio.N_SAMPLES
    )
    num_frames = mel.shape[-1] - whisper.audio.N_FRAMES
    mel = whisper.audio.pad_or_trim(mel, whisper.audio.N_FRAMES)
    mel = mel[None, :, :].to(self.model.device)
    return mel, num_frames

  def get_encode_flops(self, data: types.MultiModalObject):
    if self._flops_cache is not None:
      return self._flops_cache
    assert isinstance(data, types.Sound)
    sound = encoder.resample_sound(data, whisper.audio.SAMPLE_RATE)
    mel, _ = self.get_mel_inputs(sound.waveform)
    flop_analyzer = nn.FlopCountAnalysis(self.model.encoder, mel)
    self._flops_cache = flop_analyzer.total()
    return self._flops_cache


class WhisperJointEncoder(encoder.MultiModalEncoder):
  """Extracts acoustic hidden states and transcripts from Whisper.

  This encoder leverages the robust Mel-preprocessing logic from
  PooledAudioEncoder to ensure compatibility with Whisper's 30-second
  positional embeddings while maintaining accurate temporal alignment.
  """

  def __init__(self, model_path: str, device: str | None = None):
    """Initializes the Joint Encoder.

    Args:
      model_path: The Whisper model size or path (e.g., 'base', 'large-v3').
      device: Hardware device for inference (e.g., 'cuda', 'cpu').
    """
    super().__init__()
    self.model_path = model_path
    self.device = device
    self.model = None
    self.encoder_stride = 2
    self.frame_shift_seconds = 0.01

  def _setup(self):
    """Loads the Whisper model into memory."""
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = self.device if self.device else default_device
    self.model = whisper.load_model(self.model_path, device=device)

  def _check_input_types(self, batch: Sequence[types.MultiModalObject]) -> None:
    """Validates that all inputs are Sound objects."""
    if not all(isinstance(x, types.Sound) for x in batch):
      raise ValueError('WhisperJointEncoder only supports Sound inputs.')

  def get_mel_inputs(self, waveform: np.ndarray):
    """Prepares Mel spectrograms padded exactly to 30 seconds.

    Args:
      waveform: 1D NumPy array (16kHz).

    Returns:
      mel: Torch tensor of shape (1, 80, 3000) on the target device.
      num_frames: The number of mel frames in the actual signal before padding.
    """
    assert self.model is not None
    # padding=N_SAMPLES adds 30s of zeros to the waveform
    mel = whisper.audio.log_mel_spectrogram(
        waveform, self.model.dims.n_mels, padding=whisper.audio.N_SAMPLES
    )
    # Subtracting N_FRAMES (3000) recovers the unpadded frame count
    num_frames = mel.shape[-1] - whisper.audio.N_FRAMES
    # Ensure shape is exactly 3000 to match positional embeddings
    mel_fixed = whisper.audio.pad_or_trim(mel, whisper.audio.N_FRAMES)
    mel_fixed = mel_fixed[None, :, :].to(self.model.device)
    return mel_fixed, num_frames

  def _encode(
      self,
      batch: Sequence[types.MultiModalObject]
  ) -> Sequence[types.SoundEmbeddingCollection]:
    """Encodes a batch of sounds."""
    outputs = []
    for example in batch:
      sound = encoder.resample_sound(example, whisper.audio.SAMPLE_RATE)
      outputs.append(self._encode_single_sound(sound.waveform, sound.context))
    return outputs

  def _encode_single_sound(
      self,
      waveform: np.ndarray,
      params: types.SoundContextParams
  ) -> types.SoundEmbeddingCollection:
    """Extracts hidden states and transcripts for one waveform."""
    assert self.model is not None

    # 1. Acoustic Path: Encoder Hidden States
    # Use the robust helper to avoid "incorrect audio shape" AssertionError
    mel, num_frames = self.get_mel_inputs(waveform)

    with torch.no_grad():
      hiddens = self.model.embed_audio(mel)

    hiddens_np = hiddens.to('cpu').detach().numpy().squeeze(0)
    num_steps = num_frames // self.encoder_stride

    # Accurate 20ms step timestamps
    step_dur = self.frame_shift_seconds * self.encoder_stride
    starts = np.arange(num_steps) * step_dur
    acoustic_timestamps = np.stack([starts, starts + step_dur], axis=1)

    # 2. Semantic Path: Transcription
    result = self.model.transcribe(
        waveform.astype(np.float32),
        word_timestamps=True,
        language=params.language.split('_')[0] if params.language else None
    )

    # 3. Packaging
    acoustic_emb = types.SoundEmbedding(
        embedding=hiddens_np[:num_steps, :],
        timestamps=acoustic_timestamps,
        context=params
    )

    words = []
    word_ts = []
    for segment in result['segments']:
      for w in segment['words']:
        words.append(w['word'])
        word_ts.append([w['start'], w['end']])

    semantic_emb = types.SoundEmbedding(
        embedding=np.array(words, dtype=object),
        timestamps=np.array(word_ts, dtype=float),
        context=params
    )

    return types.SoundEmbeddingCollection(
        embeddings={
            'encoder_hiddens': acoustic_emb,
            'transcripts': semantic_emb
        },
        context=params
    )
