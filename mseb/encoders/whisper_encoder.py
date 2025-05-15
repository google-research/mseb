# Copyright 2024 The MSEB Authors.
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
from typing import Any, Union, Optional, Sequence, Tuple

from absl import logging
import librosa
from mseb import encoder
import numpy as np
import whisper


class Whisper(encoder.Encoder):
  """A base class for encoding speech with Whisper model."""

  def __init__(self, model: whisper.Whisper):
    """Initializes the Whisper encoder.

    Args:
      model: An instance of whisper model.
    """
    self.model = model

  def preprocess(self,
                 sequence: Union[str, np.ndarray],
                 sample_rate: Optional[int] = None) -> np.ndarray:
    """Preprocesses sequence for Whisper models' input compatibility.

    Args:
      sequence: The input audio. Can be a string (path to an audio file)
                or a sequence of floating-point numbers (raw audio waveform).
      sample_rate: The sample rate associated with the sequence.
                   Required if sequence is a float sequence.
                   Must be None if sequence is a string filepath.

    Returns:
      Waveform data sampled at whisper.audio.SAMPLE_RATE hertz.

    Raises:
      ValueError: If sample_rate is inconsistent with ssequence tpye,
                  or if sequence is neither a string or Sequence[float].
    """
    if isinstance(sequence, str):
      if sample_rate is not None:
        raise ValueError(
            'The sample_rate argument should be set to None when '
            'sequence is an audio filepath.'
        )
      sequence_data, sample_rate = librosa.load(sequence, sr=None)
    elif isinstance(sequence, np.ndarray):
      if sample_rate is None:
        raise ValueError(
            'The sample_rate argument must not be None when '
            'sequence input is Sequence[float].'
        )
      if not np.issubdtype(sequence.dtype, np.floating):
        raise ValueError(
            'The input array is a NumPy ndarray but its data tpye is '
            f'{sequence.dtype} whereas a floating-point data type is required.'
        )
      if sample_rate == whisper.audio.SAMPLE_RATE:
        return sequence
      sequence_data = sequence
    else:
      raise ValueError(
          'Input sequence must be a string or a NumPy ndarray, but got '
          f'not {type(sequence).__name__}'
      )

    return librosa.resample(
        sequence_data,
        orig_sr=sample_rate,
        target_sr=whisper.audio.SAMPLE_RATE)

  @abc.abstractmethod
  def _encode(self,
              waveform: np.ndarray,
              context: encoder.ContextParams,
              **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Encodes speech using Whisper model.

    The embedding can be the transcription output or some activations extracted
    from Whisper model.

    Args:
      waveform: A one-dimensional NumPy array of floating-point numbers,
                representing the audio waveform. This array must be sampled at
                the same rate as whisper.audio.SAMPLE_RATE
      context: Encoder input context parameters.
      **kwargs: Additional arguments to pass to the encoder.

    Returns:
      A tuple (timestamps, embeddings).
      timestamps: A numpy.ndarray of shape [n_segments, 2], where n_segments
                  is the number of identified segments. Each inner element is
                  [segment_start_time_seconds, segment_end_time_seconds].
                  If a single embedding represents the entire audio sequence,
                  timestamps will contain one element corresponding
                  to the start and end times of the whole input audio.
      embeddings: A numpy.ndarray representing an embedding for each
                  corresponding timestamp/segment. Its shape will depend on
                  the type of embedding and the number of segments (e.g.,
                  [n_segments, embedding_dimension]).
    """

  def encode(self,
             sequence: Union[str, Sequence[float]],
             context: encoder.ContextParams,
             **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    sequence_data = self.preprocess(sequence, context.sample_rate)
    return self._encode(sequence_data, context, **kwargs)


class SpeechToTextEncoder(Whisper):
  """Represents speech with its transcription derived by Whisper model."""

  def _encode(self,
              waveform: np.ndarray,
              context: encoder.ContextParams,
              word_timestamps: Optional[bool] = False,
              temperature: Optional[float] = 0.0
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Encodes speech to text using the Whisper model.

    The output structure of the timestamps and embeddings arrays
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
                representing the audio waveform. This array must be sampled at
                the same rate as whisper.audio.SAMPLE_RATE
      context: Encoder input context parameters.
      word_timestamps: Whether to output word-level timing and text. If `False`,
                       segment-level timing and text are returned.
      temperature: The sampling temperature for transcription.

    Returns:
      A tuple (timestamps, embeddings).
      timestamps: A numpy.ndarray containing the start and end times.
                  The shape is [n_segments, 2] or [n_words, 2] depending
                  on word_timestamps.
      embeddings: A numpy.ndarray containing the transcribed text or words.
                  The shape is [n_segments] or [n_words] depending
                  on word_timestamps. Dtype will be object for strings.
    """
    recognition_result = self.model.transcribe(
        waveform,
        language=context.language,
        temperature=temperature,
        word_timestamps=word_timestamps
    )
    if word_timestamps:
      timestamps_list = []
      embeddings_list = []
      for segment in recognition_result['segments']:
        for word in segment['words']:
          timestamps_list.append([word['start'], word['end']])
          embeddings_list.append(word['word'])
      timestamps = np.array(timestamps_list, dtype=float)
      embeddings = np.array(embeddings_list, dtype=object)
    else:
      n_segments = len(recognition_result['segments'])
      timestamps = np.empty((n_segments, 2), dtype=float)
      embeddings = np.empty((n_segments), dtype=object)
      for i, segment in enumerate(recognition_result['segments']):
        timestamps[i, :] = [segment['start'], segment['end']]
        embeddings[i] = segment['text']
    return timestamps, embeddings


class ForcedAlignmentEncoder(Whisper):
  """Embeds by forced-alignment of speech and text by Whisper model."""

  def __init__(self,
               model: whisper.Whisper,
               language: Optional[str] = None):
    """Initializes the Whisper encoder.

    Args:
      model: An instance of whisper model.
      language: Whisper language.
    """
    super().__init__(model)
    self.tokenizer = whisper.tokenizer.get_tokenizer(
        self.model.is_multilingual,
        num_languages=self.model.num_languages,
        language=language,
        task='transcript')

  def _encode(self,
              waveform: np.ndarray,
              context: encoder.ContextParams,
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Encodes speech and text using forced alignment with the Whisper model.

    Args:
      waveform: A one-dimensional NumPy array of floating-point numbers,
                representing the audio waveform. This array must be sampled
                at whisper.audio.SAMPLE_RATE.
      context: Encoder input context parameters, expected to have a text
               attribute containing the transcription to align.

    Returns:
      A tuple (timestamps, words).
      timestamps: A NumPy array containing the start and end times for each
                  word. Shape is [n_words, 2], where each row is
                  [start_time, end_time].
      words: A NumPy array containing the transcribed words corresponding to
             the timestamps. Shape is [n_words].

    Raise:
      ValueError: If context does not include text or tokenizer can not
                  tokenize text.
    """
    mel = whisper.audio.log_mel_spectrogram(
        waveform, self.model.dims.n_mels, padding=whisper.audio.N_SAMPLES
    )
    num_frames = mel.shape[-1] - whisper.audio.N_FRAMES
    mel = whisper.audio.pad_or_trim(mel, whisper.audio.N_FRAMES)
    if not context.text:
      logging.warning(
          'Context text is empty. No alignment will be performed.'
      )
      return np.empty((0, 2), dtype=float), np.empty((0), dtype=object)
    tokens = self.tokenizer.encode(context.text)
    if not tokens:
      logging.warning(
          'No tokens generated from context text. Ensure text is valid.'
      )
      return np.empty((0, 2), dtype=float), np.empty((0), dtype=object)
    alignment = whisper.timing.find_alignment(
        self.model, self.tokenizer, tokens, mel, num_frames)
    n_words = len(alignment)
    timestamps = np.empty((n_words, 2), dtype=float)
    words = np.empty((n_words), dtype=object)
    for i, word_timing in enumerate(alignment):
      timestamps[i, :] = [word_timing.start, word_timing.end]
      words[i] = word_timing.word

    return timestamps, words
