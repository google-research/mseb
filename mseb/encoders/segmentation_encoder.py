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

"""Segmentation encoder."""

from collections.abc import Sequence
from typing import Any, Iterator, Tuple

from mseb import encoder
from mseb.encoders import whisper_encoder
import numpy as np
import pygtrie
import spacy


class RetokenizerBase:
  """Base class for retokenizers."""

  def retokenize(self, words: Sequence[str]) -> Iterator[Tuple[str, int, int]]:
    """Retokenizes an input text defined by a sequence of words.

    Args:
      words: A sequence of words.

    Yields:
      A sequence of tuples, each corresponding to a token and consisting of
          (token_text, front, back)
      where [front, back] is the closed interval of indices defining the
      smallest span of words that covers the token.
    """
    raise NotImplementedError()


class SpacyRetokenizer(RetokenizerBase):
  """Retokenize using spacy tokenizer."""

  def __init__(self, language: str = 'xx'):
    self.model = spacy.blank(language)

  def _tokenize(self, text: str) -> Iterator[Tuple[str, int, int]]:
    for token in self.model(text):
      print(token.text, token.idx, token.idx + len(token.text) - 1)
      yield token.text, token.idx, token.idx + len(token.text) - 1

  def retokenize(self, words: Sequence[str]) -> Iterator[Tuple[str, int, int]]:
    pos2word = []
    for index, word in enumerate(words):
      for _ in range(len(word)):
        pos2word.append(index)
    for token in self._tokenize(''.join(words)):
      yield token[0].lower(), pos2word[token[1]], pos2word[token[2]]


class NormalizingRetokenizer(RetokenizerBase):
  """Represents a retokenizer that applies some normalization to input tokens."""

  def _normalize(self, token: str) -> str:
    return (
        token.removeprefix(' ')
        .removesuffix('?')
        .removesuffix('؟')
        .removeprefix('«')
        .removesuffix('»')
        .removesuffix('».')
        .removesuffix('।')
        .removesuffix('.')
        .removesuffix(',')
    )

  def retokenize(self, words: Sequence[str]) -> Iterator[Tuple[str, int, int]]:
    for index, word in enumerate(words):
      normalized_word = self._normalize(word)
      if normalized_word:
        yield normalized_word.lower(), index, index


class SegmenterBase:
  """Base class for segmenters."""

  def segment(
      self, words: Sequence[str]
  ) -> Iterator[Tuple[str, float, int, int]]:
    """Segments an input text defined by a sequence of words.

    Args:
      words: A sequence of words.

    Yields:
      A sequence of tuples, each corresponding to a segment and consisting of
          (segment_text, segment_score, front, back)
      where [front, back] is the closed interval of indices defining the
      smallest span of words that covers the
      segment.
    """
    raise NotImplementedError()


class TokenIDFSegmenter(SegmenterBase):
  """Implements a token-based IDF segmenter, looking up IDF terms after re-tokenization."""

  def __init__(self, idf_table: dict[str, float], retokenizer: RetokenizerBase):
    self.idf_table = idf_table
    self.retokenizer = retokenizer

  def segment(
      self, words: Sequence[str]
  ) -> Iterator[Tuple[str, float, int, int]]:
    for word, front, back in self.retokenizer.retokenize(words):
      if word in self.idf_table:
        yield word, self.idf_table[word], front, back


class LongestPrefixIDFSegmenter(SegmenterBase):
  """Represents an IDF term extractor using longest prefix matching."""

  def __init__(self, idf_table: dict[str, float]):
    self.trie = pygtrie.Trie(idf_table)

  def _segment_text(self, text: str) -> Iterator[Tuple[str, float, int]]:
    for pos in range(len(text)):
      idf_term = self.trie.longest_prefix(text[pos:])
      if idf_term:
        yield ''.join(idf_term[0]), idf_term[1], pos

  def segment(self, words: Sequence[str]):
    pos2word = []
    for index, word in enumerate(words):
      for _ in range(len(word)):
        pos2word.append(index)

    for term, value, pos in self._segment_text(''.join(words).lower()):
      yield term, value, pos2word[pos], pos2word[
          pos + len(term) - 1
      ]


class CascadedSegmentationEncoder(encoder.Encoder):
  """TODO(allauzen).

  Represents speech as as segments derived by Whisper model.
  """

  def __init__(self,
               asr_encoder: whisper_encoder.Whisper,
               segmenter: SegmenterBase):
    self.asr_encoder = asr_encoder
    self.segmenter = segmenter

  def encode(
      self,
      waveform: np.ndarray,
      context: encoder.ContextParams,
      **kwargs: Any
  ) -> Tuple[np.ndarray, np.ndarray]:
    """TODO(allauzen): update: Encodes speech into segments."""
    timestamps, words = self.asr_encoder.encode(
        waveform, context, **kwargs
    )
    segments = list(self.segmenter.segment(words))
    if not segments:
      return np.array([[0.0, 0.0]]), np.array([['', 0.0]])
    # TODO(allauzen): should we return all segments? And look for the best in a
    # derived class?
    term, score, front, back = max(segments, key=lambda x: x[1])
    return np.array([[timestamps[front][0], timestamps[back][1]]]), np.array(
        [[term, score]])


class MaxIDFSegmentEncoder(CascadedSegmentationEncoder):
  """TODO(allauzen)."""

  def __init__(self,
               asr_encoder: whisper_encoder.Whisper,
               idf_table: dict[str, float],
               language: str):
    """TODO(allauzen)."""
    if language.startswith('ja'):
      segmenter = LongestPrefixIDFSegmenter(idf_table)
    else:
      segmenter = TokenIDFSegmenter(
          idf_table,
          SpacyRetokenizer(
              language=('xx' if language.startswith('ko') else language)
          ),
      )
    super().__init__(asr_encoder, segmenter)
