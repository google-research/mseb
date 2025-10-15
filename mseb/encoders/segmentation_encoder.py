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
from typing import Iterator, Optional

from mseb import encoder
from mseb import types
from mseb.encoders import whisper_encoder
import numpy as np
import pandas as pd
import pygtrie
import spacy


def load_idf_table_from_csv(path: str) -> dict[str, float]:
  """Reads a CSV file into a dictionary mapping 'token' to 'idf'."""
  df = pd.read_csv(path, dtype={'token': str})
  if 'token' not in df.columns or 'idf' not in df.columns:
    raise ValueError(
        'IDF table CSV must contain token and idf columns.'
    )
  return df.set_index('token')['idf'].to_dict()


class RetokenizerBase:
  """Base class for retokenizers."""

  def retokenize(self, words: Sequence[str]) -> Iterator[tuple[str, int, int]]:
    """Retokenizes an input text defined by a sequence of words.

    Args:
      words: A sequence of words or word-pieces. This sequence is expected
        to contain explicit spaces to define word boundaries (e.g.,
        [' word1', ' word2']) for space-delimited languages.
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

  def _tokenize(self, text: str) -> Iterator[tuple[str, int, int]]:
    for token in self.model(text):
      yield token.text, token.idx, token.idx + len(token.text) - 1

  def retokenize(self, words: Sequence[str]) -> Iterator[tuple[str, int, int]]:
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

  def retokenize(self, words: Sequence[str]) -> Iterator[tuple[str, int, int]]:
    for index, word in enumerate(words):
      normalized_word = self._normalize(word)
      if normalized_word:
        yield normalized_word.lower(), index, index


class SegmenterBase:
  """Base class for segmenters."""

  def segment(
      self, words: Sequence[str]
  ) -> Iterator[tuple[str, float, int, int]]:
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
  ) -> Iterator[tuple[str, float, int, int]]:
    for word, front, back in self.retokenizer.retokenize(words):
      if word in self.idf_table:
        yield word, self.idf_table[word], front, back


class LongestPrefixIDFSegmenter(SegmenterBase):
  """Represents an IDF term extractor using longest prefix matching."""

  def __init__(self, idf_table: dict[str, float]):
    string_keyed_idf_table = {str(k): v for k, v in idf_table.items()}
    self.trie = pygtrie.Trie(string_keyed_idf_table)

  def _segment_text(self, text: str) -> Iterator[tuple[str, float, int]]:
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


class TextSegmenterEncoder(encoder.MultiModalEncoder):
  """An encoder that segments text and returns top-k segments with scores."""

  def __init__(self, segmenter: SegmenterBase, top_k: int = 1):
    super().__init__()
    self.segmenter = segmenter
    self.top_k = top_k

  def _setup(self):
    pass

  def _check_input_types(self, batch: Sequence[types.MultiModalObject]) -> None:
    if not all(isinstance(x, types.SoundEmbedding) for x in batch):
      raise ValueError(
          'TextSegmenterEncoder only supports SoundEmbedding inputs.'
      )

  def _encode(
      self, batch: Sequence[types.MultiModalObject]
  ) -> Sequence[types.SoundEmbedding]:
    outputs = []
    for sound_embedding in batch:
      assert isinstance(sound_embedding, types.SoundEmbedding)
      words = sound_embedding.embedding
      segments = list(self.segmenter.segment([str(w) for w in words]))
      if not segments:
        outputs.append(
            types.SoundEmbedding(
                embedding=np.array([['', 0.0]], dtype=object),
                timestamps=np.array([[0.0, 0.0]], dtype=float),
                context=sound_embedding.context,
            )
        )
        continue
      segments.sort(key=lambda x: x[1], reverse=True)
      top_segments = segments[: self.top_k]
      terms = np.array([term for term, _, _, _ in top_segments])
      saliency_scores = np.array(
          [score for _, score, _, _ in top_segments],
          dtype=float
      )
      outputs.append(
          types.SoundEmbedding(
              embedding=terms,
              scores=saliency_scores,
              timestamps=np.array(
                  [
                      [
                          sound_embedding.timestamps[front][0],
                          sound_embedding.timestamps[back][1],
                      ]
                      for _, _, front, back in top_segments
                  ]
              ),
              context=sound_embedding.context,
          )
      )
    return outputs


def create_max_idf_segment_encoder(
    asr_encoder: whisper_encoder.Whisper,
    language: str,
    top_k: int = 1,
    idf_table: Optional[dict[str, float]] = None,
    idf_table_path: Optional[str] = None,
) -> encoder.CascadeEncoder:
  """Factory function that assembles a cascaded salient term encoder.

  This function builds a two-stage encoder:
  1. An ASR encoder (like ForcedAlignmentEncoder) runs to get word timestamps.
  2. A TextSegmenterEncoder runs on the output to find and score salient terms.

  It requires either a pre-loaded
  idf_table or a path to a CSV file in idf_table_path.

  Args:
    asr_encoder: An initialized ASR encoder (e.g., ForcedAlignmentEncoder).
    language: The language code (e.g., 'en', 'ja') to configure the segmenter.
    top_k: The number of top salient terms to return.
    idf_table: A dictionary mapping tokens to their IDF scores.
    idf_table_path: Path to a CSV file containing the idf_table.

  Returns:
    A configured CascadeEncoder ready for use.
  """
  if idf_table is None and idf_table_path is None:
    raise ValueError(
        'Either idf_table or idf_table_path must be provided.'
    )

  if idf_table is not None and idf_table_path is not None:
    raise ValueError(
        'Provide either idf_table or idf_table_path, not both.'
    )

  if idf_table_path:
    idf_table = load_idf_table_from_csv(idf_table_path)

  if language.startswith('ja'):
    segmenter = LongestPrefixIDFSegmenter(idf_table)
  else:
    segmenter = TokenIDFSegmenter(
        idf_table,
        SpacyRetokenizer(
            language=('xx' if language.startswith('ko') else language)
        ),
    )

  return encoder.CascadeEncoder(
      encoders=[
          asr_encoder,
          TextSegmenterEncoder(segmenter, top_k),
      ]
  )
