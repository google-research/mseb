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

import jaxtyping
from mseb import encoder
from mseb import types
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


class CascadedSegmentationEncoder(encoder.MultiModalEncoder):
  """Encodes anaudio sequence into segments using a cascaded approach.

   1. Use an ASR encoder to transcribe the speech into text.
   2. Use a text-based segmenter to segment the text into segments.
   3. Select the top-k segments with the highest segment scores.
   4. Return the start and end timestamps, texts, and scores of the selected
      segments
  """

  def __init__(
      self,
      asr_encoder: whisper_encoder.Whisper,
      segmenter: SegmenterBase,
      top_k: int = 1,
      asr_kwargs: dict[str, Any] | None = None,
  ):
    super().__init__()
    self.asr_encoder = asr_encoder
    self.segmenter = segmenter
    self.top_k = top_k
    self.asr_kwargs = asr_kwargs or {}

  def _setup(self):
    self.asr_encoder.setup()

  def _check_input_types(self, batch: Sequence[types.MultiModalInput]) -> None:
    if not all(isinstance(x, types.Sound) for x in batch):
      raise ValueError(
          'CascadedSegmentationEncoder only supports a batch of all Sound '
          'inputs.'
      )

  def _encode(
      self, batch: Sequence[types.MultiModalInput]
  ) -> Sequence[types.SoundEmbedding]:
    """Encodes a batch of sound sources into segments.

    Args:
      batch: A sequence of sound sources to encode.

    Returns:
      A list of types.SoundEmbedding objects, one for each input:
       timestamps: Array of start and end times tuple for each segment.
       segments: Array of text and score for each segment.
    """
    sound_batch: list[types.Sound] = []
    for example in batch:
      assert isinstance(example, types.Sound)
      sound_batch.append(example)
    sound_embeddings = self.asr_encoder.encode_batch(
        sound_batch, **self.asr_kwargs
    )
    outputs = []
    for sound_embedding in sound_embeddings:
      words: jaxtyping.Shaped[np.ndarray, 'N'] = sound_embedding.embedding
      segments = list(self.segmenter.segment([str(x) for x in words]))
      if not segments:
        outputs.append(
            types.SoundEmbedding(
                embedding=np.array([['', 0.0]], dtype=object),
                timestamps=np.array([0.0, 0.0], dtype=float),
                context=sound_embedding.context,
            )
        )
        continue
      # TODO(allauzen): consider using a heap instead of sorting.
      segments.sort(key=lambda x: x[1], reverse=True)
      outputs.append(
          types.SoundEmbedding(
              embedding=np.array([
                  [term, score] for term, score, _, _ in segments[: self.top_k]
              ]),
              timestamps=np.array([
                  [
                      sound_embedding.timestamps[front][0],
                      sound_embedding.timestamps[back][1],
                  ]
                  for _, _, front, back in segments[: self.top_k]
              ]),
              context=sound_embedding.context,
          )
      )
    return outputs


class MaxIDFSegmentEncoder(CascadedSegmentationEncoder):
  """Encodes an audio sequence into the top-k segments with the highest IDF scores in the output of an ASR encoder."""

  def __init__(
      self,
      asr_encoder: whisper_encoder.Whisper,
      idf_table: dict[str, float],
      language: str,
      top_k: int = 1,
  ):
    """Initialize the MaxIDFSegmentEncoder.

    Args:
      asr_encoder: The ASR encoder to use.
      idf_table: The IDF table to use.
      language: The language of the speech.
      top_k: The number of segments to return.
    """
    # As workaround for some issues with the Japanase and Korean spacy
    # tokenizers, we use a different segmenter for these languages.
    if language.startswith('ja'):
      segmenter = LongestPrefixIDFSegmenter(idf_table)
    else:
      segmenter = TokenIDFSegmenter(
          idf_table,
          SpacyRetokenizer(
              language=('xx' if language.startswith('ko') else language)
          ),
      )
    super().__init__(asr_encoder, segmenter, top_k)
