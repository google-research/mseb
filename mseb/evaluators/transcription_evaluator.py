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

"""Evaluator for transcription tasks."""

from __future__ import annotations

import dataclasses
import functools
import re
from typing import Callable, Mapping, Sequence

import jaxtyping
from mseb import evaluator
from mseb import metrics
from mseb import types
import numpy as np
import opencc
from whisper.normalizers import basic
from whisper.normalizers import english


def wer(value: float = 0.0, std: float | None = None):
  return types.Score(
      metric='WER',
      description='Word Error Rate',
      value=value,
      min=0,
      max=float('inf'),
      std=std,
  )


def ser(value: float = 0.0, std: float | None = None):
  return types.Score(
      metric='SER',
      description='Sentence Error Rate',
      value=value,
      min=0,
      max=float('inf'),
      std=std,
  )


def cer(value: float = 0.0, std: float | None = None):
  return types.Score(
      metric='CER',
      description='Character Error Rate',
      value=value,
      min=0,
      max=float('inf'),
      std=std,
  )


# Locales for which a character error rate is reported in addition to the
# word-based metrics, because words are not whitespace-delimited (or are only
# inconsistently so) in these writing systems.
_CJK_LOCALES = frozenset({'cmn_hans_cn', 'ja_jp', 'ko_kr'})

# OpenCC configuration for Traditional Chinese -> Simplified Chinese. The
# '.json' suffix is required when the package's bundled config directory is not
# present on disk (e.g. when configs are loaded from runfiles).
_TRADITIONAL_TO_SIMPLIFIED_CONFIG = 't2s.json'


@functools.lru_cache(maxsize=None)
def _get_converter(config: str) -> opencc.OpenCC:
  """Returns a converter for `config`, loading its dictionaries only once."""
  return opencc.OpenCC(config)


def tc2sc(text: str) -> str:
  """Returns `text` converted from Traditional to Simplified Chinese."""
  return _get_converter(_TRADITIONAL_TO_SIMPLIFIED_CONFIG).convert(text)


_remove_spaces = functools.partial(re.sub, r'\s+', '')


# Replicates behavior of jiwer.Compose as we don't want import the dep here.
def _compose(function_list):
  """Returns a function that chains functions in function_list."""
  return lambda v: functools.reduce(lambda res, f: f(res), function_list, v)


@dataclasses.dataclass
class TranscriptTruth:
  sound_id: str
  text: str
  language: str  # For text normalization.
  text_transform: Callable[[str], str] | None = None

  def __post_init__(self):
    if self.text_transform is None:
      self.text_transform = text_transform(self.language)


@functools.cache
def text_transform(language: str) -> Callable[[str], str]:
  if language.split('_')[0].lower() == 'en':
    return english.EnglishTextNormalizer()
  else:
    return basic.BasicTextNormalizer()


class TranscriptionEvaluator:
  """Evaluator for transcription tasks.

  In addition to the word-based metrics, a character error rate is reported for
  CJK locales, where whitespace does not delimit words. The same text
  normalization is applied to the reference and the hypothesis as for the
  word-based metrics, plus language-specific transformations: traditional
  characters are converted to simplified ones with OpenCC for `cmn_hans_cn`,
  and whitespace is stripped for `cmn_hans_cn` and `ja_jp`.
  """

  def compute_predictions(
      self, embeddings_by_sound_id: types.MultiModalEmbeddingCache
  ) -> Mapping[str, types.TextPrediction]:
    """Converts the embeddings to text predictions.

    Args:
      embeddings_by_sound_id: The embeddings to evaluate.

    Returns:
      A dictionary mapping sound_id to a TextPrediction with the transcript.
    """
    transcripts_by_sound_id = {}
    for sound_id, embeddings in embeddings_by_sound_id.items():
      assert hasattr(embeddings, 'embedding')
      embedding: jaxtyping.Shaped[np.ndarray, '1'] = embeddings.embedding  # pyrefly: ignore[bad-assignment]
      transcripts_by_sound_id[sound_id] = types.TextPrediction(
          prediction=str(embedding[0]),
          context=types.PredictionContextParams(id=sound_id),
          streaming_stats=getattr(embeddings, 'streaming_stats', None),
      )
    return transcripts_by_sound_id

  def compute_metrics(
      self,
      transcript_by_sound_id: Mapping[str, types.TextPrediction],
      transcript_truths: Sequence[TranscriptTruth],
  ) -> list[types.Score]:
    """Returns quality metrics of the transcriptions."""

    values_by_metric: dict[str, list[types.WeightedValue]] = {
        'wer': [],
        'ser': [],
        'no_response': [],
    }
    for transcript_truth in transcript_truths:
      transcript = transcript_by_sound_id[transcript_truth.sound_id]

      if transcript.prediction != types.LLM_NO_RESPONSE_STR:
        word_error_count, ref_word_count = metrics.compute_word_errors(
            truth=transcript_truth.text,
            hypothesis=transcript.prediction,
            text_transform=transcript_truth.text_transform,
        )
        values_by_metric['wer'].append(
            types.WeightedValue(value=word_error_count, weight=ref_word_count)
        )
        values_by_metric['ser'].append(
            types.WeightedValue(value=float(word_error_count != 0.0))
        )
        values_by_metric['no_response'].append(
            types.WeightedValue(value=0.0, weight=1.0)
        )
      else:
        word_error_count, ref_word_count = metrics.compute_word_errors(
            truth=transcript_truth.text,
            hypothesis='',
            text_transform=transcript_truth.text_transform,
        )
        values_by_metric['wer'].append(
            types.WeightedValue(value=word_error_count, weight=ref_word_count)
        )
        values_by_metric['ser'].append(
            types.WeightedValue(value=1.0, weight=1.0)
        )
        values_by_metric['no_response'].append(
            types.WeightedValue(value=1.0, weight=1.0)
        )

    wer_score = wer(
        *evaluator.compute_weighted_average_and_std(values_by_metric['wer'])
    )
    ser_score = ser(
        *evaluator.compute_weighted_average_and_std(values_by_metric['ser'])
    )
    no_result_rate = evaluator.compute_weighted_average_and_std(
        values_by_metric['no_response']
    )
    no_result_score = types.Score(
        metric='NoResultRate',
        description=(
            'No result rate, for example, the server failed to return a result.'
        ),
        value=no_result_rate[0],
        min=0,
        max=1,
        std=no_result_rate[1],
    )
    utt_count_score = types.Score(
        metric='UtteranceCount',
        description='Number of utterances scored.',
        value=float(len(transcript_truths)),
        min=0,
        max=float('inf'),
    )
    word_count_score = types.Score(
        metric='WordCount',
        description='Number of words in reference transcripts.',
        value=float(sum(w.weight for w in values_by_metric['wer'])),
        min=0,
        max=float('inf'),
    )
    scores = [
        wer_score,
        ser_score,
        no_result_score,
        utt_count_score,
        word_count_score,
    ]
    scores.extend(
        self._compute_cjk_metrics(transcript_by_sound_id, transcript_truths)
    )
    return scores

  def _compute_cjk_metrics(
      self,
      transcript_by_sound_id: Mapping[str, types.TextPrediction],
      transcript_truths: Sequence[TranscriptTruth],
  ) -> list[types.Score]:
    """Returns CER and CharCount over the CJK subset of `transcript_truths`.

    Returns an empty list if none of the truths are in a CJK locale, so that
    non-CJK evaluations are unaffected.

    Args:
      transcript_by_sound_id: The predicted transcripts, keyed by sound id.
      transcript_truths: The reference transcripts.
    """
    cer_values = []

    for transcript_truth in transcript_truths:
      locale = transcript_truth.language.lower().replace('-', '_')
      if locale not in _CJK_LOCALES:
        continue

      transcript = transcript_by_sound_id[transcript_truth.sound_id]

      hyp = (
          transcript.prediction
          if transcript.prediction != types.LLM_NO_RESPONSE_STR
          else ''
      )
      # Apply the same text norm as the word-based metrics, and any
      # CJK-specific transforms on top of it.
      transforms = [transcript_truth.text_transform]
      if locale == 'cmn_hans_cn':
        transforms.append(tc2sc)
      if locale in {'cmn_hans_cn', 'ja_jp'}:
        transforms.append(_remove_spaces)
      char_error_count, ref_char_count = metrics.compute_character_errors(
          truth=transcript_truth.text,
          hypothesis=hyp,
          reference_transform=_compose(transforms),
          hypothesis_transform=_compose(transforms),
      )
      cer_values.append(
          types.WeightedValue(value=char_error_count, weight=ref_char_count)
      )

    if not cer_values:
      return []

    cer_val, cer_std = evaluator.compute_weighted_average_and_std(cer_values)
    char_count_score = types.Score(
        metric='CharCount',
        description='Number of characters in reference transcripts.',
        value=float(sum(w.weight for w in cer_values)),
        min=0,
        max=float('inf'),
    )
    return [cer(cer_val, cer_std), char_count_score]
