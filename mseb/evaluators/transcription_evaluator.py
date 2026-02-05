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

"""Evaluator for transcription tasks."""

from __future__ import annotations

import dataclasses
from typing import Callable, Mapping, Sequence

import jaxtyping
from mseb import evaluator
from mseb import metrics
from mseb import types
import numpy as np
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


@dataclasses.dataclass
class TranscriptTruth:
  sound_id: str
  text: str
  language: str  # For text normalization.


class TranscriptionEvaluator:
  """Evaluator for transcription tasks."""

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
      embedding: jaxtyping.Shaped[np.ndarray, '1'] = embeddings.embedding
      transcripts_by_sound_id[sound_id] = types.TextPrediction(
          prediction=str(embedding[0]),
          context=types.PredictionContextParams(id=sound_id),
      )
    return transcripts_by_sound_id

  def compute_metrics(
      self,
      transcript_by_sound_id: Mapping[str, types.TextPrediction],
      transcript_truths: Sequence[TranscriptTruth],
  ) -> list[types.Score]:
    """Returns quality metrics of the transcriptions."""

    def text_transform(language: str) -> Callable[[str], str]:
      if language.split('_')[0].lower() == 'en':
        return english.EnglishTextNormalizer()
      else:
        return basic.BasicTextNormalizer()

    values_by_metric: dict[str, list[types.WeightedValue]] = {
        'wer': [],
        'ser': [],
        'no_response': [],
    }
    for transcript_truth in transcript_truths:
      transcript = transcript_by_sound_id[transcript_truth.sound_id]

      if transcript.prediction != types.LLM_NO_RESPONSE_STR:
        word_errors, word_errors_weight = metrics.compute_word_errors(
            truth=transcript_truth.text,
            hypothesis=transcript.prediction,
            text_transform=text_transform(transcript_truth.language),
        )
        values_by_metric['wer'].append(
            types.WeightedValue(
                value=word_errors / word_errors_weight,
                weight=word_errors_weight,
            )
        )
        values_by_metric['ser'].append(
            types.WeightedValue(value=float(word_errors != 0.0))
        )
        values_by_metric['no_response'].append(
            types.WeightedValue(value=0.0, weight=1.0)
        )
      else:
        word_errors, word_errors_weight = metrics.compute_word_errors(
            truth=transcript_truth.text,
            hypothesis='',
            text_transform=text_transform(transcript_truth.language),
        )
        values_by_metric['wer'].append(
            types.WeightedValue(
                value=word_errors / word_errors_weight,
                weight=word_errors_weight,
            )
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
    return [wer_score, ser_score, no_result_score]
