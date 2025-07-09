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

"""Evaluator for reasoning tasks."""

from __future__ import annotations

import collections
import re
import string
from typing import Any, Dict, List, Sequence, Union

from mseb import encoder
from mseb import evaluator


def _normalize_answer(text: str, punc_chars: str, punc_repl: str) -> str:
  """Lower text and remove punctuation, articles and extra whitespace."""

  def remove_articles(s):
    return re.sub(r'\b(a|an|the)\b', ' ', s)

  def replace_punctuation(s):
    to_replace = set(punc_chars)
    return ''.join(punc_repl if ch in to_replace else ch for ch in s)

  def white_space_fix(s):
    return ' '.join(s.split())

  text = text.lower()
  text = replace_punctuation(text)
  text = remove_articles(text)
  text = white_space_fix(text)

  return text


def normalize_squad(answer):
  """Normalization used in official SQuAD evaluation script."""
  return _normalize_answer(answer, punc_chars=string.punctuation, punc_repl='')


def f1_score(target, prediction):
  """Token-based F1 score used XTREME-UP."""
  prediction_tokens = prediction.split()
  target_tokens = target.split()
  common = (collections.Counter(prediction_tokens) &
            collections.Counter(target_tokens))
  num_same = sum(common.values())
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(target_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1


class ReasoningEvaluator(evaluator.Evaluator):
  """Evaluator for question answering tasks."""

  def __init__(
      self,
      sound_encoder: encoder.Encoder,
      encode_kwargs: dict[str, Any],
  ):
    """Initializes the evaluator with the encoder."""
    super().__init__(sound_encoder, encode_kwargs=encode_kwargs)

  def __call__(
      self,
      sequence: Union[str, Sequence[float]],
      context: encoder.ContextParams,
      reference: str = '',
  ) -> dict[str, float]:
    """Evaluates a single question-answering example.

    Args:
      sequence: Unused.
      context: A JSON string containing the 'question', 'title', and 'context'
        for a single example is passed via the context.text field.
      reference: A string representing the reference answer for the example.

    Returns:
      A dictionary containing the F1 score for the example.
    """
    return self.evaluate_batch(
        sequences=[sequence],
        contexts=[context],
        references=[reference],
    )[0]

  def evaluate_batch(
      self,
      sequences: Sequence[Union[str, Sequence[float]]],
      contexts: Sequence[encoder.ContextParams],
      references: Sequence[str] = (),
  ) -> Sequence[dict[str, float]]:
    """Evaluates a batch of question-answering examples.

    Args:
      sequences: Unused.
      contexts: A sequence of encoder.ContextParams.
      references: A sequence of strings representing the reference answers for
        each example in `sequences`.

    Returns:
      A sequence of dictionaries, where each dictionary contains the F1 score
      for the corresponding example.
    """
    metrics_batch = []
    outputs = self.sound_encoder.encode_batch(sequences, contexts)
    for reference, output in zip(references, outputs):
      output = output[0].tolist()
      output = normalize_squad(output)
      reference = normalize_squad(reference)
      metrics_batch.append({
          'f1': f1_score(output, reference),
      })
    return metrics_batch

  def combine_scores(self, scores: List[Dict[str, float]]) -> Dict[str, float]:
    """Combines the scores of the examples."""
    return evaluator.compute_weighted_average_and_std(
        scores, (('f1', 'f1'),)
    )
