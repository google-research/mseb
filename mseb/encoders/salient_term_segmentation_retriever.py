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

"""Retrieval for embedding-based salient term segmentation.

This module performs segmentation directly in representation space. It computes
predictions using frame-level sound embeddings and salient term embeddings: for
each reference segment (a salient term with a known time span), frame embeddings
are averaged over the temporal span and scored against candidate term embeddings
via dot-product similarity to produce ranked predictions.
"""

import dataclasses
import os
from typing import Mapping, Sequence

from absl import logging
import jaxtyping
from mseb import encoder
from mseb import types
from mseb.evaluators import segmentation_evaluator
from mseb.tasks import segmentation as segmentation_task
import numpy as np


@dataclasses.dataclass(frozen=True)
class SegmentEmbedding(segmentation_evaluator.Segment):
  vector: types.MultiModalEmbedding | None = None


def _average_embedding_over_span(
    frame_embeddings: jaxtyping.Float[jaxtyping.Array, 'T D'],
    frame_timestamps: jaxtyping.Float[jaxtyping.Array, 'T 2'],
    start_time: float,
    end_time: float,
) -> np.ndarray:
  """Averages frame-level embeddings that overlap with a given time span.

  A frame is considered to overlap with the span [start_time, end_time] if
  its midpoint falls within the span.

  Args:
    frame_embeddings: Frame-level embeddings of shape [T, D].
    frame_timestamps: Frame-level timestamps of shape [T, 2], where each row is
      [frame_start, frame_end] in seconds.
    start_time: Start of the target span in seconds.
    end_time: End of the target span in seconds.

  Returns:
    The mean embedding vector [D] over the overlapping frames, or a zero
    vector if no frames overlap.
  """
  midpoints = (frame_timestamps[:, 0] + frame_timestamps[:, 1]) / 2.0
  mask = (midpoints >= start_time) & (midpoints <= end_time)
  selected = frame_embeddings[mask]
  if len(selected) == 0:
    return np.zeros_like(frame_embeddings[0])
  return np.mean(selected, axis=0)


class SalientTermSegmentationRetriever(encoder.MultiModalEncoder):
  """Encoder that produces segmentation predictions via retrieval.

  This encoder takes frame-level sound embeddings (from a model) and
  salient term embeddings (from the task) and computes segmentation
  predictions. For each reference segment (a salient term with a known
  time span):
    1. Selects the sound's frame embeddings [T, D] that overlap with the
       segment's [start_time, end_time] span.
    2. Averages those frame embeddings to produce a single span embedding [D].
    3. Computes the dot product between the span embedding and each salient
       term's embedding to produce a confidence score.
    4. Returns the term with the highest score as the prediction for that
       segment.

  The encoder must be configured with a SegmentationTask via `set_task`
  before use, so it can access the salient term lists and their embeddings.
  """

  def __init__(self):
    super().__init__()
    # Set in `set_task`.
    self._embeddings_dir: str | None = None
    self._terms_by_sound_id: (
        Mapping[str, Sequence[segmentation_evaluator.Segment]] | None
    ) = None
    # Set in `_setup`.
    self._term_embeddings_by_sound_id: (
        Mapping[str, Sequence[SegmentEmbedding]] | None
    ) = None

  def set_task(self, task: segmentation_task.SegmentationSelectionTask) -> None:  # pyrefly: ignore[bad-override]
    """Sets the segmentation task to access salient term embeddings.

    This should be called before `setup`. The task's `setup` method must have
    already been called to populate the term embeddings.

    Args:
      task: A SegmentationTask with salient_term_lists().
    """
    self._embeddings_dir = task.embeddings_dir
    self._terms_by_sound_id = {k: v for k, v in task.salient_term_lists()}  # pyrefly: ignore[missing-attribute]

  def _setup(self):
    """Sets up term embeddings from the task's evaluator."""
    if self._embeddings_dir is None:
      raise ValueError(
          'SalientTermSegmentationRetriever must be configured with a'
          ' SegmentationTask via `set_task` before `setup` is called.'
      )
    if self._terms_by_sound_id is None:
      raise ValueError(
          'SalientTermSegmentationRetriever must be configured with a'
          ' SegmentationTask via `set_task` before `_setup` is called.'
      )
    embeddings_path_prefix = os.path.join(self._embeddings_dir, 'embeddings')
    logging.info(
        'Loading salient term embeddings cache from %s',
        embeddings_path_prefix,
    )
    term_embeddings = segmentation_task.runner_lib.load_embeddings(
        embeddings_path_prefix
    )
    self._term_embeddings_by_sound_id = {
        utt_id: [
            SegmentEmbedding(
                embedding=st.embedding,
                start_time=st.start_time,
                end_time=st.end_time,
                vector=term_embeddings[st.embedding],
            )
            for st in stl
        ]
        for utt_id, stl in self._terms_by_sound_id.items()
    }

  def _check_input_types(self, batch: Sequence[types.MultiModalObject]) -> None:
    if not all(isinstance(x, types.SoundEmbedding) for x in batch):
      raise ValueError(
          'SalientTermSegmentationRetriever only supports SoundEmbedding '
          'inputs with frame-level embeddings.'
      )

  def _compute_prediction(
      self,
      sound_embedding: types.SoundEmbedding,
      term_embeddings: Sequence[SegmentEmbedding],
  ) -> types.SoundEmbedding:
    """Computes segmentation prediction for a single sound.

    Args:
      sound_embedding: Frame-level sound embedding with shape [T, D] and
        timestamps of shape [T, 2].
      term_embeddings: Sequence of SegmentEmbedding, each with a known time span
        and an embedding vector for the salient term.

    Returns:
      A SoundEmbedding with predicted term labels, confidence scores, and
      timestamps.
    """
    sound_frames: jaxtyping.Float[jaxtyping.Array, 'T D'] = sound_embedding.embedding  # pyrefly: ignore[bad-assignment]
    if (
        hasattr(sound_embedding, 'timestamps')
        and sound_embedding.timestamps is not None
    ):
      sound_timestamps: jaxtyping.Float[jaxtyping.Array, 'T 2'] = sound_embedding.timestamps  # pyrefly: ignore[bad-assignment]
    else:
      sound_timestamps = np.array([[0.0, -1.0]] * sound_frames.shape[0])  # pyrefly: ignore[bad-assignment]

    pred_terms = []
    pred_scores = []
    pred_timestamps = []

    for segment in term_embeddings:
      embeds = segment.vector
      assert embeds is not None and hasattr(embeds, 'embedding')
      embed: jaxtyping.Float[jaxtyping.Array, 'T D'] = embeds.embedding  # pyrefly: ignore[bad-assignment]

      if sound_frames.shape[0] > 1:
        span_embedding = _average_embedding_over_span(
            sound_frames,
            sound_timestamps,
            segment.start_time,
            segment.end_time,
        )
      else:
        span_embedding = sound_frames[0]

      score = float(np.dot(span_embedding, embed[0]))  # pyrefly: ignore[bad-argument-type]
      pred_terms.append(embeds.context.id)  # pyrefly: ignore[missing-attribute]
      pred_scores.append(score)
      pred_timestamps.append((segment.start_time, segment.end_time))

    # Sort by score descending.
    sorted_indices = np.argsort(-np.array(pred_scores))
    top_terms = [pred_terms[i] for i in sorted_indices]
    top_scores = [pred_scores[i] for i in sorted_indices]
    top_timestamps = [pred_timestamps[i] for i in sorted_indices]

    return types.SoundEmbedding(
        embedding=np.array(top_terms),
        scores=np.array(top_scores),  # pyrefly: ignore[bad-argument-type]
        timestamps=np.array(top_timestamps),  # pyrefly: ignore[bad-argument-type]
        context=types.SoundContextParams(
            id=sound_embedding.context.id,
            sample_rate=-1,
            length=-1,
        ),
    )

  def _encode(
      self, batch: Sequence[types.MultiModalObject]
  ) -> Sequence[types.SoundEmbedding]:
    if self._term_embeddings_by_sound_id is None:
      raise ValueError(
          'Term embeddings are not set. Did you call set_task() and '
          'task.setup() before encoding?'
      )

    outputs = []
    for sound_embedding in batch:
      assert isinstance(sound_embedding, types.SoundEmbedding)
      sound_id = sound_embedding.context.id
      if sound_id not in self._term_embeddings_by_sound_id:
        logging.warning(
            'No term embeddings found for sound_id=%s, skipping.', sound_id
        )
        outputs.append(
            types.SoundEmbedding(
                embedding=np.array([], dtype=object),
                scores=np.array([], dtype=float),  # pyrefly: ignore[bad-argument-type]
                timestamps=np.array([[]], dtype=float),  # pyrefly: ignore[bad-argument-type]
                context=sound_embedding.context,
            )
        )
        continue

      term_embeddings = self._term_embeddings_by_sound_id[sound_id]
      prediction = self._compute_prediction(sound_embedding, term_embeddings)
      outputs.append(prediction)

    return outputs
