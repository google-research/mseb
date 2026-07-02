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

"""TF-IDF encoder."""

from collections.abc import Sequence
from typing import Optional

from mseb import encoder as encoder_lib
from mseb import types
from mseb.encoders import segmentation_encoder
from mseb.encoders import whisper_encoder
import numpy as np


def _segment_embedding_to_term_vector(
    emb: types.SoundEmbedding,
) -> dict[str, np.float64]:
  tv = {}
  assert emb.scores is not None
  for term, value in zip(emb.embedding.tolist(), emb.scores.tolist()):
    if term not in tv:
      tv[term] = value
    else:
      tv[term] += value
  return tv


class TermExtractorEncoder(segmentation_encoder.TextSegmenterEncoder):
  """Encodes as term vectors."""

  def __init__(
      self, segmenter: segmentation_encoder.SegmenterBase, top_k: int = 100
  ):
    super().__init__(segmenter=segmenter, top_k=top_k)

  def _check_input_types(self, batch: Sequence[types.MultiModalObject]) -> None:
    if not all(isinstance(x, types.SoundEmbedding) for x in batch):
      raise ValueError(
          'TermExtractorEncoder only supports SoundEmbedding inputs.'
      )

  def _encode(
      self, batch: Sequence[types.MultiModalObject]
  ) -> Sequence[types.SoundEmbedding]:
    segment_embeddings = super()._encode(batch)
    tvs = [_segment_embedding_to_term_vector(emb) for emb in segment_embeddings]
    outputs = []
    for sound_embedding, tv in zip(batch, tvs):
      assert isinstance(sound_embedding, types.SoundEmbedding)
      outputs.append(
          types.SoundEmbedding(
              embedding=np.array([tv], dtype=object),
              scores=sound_embedding.scores,
              timestamps=np.array([[  # pyrefly: ignore[bad-argument-type]
                  sound_embedding.timestamps[0, 0],
                  sound_embedding.timestamps[-1, 1],
              ]]),
              context=sound_embedding.context,
          )
      )
    return outputs


def create_tf_idf_encoder(
    language: str,
    top_k: int,
    idf_table: Optional[dict[str, float]],
    idf_table_path: Optional[str],
) -> TermExtractorEncoder:
  """Creates a TF-IDF extrator-encoder.

  Args:
    language: The language code (e.g., 'en', 'ja') to configure the segmenter.
    top_k: The number of top salient terms to return.
    idf_table: A dictionary mapping tokens to their IDF scores.
    idf_table_path: Path to a CSV file containing the idf_table.

  Returns:
    A configured TermExtractorEncoder ready for use.
  """
  if language.startswith('ja'):
    segmenter = segmentation_encoder.LongestPrefixIDFSegmenter(
        idf_table=idf_table, idf_table_path=idf_table_path
    )
  else:
    retokenizer = segmentation_encoder.SpacyRetokenizer(
        language=('xx' if language.startswith('ko') else language)
    )
    segmenter = segmentation_encoder.TokenIDFSegmenter(
        retokenizer=retokenizer,
        idf_table=idf_table,
        idf_table_path=idf_table_path,
    )
  return TermExtractorEncoder(segmenter, top_k)


def create_tf_idf_cascade(
    asr_encoder: encoder_lib.MultiModalEncoder,
    language: str,
    top_k: int,
    idf_table: Optional[dict[str, float]],
    idf_table_path: Optional[str],
) -> encoder_lib.CascadeEncoder:
  """Internal helper to assemble the segmenter and cascade encoder.

  This function builds a two-stage encoder:
  1. An ASR encoder .
  2. A TextSegmenterEncoder runs on the output to find and score salient terms.

  It requires either a pre-loaded
  idf_table or a path to a CSV file in idf_table_path.

  Args:
    asr_encoder: An initialized ASR encoder..
    language: The language code (e.g., 'en', 'ja') to configure the segmenter.
    top_k: The number of top salient terms to return.
    idf_table: A dictionary mapping tokens to their IDF scores.
    idf_table_path: Path to a CSV file containing the idf_table.

  Returns:
    A configured CascadeEncoder ready for use.
  """
  return encoder_lib.CascadeEncoder(
      encoders=[
          asr_encoder,
          create_tf_idf_encoder(
              language=language,
              top_k=top_k,
              idf_table=idf_table,
              idf_table_path=idf_table_path,
          ),
      ]
  )


def create_tf_idf_cascade_whisper(
    whisper_model_path: str,
    language: str,
    top_k: int,
    idf_table: Optional[dict[str, float]],
    idf_table_path: Optional[str],
    device: str | None = None,
) -> encoder_lib.CascadeEncoder:
  return create_tf_idf_cascade(
      asr_encoder=whisper_encoder.SpeechToTextEncoder(
          model_path=whisper_model_path,
          device=device,
          word_timestamps=False,
      ),
      language=language,
      top_k=top_k,
      idf_table=idf_table,
      idf_table_path=idf_table_path,
  )


def term_vector_weighted_sum(
    tvs: Sequence[dict[str, np.float64]],
    weights: Sequence[np.float64],
) -> dict[str, np.float64]:
  """Computes a weighted average of term vectors."""
  weighted_sum_tv = {}
  for tv, weight in zip(tvs, weights):
    for term, value in tv.items():
      if term not in weighted_sum_tv:
        weighted_sum_tv[term] = value * weight
      else:
        weighted_sum_tv[term] += value * weight
  return weighted_sum_tv


def term_vector_weighted_average(
    tvs: Sequence[dict[str, np.float64]],
    weights: Sequence[np.float64],
) -> dict[str, np.float64]:
  """Computes a weighted average of term vectors."""
  weighted_average_tv = term_vector_weighted_sum(tvs, weights)
  total_weight = sum(weights)
  for term in weighted_average_tv:
    weighted_average_tv[term] /= total_weight
  return weighted_average_tv


def combine_tf_idf_embeddings(
    embeddings: Sequence[types.SoundEmbedding],
    params: types.SoundContextParams,
) -> types.SoundEmbedding:
  """Combines TF-IDF embeddings into a single embedding."""
  tvs = [dict(emb.embedding[0]) for emb in embeddings]
  weights = [np.exp(emb.scores[0]) for emb in embeddings]  # pyrefly: ignore[unsupported-operation]
  tv = term_vector_weighted_average(tvs, weights)  # pyrefly: ignore[bad-argument-type]
  return types.SoundEmbedding(
      embedding=np.array([tv], dtype=object),
      scores=None,
      timestamps=np.array([[  # pyrefly: ignore[bad-argument-type]
          params.waveform_start_second,
          params.waveform_end_second,
      ]]),
      context=params,
  )
