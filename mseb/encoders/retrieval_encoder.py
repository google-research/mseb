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

"""Retrieval-based encoder for MultiModalObjects.

We support both non-partitioned and partitioned indexes. Note that the
implementation for partitioned indexes is inefficient because it reloads the
different partitions for each batch.
"""

import functools
import logging
from typing import Callable, Iterable, Sequence

import jaxtyping
from mseb import encoder
from mseb import types
from mseb.evaluators import retrieval_evaluator
from mseb.tasks import retrieval as retrieval_task


class RetrievalEncoder(encoder.MultiModalEncoder):
  """Encoder that uses a retrieval model."""

  def __init__(self, top_k: int = 10, id_by_index_id_filepath: str = 'ids.txt'):
    super().__init__()
    self._top_k = top_k
    self._id_by_index_id_filepath = id_by_index_id_filepath
    self._index_dir: str | None = None
    self._evaluator: (
        retrieval_evaluator.RetrievalEvaluator
        | retrieval_evaluator.RetrievalEvaluatorPartitioned
    ) | None = None
    self._documents: Callable[[], Iterable[types.Text]] | None = None
    self._text_by_id: dict[str, str] | None = None

  def set_task(self, task: retrieval_task.RetrievalTask) -> None:
    self._index_dir = retrieval_task.INDEX_DIR.value or task.index_dir
    self._documents = functools.partial(
        task.documents_generator, task.get_documents_source()
    )

  def _setup(self):
    self._text_by_id = {}
    for document in self._documents():
      self._text_by_id[document.context.id] = document.text
    logging.info(
        'Created text_by_id mapping for %d documents.', len(self._text_by_id)
    )
    if retrieval_task._NUM_PARTITIONS.value == 1:  # pylint: disable=protected-access
      searcher, id_by_index_id = retrieval_evaluator.load_index(
          self._index_dir, self._id_by_index_id_filepath
      )
      self._evaluator = retrieval_evaluator.RetrievalEvaluator(
          searcher=searcher,
          id_by_index_id=id_by_index_id,
          top_k=self._top_k,
      )
    else:
      self._evaluator = retrieval_evaluator.RetrievalEvaluatorPartitioned(
          index_dir=self._index_dir, top_k=self._top_k
      )

  def _check_input_types(
      self, inputs: Sequence[types.MultiModalObject]
  ) -> None:
    if not all(isinstance(x, types.TextEmbedding) for x in inputs):
      raise ValueError(
          'RetrievalEncoder only supports a batch of TextEmbedding inputs.'
      )

  def _encode(
      self, batch: Sequence[types.MultiModalObject]
  ) -> Sequence[types.TextWithTitleAndContext]:
    embeddings_by_id = {}
    for x in batch:
      assert isinstance(x, types.TextEmbedding)
      embedding: jaxtyping.Float[jaxtyping.Array, '1 D'] = x.embedding
      embeddings_by_id[x.context.id] = types.TextEmbedding(
          embedding=embedding, spans=x.spans, context=x.context
      )
    assert self._evaluator is not None
    predictions: retrieval_evaluator.RetrievalPredictionsCache = (
        self._evaluator.compute_predictions(embeddings_by_id)
    )
    outputs = []
    assert self._text_by_id is not None
    for x in batch:
      prediction = predictions[x.context.id]
      assert isinstance(prediction, types.ValidListPrediction)
      prediction = types.ValidListPrediction([
          {**item, 'text': self._text_by_id[item['id']]}
          for item in prediction.items
      ])
      prediction.normalize(k=self._top_k)
      assert isinstance(x, types.TextEmbedding)
      output = types.TextWithTitleAndContext(
          text=x.context.text,
          title_text=None,
          context=x.context,
          context_text=prediction.to_json(),
      )
      outputs.append(output)
    return outputs
