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

"""Clustering evaluator."""

import dataclasses
from typing import Iterable
from mseb import types
import numpy as np
import sklearn


def cluster_kmeans(
    data: np.ndarray, nlabels: int, batch_size: int
) -> np.ndarray:
  model = sklearn.cluster.MiniBatchKMeans(
      n_clusters=nlabels, batch_size=batch_size, n_init='auto'
  )
  model.fit(data)
  return model.labels_


def vmeasure_score(value: float = 0.0):
  return types.Score(
      metric='VMeasure',
      description=(
          'V-measure clustering metric: Normalised mutal information'
          ' against a set of labels.'
      ),
      value=value,
      min=0,
      max=1,
  )


@dataclasses.dataclass
class ClusteringExample:
  sound_id: str
  label: str


class ClusteringEvaluator:
  """Evaluator for KMeans clustering."""

  def __call__(
      self,
      embeddings: types.MultiModalEmbeddingCache,
      examples: Iterable[ClusteringExample],
  ) -> list[types.Score]:
    embedded = []
    labels = []
    for ex in examples:
      embedded.append(embeddings[ex.sound_id].embedding)
      labels.append(ex.label)
    data = np.vstack(embedded)
    clusters = cluster_kmeans(data, nlabels=len(set(labels)), batch_size=32)
    v_measure = sklearn.metrics.v_measure_score(
        labels_true=labels, labels_pred=clusters
    )
    return [vmeasure_score(v_measure)]
