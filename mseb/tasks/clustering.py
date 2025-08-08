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

"""Clustering tasks."""

import os
import apache_beam as beam
from mseb import encoder as encoder_lib
from mseb import svq_data
from mseb import task
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


class EncodeDoFn(beam.DoFn):
  """A DoFn that wraps a SoundEncoder."""

  def __init__(self, encoder: encoder_lib.SoundEncoder):
    self._encoder = encoder
    self._sample_rate = 48000

  def setup(self):
    self._encoder.setup()

  def process(self, element):
    waveform = element['waveform']
    params = types.SoundContextParams(
        sample_rate=self._sample_rate,
        length=len(waveform),
    )
    yield self._encoder.encode(waveform, params=params)


def encode_svq_beam(base_path, encoder: encoder_lib.SoundEncoder):
  """Run SoundEncoder over svq dataset using beam."""
  with beam.Pipeline() as root:
    examples = svq_data.generate_examples_beam(
        root, os.path.join(base_path, 'utt_index.jsonl')
    )
    encoded = (
        root
        | 'ReadExamples' >> beam.Create(examples)
        | 'Encode' >> beam.ParDo(EncodeDoFn(encoder))
    )
    return encoded


def encode_svq(
    base_path: str, encoder: encoder_lib.SoundEncoder
):
  """Run SoundEncoder over svq dataset returning ndarray of embeddings."""
  examples = svq_data.generate_examples(
      os.path.join(base_path, 'utt_index.jsonl')
  )
  encoder.setup()
  embeddings = task.SoundEmbeddingsInMemory(encoder)
  for ex in examples:
    context = types.SoundContextParams(
        sample_rate=48000, length=len(ex['waveform'])
    )
    embedding, timestamps = encoder.encode(ex['waveform'], context)
    embeddings.embeddings[ex['utt_id']] = types.SoundEmbedding(
        embedding, timestamps, context
    )
    break
  return embeddings


def get_examples_svq(
    base_path: str, label_field: str, embeddings: task.SoundEmbeddings
) -> tuple[np.ndarray, np.ndarray]:
  """Get (utt_id, label) examples from svq dataset."""
  examples = svq_data.generate_examples(
      os.path.join(base_path, 'utt_index.jsonl')
  )
  embedded = []
  labels = []
  for ex in examples:
    embedded.append(embeddings[ex['utt_id']].embedding)
    labels.append(ex[label_field])
    break
  return np.vstack(embedded), np.array(labels)


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


class ClusteringTask(task.MSEBTask):
  """Clustering task."""
  metadata = types.TaskMetadata(
      name='clustering',
      description='Clustering task.',
      reference='TODO',
      type='Clustering',
      category='speech',
      main_score='VMeasure',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[vmeasure_score()],
      eval_splits=['test'],
      eval_langs=['en-US'],
      domains=['speech'],
      task_subtypes=['clustering'],
  )

  def __init__(self, sound_encoder: encoder_lib.SoundEncoder, base_path: str):
    self._base_path = base_path
    self._sound_encoder = sound_encoder

  def setup(self):
    self._sound_encoder.setup()

  def load_data(self):
    yield [0.0], types.SoundContextParams(
        sample_rate=48000, length=16000
    )

  def encode(self, embeddings: task.SoundEmbeddings) -> task.SoundEmbeddings:
    return encode_svq(
        self._base_path,
        self._sound_encoder,
    )

  def evaluate(
      self, embeddings: task.SoundEmbeddings
  ) -> dict[str, list[types.Score]]:
    scores = {}
    for label_field in ['speaker_gender', 'speaker_age', 'speaker_id']:
      embedded, labels = get_examples_svq(
          self._base_path, label_field, embeddings
      )
      clusters = cluster_kmeans(
          embedded, nlabels=len(set(labels)), batch_size=32
      )
      v_measure = sklearn.metrics.v_measure_score(
          labels_true=labels, labels_pred=clusters
      )
      scores[label_field] = [vmeasure_score(v_measure)]
    return scores

  def run(
      self, embeddings: task.SoundEmbeddings
  ) -> dict[str, list[types.Score]]:
    embeddings = self.encode(embeddings)
    return self.evaluate(embeddings)
