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

"""Runners for executing encoders and storing embeddings."""

import abc
from concurrent import futures
import os
import pickle
from typing import Iterable, overload

import apache_beam as beam
from mseb import encoder as encoder_lib
from mseb import types
import tensorflow as tf


Encoder = encoder_lib.SoundEncoder | encoder_lib.TextEncoder
Embeddings = types.SoundEmbedding | types.TextEmbeddings


class EncoderRunner(abc.ABC):
  """Interface for running a SoundEncoder to get a cache of embeddings.

  Usage:

  runner = ... # Create a runner, encode task sounds, and run on the embeddings.
  cache = runner.run(task.sounds())
  task.compute_scores(cache)
  """

  def __init__(self, encoder: Encoder):
    self._encoder = encoder

  @overload
  @abc.abstractmethod
  def run(self, elements: Iterable[types.Sound]) -> types.SoundEmbeddingCache:
    ...

  @overload
  @abc.abstractmethod
  def run(self, elements: Iterable[types.Text]) -> types.TextEmbeddingCache:
    ...

  @abc.abstractmethod
  def run(
      self, elements: Iterable[types.Sound] | Iterable[types.Text]
  ) -> types.EmbeddingCache:
    """Encode the given sounds/texts and return a cache of the embeddings."""


class DirectRunner(EncoderRunner):
  """Simple runner that encodes locally, stores results in a dict in-memory."""

  embeddings: (
      dict[str, types.SoundEmbedding] | dict[str, types.TextEmbeddings]
  ) = {}

  def __init__(self, batch_size=0, num_threads: int = 1, **kwargs):
    super().__init__(**kwargs)
    self._batch_size = batch_size
    self._num_threads = num_threads

  def _batch_elements(
      self, elements: Iterable[types.Sound] | Iterable[types.Text]
  ):
    """Yields batches of elements (sounds or texts)."""
    batch = []
    for element in elements:
      batch.append(element)
      if len(batch) == self._batch_size:
        yield batch
        batch = []
    if batch:
      # TODO(tombagby): What do we do about short batches? Is this well defined?
      yield batch

  @overload
  def _encode_element(
      self, element: types.Sound
  ) -> tuple[str, types.SoundEmbedding]:
    ...

  @overload
  def _encode_element(
      self, element: types.Text
  ) -> tuple[str, types.TextEmbeddings]:
    ...

  def _encode_element(
      self, element: types.Sound | types.Text
  ) -> tuple[str, types.SoundEmbedding] | tuple[str, types.TextEmbeddings]:
    return element.context.id, self._encoder.encode(element)

  def run(
      self, elements: Iterable[types.Sound] | Iterable[types.Text]
  ) -> types.EmbeddingCache:
    embeddings = {}
    self._encoder.setup()
    if self._num_threads > 1:
      with futures.ThreadPoolExecutor(
          max_workers=self._num_threads
      ) as executor:
        for element_id, embedding in executor.map(
            self._encode_element, elements
        ):
          embeddings[element_id] = embedding
    elif self._batch_size > 0:
      for batch in self._batch_elements(elements):
        encoded = self._encoder.encode_batch(batch)
        for element, embedding in zip(batch, encoded):
          embeddings[element.context.id] = embedding
    else:
      for element in elements:
        embeddings[element.context.id] = self._encoder.encode(element)

    return embeddings


class EncodeDoFn(beam.DoFn):
  """A DoFn that wraps an Encoder."""

  def __init__(self, encoder: Encoder):
    self._encoder = encoder

  def setup(self):
    self._encoder.setup()

  def process(self, element: types.Sound | types.Text):
    yield self._encoder.encode(element)


class BeamRunner(EncoderRunner):
  """Runner that encodes using beam, then loads results into in-memory dict."""

  def __init__(
      self, output_path: str, runner: beam.runners.PipelineRunner, **kwargs
  ):
    """Initializes the BeamRunner.

    Args:
      output_path: The root directory to write temporary output files to.
      runner: The beam pipeline runner to use.
      **kwargs: Additional keyword arguments for the base class.
    """
    super().__init__(**kwargs)
    self._output_path = output_path
    self._runner = runner

  def _load_embeddings(self, output_prefix: str) -> types.EmbeddingCache:
    """Loads embeddings from TFRecord files into a dict."""
    embeddings = {}
    output_files = tf.io.gfile.glob(output_prefix + '*')
    for filename in output_files:
      # But don't know how to read back from TFRecord except via tf.data.
      dataset = tf.data.TFRecordDataset(filename)
      for record in dataset:
        embedding: Embeddings = pickle.loads(record.numpy())
        embeddings[embedding.context.id] = embedding
    return embeddings

  def run(
      self, elements: Iterable[types.Sound] | Iterable[types.Text]
  ) -> types.EmbeddingCache:
    output_prefix = os.path.join(self._output_path, 'embeddings')
    try:
      return self._load_embeddings(output_prefix)
    except FileNotFoundError:
      pass

    with beam.Pipeline(runner=self._runner) as root:
      _ = (
          root
          | 'ReadExamples' >> beam.Create(list(elements))
          | 'Encode' >> beam.ParDo(EncodeDoFn(self._encoder))
          | 'Serialize' >> beam.Map(pickle.dumps)
          # Using TFRecord because it's available as standard beam io.
          | 'WriteTFRecord' >> beam.io.tfrecordio.WriteToTFRecord(output_prefix)
      )

    return self._load_embeddings(output_prefix)
