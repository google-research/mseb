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
from collections.abc import Iterable, Iterator, Sequence
from concurrent import futures
import math
import os
import pickle

from absl import flags
from absl import logging
import apache_beam as beam
from apache_beam.utils import shared
from mseb import encoder as encoder_lib
from mseb import types
import tensorflow as tf
import tqdm



cpu_resource_hints = dict()
tqdm = tqdm.tqdm

Encoder = encoder_lib.MultiModalEncoder


RUNNER_CACHE_BASEPATH = flags.DEFINE_string(
    'runner_cache_basepath',
    None,
    'Base path for the runner cache.',
)


class EncoderRunner(abc.ABC):
  """Interface for running a MultiModalEncoder to get a cache of embeddings.

  Usage:

  # Create a runner, encode task multimodal inputs, and run on the embeddings.
  runner = ...
  cache = runner.run(task.sounds())
  task.compute_scores(cache)
  """

  def __init__(self, encoder: Encoder):
    self._encoder = encoder

  @abc.abstractmethod
  def run(
      self,
      elements: Iterable[types.MultiModalObject],
      output_name: str = 'embeddings',
  ) -> types.MultiModalEmbeddingCache:
    """Encode the given multimodal objects and return a cache of embeddings."""


class DirectRunner(EncoderRunner):
  """Simple runner that encodes locally, stores results in a dict in-memory."""

  embeddings: dict[str, types.MultiModalEmbedding] = {}

  def __init__(
      self,
      batch_size=1,
      num_threads: int = 1,
      output_path: str | None = None,
      **kwargs
  ):
    super().__init__(**kwargs)
    self._batch_size = batch_size
    self._num_threads = num_threads
    self._output_path = output_path

  def _batch_elements(
      self, elements: Iterable[types.MultiModalObject]
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

  def _encode_batch(
      self, batch: Sequence[types.MultiModalObject]
  ) -> Sequence[tuple[str, types.MultiModalEmbedding]]:
    encoded = self._encoder.encode(batch)
    return [
        (element.context.id, embedding)
        for element, embedding in zip(batch, encoded)
    ]

  def run(
      self,
      elements: Iterable[types.MultiModalObject],
      output_name: str = 'embeddings',
  ) -> types.MultiModalEmbeddingCache:
    output_prefix = (
        os.path.join(self._output_path, output_name)
        if self._output_path
        else None
    )
    if output_prefix is not None:
      try:
        return load_embeddings(output_prefix)
      except FileNotFoundError:
        pass

    embeddings = {}
    self._encoder.setup()
    if self._num_threads > 1:
      with futures.ThreadPoolExecutor(
          max_workers=self._num_threads
      ) as executor:
        for element_id_and_embedding_batch in tqdm(
            executor.map(self._encode_batch, self._batch_elements(elements)),
            desc='Encoding batches of elements',
        ):
          for element_id, embedding in element_id_and_embedding_batch:
            embeddings[element_id] = embedding
    else:
      for batch in tqdm(
          self._batch_elements(elements),
          desc='Encoding batches of elements',
      ):
        encoded = self._encoder.encode(batch)
        for element, embedding in zip(batch, encoded):
          embeddings[element.context.id] = embedding

    if output_prefix is not None:
      save_embeddings(output_prefix, embeddings)

    return embeddings


class EncodeDoFn(beam.DoFn):
  """A DoFn that wraps an Encoder."""

  def __init__(self, encoder: Encoder, batch_size: int = 0):
    self._encoder = encoder
    self._shared_handle = shared.Shared()
    self._batch_size = batch_size
    self._batch = []

  def setup(self):
    self._encoder: Encoder = self._shared_handle.acquire(lambda: self._encoder)
    self._encoder.setup()

  def start_bundle(self) -> None:
    """Resets the batch."""
    self._batch = []

  def process(self, element: types.MultiModalObject):
    self._batch.append(element)
    if len(self._batch) == self._batch_size:
      embeds = self._encoder.encode(self._batch)
      for embed in embeds:
        beam.metrics.Metrics.counter('EncodeDoFn', 'num_embeddings').inc()
        yield embed
      self._batch = []

  def finish_bundle(self) -> Iterator[beam.utils.windowed_value.WindowedValue]:
    """Writes the remaining batch."""
    if self._batch:
      embeds = self._encoder.encode(self._batch)
      for embed in embeds:
        beam.metrics.Metrics.counter('EncodeDoFn', 'num_embeddings').inc()
        yield beam.transforms.window.GlobalWindows.windowed_value(embed)
    self._batch = []


def load_embeddings(output_prefix: str) -> types.MultiModalEmbeddingCache:
  """Loads embeddings from TFRecord files into a dict."""
  logging.info('Loading embeddings from %s', output_prefix + '*')
  embeddings = {}
  file_glob = output_prefix + '*'
  output_files = tf.io.gfile.glob(file_glob)
  if not output_files:
    raise FileNotFoundError(f'No files found matching {file_glob}')
  for filename in output_files:
    # But don't know how to read back from TFRecord except via tf.data.
    dataset = tf.data.TFRecordDataset(filename)
    for record in dataset:
      embedding: types.MultiModalEmbedding = pickle.loads(record.numpy())
      embeddings[embedding.context.id] = embedding
  logging.info(
      'Loaded %d embeddings from %s', len(embeddings), output_prefix + '*'
  )
  return embeddings


def save_embeddings(
    output_prefix: str,
    embeddings: types.MultiModalEmbeddingCache,
    shard_size: int = 25_000,
):
  """Saves embeddings from a dict into to TFRecord files."""
  num_embeddings = len(embeddings)
  num_shards = math.ceil(num_embeddings / shard_size)
  logging.info('Saving embeddings to %s', f'{output_prefix}@{num_shards}')
  tf.io.gfile.makedirs(os.path.dirname(output_prefix))
  embed_it = iter(embeddings.values())
  for shard_id in range(num_shards):
    with tf.io.TFRecordWriter(
        f'{output_prefix}-{shard_id:05d}-of-{num_shards:05d}'
    ) as writer:
      for _ in range(
          shard_id * shard_size,
          min(num_embeddings, (shard_id + 1) * shard_size),
      ):
        record_bytes = pickle.dumps(next(embed_it))
        writer.write(record_bytes)  # pytype: disable=attribute-error


class BeamRunner(EncoderRunner):
  """Runner that encodes using beam, then loads results into in-memory dict."""

  def __init__(
      self,
      output_path: str,
      runner: beam.runners.PipelineRunner,
      batch_size: int = 1,
      accelerator: str | None = None,
      **kwargs
  ):
    """Initializes the BeamRunner.

    Args:
      output_path: The root directory to write temporary output files to.
      runner: The beam pipeline runner to use.
      batch_size: The batch size to use for encoding.
      accelerator: The accelerator to use for encoding.
      **kwargs: Additional keyword arguments for the base class.
    """
    super().__init__(**kwargs)
    self._output_path = output_path
    self._runner = runner
    self._batch_size = batch_size
    self._accelerator = accelerator

  def run(
      self,
      elements: Iterable[types.MultiModalObject],
      output_name: str = 'embeddings',
  ) -> types.MultiModalEmbeddingCache:
    output_prefix = os.path.join(self._output_path, output_name)
    try:
      logging.info('Loading embeddings from %s', output_prefix)
      return load_embeddings(output_prefix)
    except FileNotFoundError:
      logging.info('No embeddings found at %s', output_prefix)
      pass

    resource_hints = cpu_resource_hints

    pipeline = beam.Pipeline(runner=self._runner)
    _ = (
        pipeline
        | 'ReadExamples' >> beam.Create(list(elements))
        | 'Encode'
        >> beam.ParDo(
            EncodeDoFn(self._encoder, batch_size=self._batch_size)
        ).with_resource_hints(**resource_hints)
        | 'Serialize' >> beam.Map(pickle.dumps)
        # Using TFRecord because it's available as standard beam io.
        | 'WriteTFRecord' >> beam.io.tfrecordio.WriteToTFRecord(output_prefix)
    )
    logging.info('Running pipeline')
    pipeline.run().wait_until_finish()
    logging.info('Loading embeddings from %s', output_prefix)
    return load_embeddings(output_prefix)
