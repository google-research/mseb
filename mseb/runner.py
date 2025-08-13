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
import os
import pickle
from typing import Iterable

import apache_beam as beam
from mseb import encoder as encoder_lib
from mseb import types
import tensorflow as tf


class EncoderRunner(abc.ABC):
  """Interface for running a SoundEncoder to get a cache of embeddings.

  Usage:

  runner = ... # Create a runner, encode task sounds, and run on the embeddings.
  cache = runner.run(task.sounds())
  task.compute_scores(cache)
  """

  def __init__(self, sound_encoder: encoder_lib.SoundEncoder):
    self._encoder = sound_encoder

  @abc.abstractmethod
  def run(self, sounds: Iterable[types.Sound]) -> types.SoundEmbeddingCache:
    """Encode the given sounds and return a cache of the embeddings."""


class DirectRunner(EncoderRunner):
  """Simple runner that encodes locally, stores results in a dict in-memory."""

  embeddings: dict[str, types.SoundEmbedding] = {}

  def __init__(self, batch_size=0, **kwargs):
    super().__init__(**kwargs)
    self._batch_size = batch_size

  def _batch_sounds(self, sounds: Iterable[types.Sound]):
    """Yields batches of sounds."""
    batch = []
    for sound in sounds:
      batch.append(sound)
      if len(batch) == self._batch_size:
        yield batch
        batch = []
    if batch:
      # TODO(tombagby): What do we do about short batches? Is this well defined?
      yield batch

  def run(self, sounds: Iterable[types.Sound]) -> types.SoundEmbeddingCache:
    embeddings = {}
    self._encoder.setup()
    if self._batch_size > 0:
      for batch in self._batch_sounds(sounds):
        encoded = self._encoder.encode_batch(batch)
        for sound, embedding in zip(batch, encoded):
          embeddings[sound.context.sound_id] = embedding
    else:
      for sound in sounds:
        embeddings[sound.context.sound_id] = self._encoder.encode(sound)

    for sound in sounds:
      embeddings[sound.context.sound_id] = self._encoder.encode(sound)

    return embeddings


class EncodeDoFn(beam.DoFn):
  """A DoFn that wraps a SoundEncoder."""

  def __init__(self, encoder: encoder_lib.SoundEncoder):
    self._encoder = encoder

  def setup(self):
    self._encoder.setup()

  def process(self, element: types.Sound):
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

  def run(self, sounds: Iterable[types.Sound]) -> types.SoundEmbeddingCache:
    output_prefix = os.path.join(self._output_path, 'embeddings')
    with beam.Pipeline(runner=self._runner) as root:
      _ = (
          root
          | 'ReadExamples' >> beam.Create(list(sounds))
          | 'Encode' >> beam.ParDo(EncodeDoFn(self._encoder))
          | 'Serialize' >> beam.Map(pickle.dumps)
          # Using TFRecord because it's available as standard beam io.
          | 'WriteTFRecord' >> beam.io.tfrecordio.WriteToTFRecord(output_prefix)
      )

    embeddings = {}
    output_files = tf.io.gfile.glob(output_prefix + '*')
    for filename in output_files:
      # But don't know how to read back from TFRecord except via tf.data.
      dataset = tf.data.TFRecordDataset(filename)
      for record in dataset:
        embedding: types.SoundEmbedding = pickle.loads(record.numpy())
        embeddings[embedding.context.sound_id] = embedding
    return embeddings
