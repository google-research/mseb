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

from collections.abc import Sequence
import dataclasses
from os import path

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
from mseb import beam as mseb_beam
from mseb import record
from mseb import types
import numpy as np

_RUNNERS = (
    ('direct', 'direct'),
)


def _make_runner(runner_name: str) -> beam.runners.PipelineRunner:
  if runner_name == 'direct':
    return beam.runners.direct.direct_runner.DirectRunner()
  raise ValueError(f'Unsupported runner: {runner_name}')


def _sample_items() -> list[types.MultiModalObject]:
  return [
      types.TextEmbedding(
          embedding=np.array([[1.0, 2.0]], dtype=np.float32),
          spans=np.array([[0, 5]], dtype=np.int32),
          context=types.TextContextParams(id='item_1'),
      ),
      types.TextEmbedding(
          embedding=np.array([[3.0, 4.0]], dtype=np.float32),
          spans=np.array([[0, 5]], dtype=np.int32),
          context=types.TextContextParams(id='item_2'),
      ),
      types.Sound(
          waveform=np.array([0.1, -0.2, 0.3], dtype=np.float32),
          context=types.SoundContextParams(
              id='sound_1', sample_rate=16000, length=3
          ),
      ),
  ]


def _assert_objects_equal(expected: Sequence[types.MultiModalObject]):
  expected_by_id = {
      obj.context.id: (type(obj), dataclasses.asdict(obj)) for obj in expected
  }

  def _check(actual):
    actual_by_id = {
        obj.context.id: (type(obj), dataclasses.asdict(obj)) for obj in actual
    }
    np.testing.assert_equal(actual_by_id, expected_by_id)

  return _check


class BeamTest(parameterized.TestCase):

  @parameterized.named_parameters(*_RUNNERS)
  def test_write_to_bagz(self, runner_name: str):
    basedir = self.create_tempdir().full_path
    prefix = path.join(basedir, 'beam_write')
    items = _sample_items()

    with beam.Pipeline(runner=_make_runner(runner_name)) as pipeline:
      _ = (
          pipeline
          | beam.Create(items)
          | mseb_beam.WriteToBagz(prefix, num_shards=2)
      )

    loaded = record.play(prefix)
    _assert_objects_equal(items)(loaded.values())

  @parameterized.named_parameters(*_RUNNERS)
  def test_read_from_bagz(self, runner_name: str):
    basedir = self.create_tempdir().full_path
    prefix = path.join(basedir, 'beam_read')
    items = _sample_items()
    record.etch(prefix, items, num_shards=2)

    with beam.Pipeline(runner=_make_runner(runner_name)) as pipeline:
      read_items = pipeline | mseb_beam.ReadFromBagz(prefix)
      util.assert_that(read_items, _assert_objects_equal(items))

  @parameterized.named_parameters(*_RUNNERS)
  def test_write_and_read_roundtrip(self, runner_name: str):
    basedir = self.create_tempdir().full_path
    prefix = path.join(basedir, 'beam_roundtrip')
    items = _sample_items()

    with beam.Pipeline(runner=_make_runner(runner_name)) as write_pipeline:
      _ = (
          write_pipeline
          | beam.Create(items)
          | mseb_beam.WriteToBagz(prefix, num_shards=2)
      )

    with beam.Pipeline(runner=_make_runner(runner_name)) as read_pipeline:
      read_items = read_pipeline | mseb_beam.ReadFromBagz(f'{prefix}@*.bagz')
      util.assert_that(read_items, _assert_objects_equal(items))


if __name__ == '__main__':
  absltest.main()
