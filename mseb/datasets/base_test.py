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

"""Tests for MSEB dataset base class and Beam sound-loading helpers."""

import copy
from typing import Any, Mapping
from unittest import mock

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import test_pipeline
from apache_beam.testing import util as beam_util
from mseb import types
from mseb.datasets import base
import numpy as np
import pandas as pd


class _FakeDataset(base.MsebDataset):
  """Minimal concrete MsebDataset for unit testing."""

  def __init__(self, count: int = 3):
    super().__init__(base_path='/tmp/fake', split='test')
    self._df = pd.DataFrame({
        'id': [f'utt_{i}' for i in range(count)],
        'task': ['sub_a' if i % 2 == 0 else 'sub_b' for i in range(count)],
        'value': list(range(count)),
    })

  @property
  def metadata(self) -> types.DatasetMetadata:
    return types.DatasetMetadata(
        name='fake',
        description='Fake dataset.',
        homepage='',
        version='1.0',
        license='Apache-2.0',
        mseb_tasks=['transcription'],
    )

  def __len__(self) -> int:
    return len(self._df)

  def get_task_data(
      self, task_name: str | None = None, dtype: Mapping[str, Any] | None = None
  ) -> pd.DataFrame:
    del dtype
    if task_name is not None:
      return self._df[self._df['task'] == task_name]
    return self._df

  def get_sound(self, record: dict[str, Any]) -> types.Sound:
    val = int(record['value'])
    waveform: Any = np.full(8, val, dtype=np.int16)
    context = types.SoundContextParams(
        id=str(record['id']),
        sample_rate=16000,
        length=len(waveform),
    )
    return types.Sound(waveform=waveform, context=context)


class _ThreadedFakeDataset(_FakeDataset):
  beam_load_threads = 4


class MsebDatasetBaseTest(absltest.TestCase):

  def test_threaded_load_sounds_dofn_lifecycle_and_fallback(self):
    dataset = _FakeDataset(count=3)
    dofn = base.ThreadedLoadSoundsDoFn(dataset, max_workers=4)
    self.assertIsNone(dofn._executor)

    fallback = list(dofn.process([{'id': 'utt_0', 'value': 0}]))
    self.assertLen(fallback, 1)
    self.assertEqual(fallback[0].context.id, 'utt_0')
    self.assertIsNone(dofn._executor)

    dofn.setup()
    executor = dofn._executor
    self.assertIsNotNone(executor)
    dofn.setup()
    self.assertIs(dofn._executor, executor)
    self.assertIsNone(copy.deepcopy(dofn)._executor)

    batch = list(
        dofn.process([{'id': 'utt_1', 'value': 1}, {'id': 'utt_2', 'value': 2}])
    )
    self.assertEqual([s.context.id for s in batch], ['utt_1', 'utt_2'])

    with mock.patch.object(
        executor, 'shutdown', wraps=executor.shutdown
    ) as mock_shutdown:
      dofn.teardown()
      mock_shutdown.assert_called_once_with(wait=True)
    self.assertIsNone(dofn._executor)

  def test_get_task_sounds_beam_single_threaded_and_task_filtering(self):
    dataset = _FakeDataset(count=4)
    with test_pipeline.TestPipeline() as p:
      pcoll = (
          p
          | dataset.get_task_sounds_beam(task_name='sub_a')
          | beam.Map(lambda s: s.context.id)
      )
      beam_util.assert_that(pcoll, beam_util.equal_to(['utt_0', 'utt_2']))

  def test_get_task_sounds_beam_threaded(self):
    dataset = _ThreadedFakeDataset(count=5)
    with test_pipeline.TestPipeline() as p:
      pcoll = (
          p
          | dataset.get_task_sounds_beam()
          | beam.Map(lambda s: s.context.id)
      )
      beam_util.assert_that(
          pcoll, beam_util.equal_to([f'utt_{i}' for i in range(5)])
      )


if __name__ == '__main__':
  absltest.main()
