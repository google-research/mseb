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

import io
import json
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from mseb import types
from mseb.datasets import fsd50k
import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq
from scipy.io import wavfile

_CC0 = 'http://creativecommons.org/publicdomain/zero/1.0/'
_CC_BY = 'http://creativecommons.org/licenses/by/3.0/'
_CC_BY_NC = 'http://creativecommons.org/licenses/by-nc/3.0/'
_CC_SAMPLING_PLUS = 'http://creativecommons.org/licenses/sampling+/1.0/'


class FSD50KDatasetTest(absltest.TestCase):
  """Tests for the FSD50KDataset class."""

  def setUp(self):
    """Set up a temporary directory and mock data files."""
    super().setUp()
    self.testdata_dir = self.create_tempdir()

    # Create a mock vocabulary.csv file, which the class needs to load.
    labels_dir = os.path.join(self.testdata_dir.full_path, 'labels')
    os.makedirs(labels_dir)
    vocab_data = {
        'index': [0, 1, 2, 3, 4],
        'display_name': [
            'Electric_guitar',
            'Guitar',
            'Plucked_string_instrument',
            'Musical_instrument',
            'Music',
        ],
        'mid': ['/m/02sgy', '/m/0342h', '/m/0fx80y', '/m/04szw', '/m/04rlf'],
    }
    pd.DataFrame(vocab_data).to_csv(
        os.path.join(labels_dir, 'vocabulary.csv'), header=False, index=False
    )
    eval_data = {
        'fname': [37199, 175151],
        'labels': [
            'Electric_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music',
            'Electric_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music',
        ],
        'mids': [
            '/m/02sgy,/m/0342h,/m/0fx80y,/m/04szw,/m/04rlf',
            '/m/02sgy,/m/0342h,/m/0fx80y,/m/04szw,/m/04rlf',
        ],
        'split': ['test', 'test'],
    }
    pd.DataFrame(eval_data).to_csv(
        os.path.join(labels_dir, 'eval.csv'), index=False
    )
    metadata_dir = os.path.join(self.testdata_dir.full_path, 'metadata')
    os.makedirs(metadata_dir)
    with open(
        os.path.join(metadata_dir, 'eval_clips_info_FSD50K.json'), 'w'
    ) as f:
      json.dump(
          {
              '37199': {'license': _CC0},
              '175151': {'license': _CC_BY},
              '999999': {'license': _CC0},
              '888888': {'license': _CC_BY},
              '253463': {'license': _CC_BY_NC},
              '865': {'license': _CC_SAMPLING_PLUS},
          },
          f,
      )
    clips_dir = os.path.join(self.testdata_dir.full_path, 'clips', 'eval')
    os.makedirs(clips_dir)
    wavfile.write(
        os.path.join(clips_dir, '37199.wav'),
        16000,
        np.zeros(16000, dtype=np.float32),
    )
    wavfile.write(
        os.path.join(clips_dir, '175151.wav'),
        16000,
        np.zeros(32000, dtype=np.float32),
    )

  @mock.patch('mseb.utils.download_from_hf')
  def test_loading_and_label_methods(
      self,
      mock_download_from_hf,
  ):
    dataset = fsd50k.FSD50KDataset(
        split='test',
        base_path=self.testdata_dir.full_path,
    )

    mock_download_from_hf.assert_called_once()
    self.assertLen(dataset, 2)
    self.assertListEqual(
        dataset.class_labels,
        [
            'Electric_guitar',
            'Guitar',
            'Plucked_string_instrument',
            'Musical_instrument',
            'Music',
        ],
    )
    task_data = dataset.get_task_data()
    self.assertEqual(
        task_data.iloc[0]['labels'],
        (
            'Electric_guitar,'
            'Guitar,'
            'Plucked_string_instrument,'
            'Musical_instrument,'
            'Music'
        ),
    )
    self.assertEqual(
        task_data.iloc[1]['labels'],
        (
            'Electric_guitar,'
            'Guitar,'
            'Plucked_string_instrument,'
            'Musical_instrument,'
            'Music'
        ),
    )
    sound1 = dataset.get_sound(task_data.iloc[0].to_dict())
    self.assertIsInstance(sound1, types.Sound)
    self.assertEqual(sound1.context.id, '37199')
    self.assertLen(sound1.waveform, 16000)

    sound2 = dataset.get_sound(task_data.iloc[1].to_dict())
    self.assertEqual(sound2.context.id, '175151')
    self.assertLen(sound2.waveform, 32000)

  def test_load_class_labels_without_split_metadata(self):
    vocab_only_dir = self.create_tempdir()
    labels_dir = os.path.join(vocab_only_dir.full_path, 'labels')
    os.makedirs(labels_dir)
    pd.DataFrame({
        'index': [0, 1],
        'display_name': ['Alarm', 'Bark'],
        'mid': ['/m/07pp_mv', '/m/05tny_'],
    }).to_csv(
        os.path.join(labels_dir, 'vocabulary.csv'), header=False, index=False
    )
    self.assertEqual(
        fsd50k.FSD50KDataset.load_class_labels(
            base_path=vocab_only_dir.full_path
        ),
        ['Alarm', 'Bark'],
    )

  @mock.patch.object(fsd50k, '_PARQUET_BATCH_SIZE', 1)
  def test_filter_fn_with_parquet_cache(self):
    pd.DataFrame({
        'fname': [37199, 999999, 175151],
        'labels': ['Music', 'Music', 'Guitar'],
        'waveform': [
            np.ones(160, dtype=np.float32),
            np.ones(320, dtype=np.float32),
            np.ones(480, dtype=np.float32),
        ],
        'sample_rate': [16000, 16000, 16000],
    }).to_parquet(os.path.join(self.testdata_dir.full_path, 'test.parquet'))

    dataset = fsd50k.FSD50KDataset(
        split='test',
        base_path=self.testdata_dir.full_path,
        filter_fn=fsd50k.is_member_of_debug,
    )

    task_data = dataset.get_task_data()
    self.assertEqual(list(task_data['fname']), [37199, 175151])
    sound = dataset.get_sound(task_data.iloc[1].to_dict())
    self.assertEqual(sound.context.id, '175151')
    self.assertLen(sound.waveform, 480)
    self.assertLen(
        fsd50k.FSD50KDataset(
            split='test', base_path=self.testdata_dir.full_path
        ),
        3,
    )

  @mock.patch.object(fsd50k, '_PARQUET_BATCH_SIZE', 1)
  def test_filter_fn_stops_reading_after_last_match(self):
    pd.DataFrame({
        'fname': [37199, 175151, 999999],
        'labels': ['Music', 'Guitar', 'Music'],
        'waveform': [np.ones(160, dtype=np.float32)] * 3,
        'sample_rate': [16000] * 3,
    }).to_parquet(os.path.join(self.testdata_dir.full_path, 'test.parquet'))
    iter_batches = pq.ParquetFile.iter_batches
    num_batches = 0

    def counting_iter_batches(*args, **kwargs):
      nonlocal num_batches
      for batch in iter_batches(*args, **kwargs):
        num_batches += 1
        yield batch

    with mock.patch.object(
        pq.ParquetFile, 'iter_batches', counting_iter_batches
    ):
      dataset = fsd50k.FSD50KDataset(
          split='test',
          base_path=self.testdata_dir.full_path,
          filter_fn=fsd50k.is_member_of_debug,
      )

    self.assertEqual(list(dataset.get_task_data()['fname']), [37199, 175151])
    self.assertEqual(num_batches, 2)

  @mock.patch('mseb.utils.download_from_hf')
  def test_filter_fn_without_parquet_cache(self, _):
    dataset = fsd50k.FSD50KDataset(
        split='test',
        base_path=self.testdata_dir.full_path,
        filter_fn=lambda fname: fname == 175151,
    )

    self.assertEqual(list(dataset.get_task_data()['fname']), [175151])
    # The cache that was written contains all examples.
    self.assertLen(
        fsd50k.FSD50KDataset(
            split='test', base_path=self.testdata_dir.full_path
        ),
        2,
    )

  @mock.patch.object(fsd50k, '_PARQUET_BATCH_SIZE', 1)
  def test_filter_fn_without_matches(self):
    pd.DataFrame({
        'fname': [999999, 888888],
        'labels': ['Music', 'Guitar'],
        'waveform': [np.ones(160, dtype=np.float32)] * 2,
        'sample_rate': [16000] * 2,
    }).to_parquet(os.path.join(self.testdata_dir.full_path, 'test.parquet'))

    with self.assertLogs(level='WARNING'):
      dataset = fsd50k.FSD50KDataset(
          split='test',
          base_path=self.testdata_dir.full_path,
          filter_fn=fsd50k.is_member_of_debug,
      )

    task_data = dataset.get_task_data()
    self.assertEmpty(task_data)
    self.assertIn('fname', task_data.columns)

  def test_default_base_path_resolution(self):
    pd.DataFrame({
        'fname': [37199],
        'labels': ['Music'],
        'waveform': [np.ones(160, dtype=np.float32)],
        'sample_rate': [16000],
    }).to_parquet(os.path.join(self.testdata_dir.full_path, 'test.parquet'))
    with flagsaver.flagsaver(dataset_basepath=None):
      with mock.patch.object(
          fsd50k, 'DEFAULT_BASE_PATH', self.testdata_dir.full_path
      ):
        dataset = fsd50k.FSD50KDataset(split='test')
        self.assertEqual(dataset.base_path, self.testdata_dir.full_path)
        with self.assertRaisesRegex(ValueError, 'basepath must be provided'):
          fsd50k.FSD50KDataset(split='test', commercial_only=False)

  @mock.patch('mseb.utils.download_from_hf')
  def test_default_base_path_missing_cache_raises_without_download(
      self, mock_download
  ):
    empty_dir = self.create_tempdir().full_path
    with flagsaver.flagsaver(dataset_basepath=None):
      with mock.patch.object(fsd50k, 'DEFAULT_BASE_PATH', empty_dir):
        with self.assertRaises(FileNotFoundError):
          fsd50k.FSD50KDataset(split='test')
        mock_download.assert_not_called()

  @mock.patch.object(fsd50k, '_PARQUET_BATCH_SIZE', 1)
  def test_commercial_only_filters_non_commercial_parquet_rows(self):
    pd.DataFrame({
        'fname': [37199, 253463, 865, 175151],
        'labels': ['Music', 'Music', 'Music', 'Guitar'],
        'license': [_CC0, _CC_BY_NC, _CC_SAMPLING_PLUS, _CC_BY],
        'waveform': [np.ones(160, dtype=np.float32)] * 4,
        'sample_rate': [16000] * 4,
    }).to_parquet(os.path.join(self.testdata_dir.full_path, 'test.parquet'))
    # Remove metadata JSON to confirm filtering reads the Parquet `license`
    # column directly without requiring sidecar metadata.
    os.remove(
        os.path.join(
            self.testdata_dir.full_path,
            'metadata',
            'eval_clips_info_FSD50K.json',
        )
    )

    commercial_ds = fsd50k.FSD50KDataset(
        split='test',
        base_path=self.testdata_dir.full_path,
    )
    self.assertEqual(
        list(commercial_ds.get_task_data()['fname']), [37199, 175151]
    )
    self.assertEqual(commercial_ds.metadata.version, fsd50k.COMMERCIAL_VERSION)
    self.assertIn('CC0 1.0 and CC BY 3.0', commercial_ds.metadata.license)

    unfiltered_ds = fsd50k.FSD50KDataset(
        split='test',
        base_path=self.testdata_dir.full_path,
        commercial_only=False,
    )
    self.assertEqual(
        list(unfiltered_ds.get_task_data()['fname']),
        [37199, 253463, 865, 175151],
    )
    self.assertEqual(unfiltered_ds.metadata.version, fsd50k.FULL_VERSION)
    self.assertIn(
        'per-clip Creative Commons licenses', unfiltered_ds.metadata.license
    )

  @mock.patch.object(fsd50k, '_PARQUET_BATCH_SIZE', 1)
  def test_commercial_only_parquet_without_license_or_metadata_raises(self):
    pd.DataFrame({
        'fname': [37199, 253463],
        'labels': ['Music', 'Music'],
        'waveform': [np.ones(160, dtype=np.float32)] * 2,
        'sample_rate': [16000] * 2,
    }).to_parquet(os.path.join(self.testdata_dir.full_path, 'test.parquet'))
    os.remove(
        os.path.join(
            self.testdata_dir.full_path,
            'metadata',
            'eval_clips_info_FSD50K.json',
        )
    )

    with self.assertRaisesRegex(ValueError, 'has no `license` column'):
      fsd50k.FSD50KDataset(
          split='test',
          base_path=self.testdata_dir.full_path,
          commercial_only=True,
      )

    unfiltered_ds = fsd50k.FSD50KDataset(
        split='test',
        base_path=self.testdata_dir.full_path,
        commercial_only=False,
    )
    self.assertEqual(
        list(unfiltered_ds.get_task_data()['fname']), [37199, 253463]
    )

  @mock.patch.object(fsd50k, '_PARQUET_BATCH_SIZE', 1)
  def test_commercial_only_parquet_cache_guard_and_all_match_fast_path(self):
    table = pa.Table.from_pydict(
        {
            'fname': [37199, 175151],
            'labels': ['Music', 'Guitar'],
            'license': [_CC0, _CC_BY],
            'waveform': [[0.1, 0.2], [0.3, 0.4]],
            'sample_rate': [16000, 16000],
        },
        schema=pa.schema(
            [
                ('fname', pa.int64()),
                ('labels', pa.string()),
                ('license', pa.string()),
                ('waveform', pa.list_(pa.float32())),
                ('sample_rate', pa.int64()),
            ],
            metadata={
                fsd50k.COMMERCIAL_ONLY_PARQUET_METADATA_KEY: (
                    fsd50k.COMMERCIAL_ONLY_PARQUET_METADATA_VALUE
                )
            },
        ),
    )
    pq.write_table(
        table, os.path.join(self.testdata_dir.full_path, 'test.parquet')
    )

    with mock.patch.object(
        fsd50k, 'is_commercial_license', wraps=fsd50k.is_commercial_license
    ) as mock_is_commercial:
      commercial_ds = fsd50k.FSD50KDataset(
          split='test',
          base_path=self.testdata_dir.full_path,
          commercial_only=True,
      )
      # Each row is checked once during the initial column scan; per-batch
      # filtering is skipped because all rows match.
      self.assertEqual(mock_is_commercial.call_count, 2)
    self.assertEqual(
        list(commercial_ds.get_task_data()['fname']), [37199, 175151]
    )

    with self.assertRaisesRegex(ValueError, 'commercial-licensed clips'):
      fsd50k.FSD50KDataset(
          split='test',
          base_path=self.testdata_dir.full_path,
          commercial_only=False,
      )

  @mock.patch.object(fsd50k, '_PARQUET_BATCH_SIZE', 1)
  def test_commercial_only_empty_logs_warning(self):
    pd.DataFrame({
        'fname': [253463],
        'labels': ['Music'],
        'license': [_CC_BY_NC],
        'waveform': [np.ones(160, dtype=np.float32)],
        'sample_rate': [16000],
    }).to_parquet(os.path.join(self.testdata_dir.full_path, 'test.parquet'))

    with self.assertLogs(level='WARNING') as logs:
      dataset = fsd50k.FSD50KDataset(
          split='test',
          base_path=self.testdata_dir.full_path,
          commercial_only=True,
      )

    self.assertEmpty(dataset.get_task_data())
    self.assertTrue(
        any('commercial_only=True' in line for line in logs.output),
        msg=logs.output,
    )

  def test_commercial_only_filters_csv_and_caches_license_column(self):
    pd.DataFrame({
        'fname': [37199, 253463, 865, 175151],
        'labels': ['Music', 'Music', 'Music', 'Guitar'],
        'mids': ['/m/04rlf'] * 4,
        'split': ['test'] * 4,
    }).to_csv(
        os.path.join(self.testdata_dir.full_path, 'labels', 'eval.csv'),
        index=False,
    )
    with mock.patch.object(
        fsd50k.utils, 'download_from_hf', autospec=True
    ):
      filtered_ds = fsd50k.FSD50KDataset(
          split='test',
          base_path=self.testdata_dir.full_path,
          filter_fn=lambda fname: fname in (253463, 175151),
      )

    self.assertEqual(list(filtered_ds.get_task_data()['fname']), [175151])
    self.assertEqual(list(filtered_ds.get_task_data().index), [0])
    self.assertNotIn('index', filtered_ds.get_task_data().columns)
    cached = pq.read_table(
        os.path.join(self.testdata_dir.full_path, 'test.parquet')
    )
    self.assertIn('license', cached.schema.names)
    self.assertEqual(
        cached.column('license').to_pylist(),
        [_CC0, _CC_BY_NC, _CC_SAMPLING_PLUS, _CC_BY],
    )

  def test_commercial_only_csv_without_metadata_raises(self):
    os.remove(
        os.path.join(
            self.testdata_dir.full_path,
            'metadata',
            'eval_clips_info_FSD50K.json',
        )
    )
    with mock.patch.object(
        fsd50k.utils, 'download_from_hf', autospec=True
    ):
      with self.assertRaisesRegex(ValueError, 'eval_clips_info_FSD50K.json'):
        fsd50k.FSD50KDataset(
            split='test',
            base_path=self.testdata_dir.full_path,
            commercial_only=True,
        )

  def test_streaming_loads_licenses_only_when_commercial_only(self):
    csv_df = pd.DataFrame({
        'fname': [37199, 253463],
        'labels': ['Music', 'Guitar'],
        'mids': ['/m/04rlf', '/m/0342h'],
    })
    vocab_df = pd.DataFrame({
        'index': [0, 1],
        'display_name': ['Music', 'Guitar'],
        'mid': ['/m/04rlf', '/m/0342h'],
    })
    metadata_bytes = json.dumps({
        '37199': {'license': _CC0},
        '253463': {'license': _CC_BY_NC},
    }).encode('utf-8')

    def fake_read_csv(url, **kwargs):
      del kwargs
      return vocab_df if url.endswith('vocabulary.csv') else csv_df

    unused_dir = self.create_tempdir().full_path
    with (
        mock.patch.object(pd, 'read_csv', side_effect=fake_read_csv),
        mock.patch.object(
            fsd50k.urllib.request,
            'urlopen',
            return_value=io.BytesIO(metadata_bytes),
        ) as mock_urlopen,
    ):
      commercial_ds = fsd50k.FSD50KDataset(
          split='test',
          base_path=unused_dir,
          streaming=True,
          commercial_only=True,
      )
      self.assertEqual(list(commercial_ds.get_task_data()['fname']), [37199])
      mock_urlopen.assert_called_once()

      mock_urlopen.reset_mock()
      full_ds = fsd50k.FSD50KDataset(
          split='test',
          base_path=unused_dir,
          streaming=True,
          commercial_only=False,
      )
      self.assertEqual(
          list(full_ds.get_task_data()['fname']), [37199, 253463]
      )
      mock_urlopen.assert_not_called()


class DebugIdsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='cc0_http', license_url=_CC0, expected=True),
      dict(
          testcase_name='cc0_https',
          license_url='https://creativecommons.org/publicdomain/zero/1.0/',
          expected=True,
      ),
      dict(testcase_name='cc_by_3_http', license_url=_CC_BY, expected=True),
      dict(
          testcase_name='cc_by_3_https',
          license_url='https://creativecommons.org/licenses/by/3.0/',
          expected=True,
      ),
      dict(
          testcase_name='cc_by_4_http',
          license_url='http://creativecommons.org/licenses/by/4.0/',
          expected=True,
      ),
      dict(
          testcase_name='cc_by_4_https_padded',
          license_url='  https://creativecommons.org/licenses/by/4.0/ \n',
          expected=True,
      ),
      dict(
          testcase_name='cc_by_nc',
          license_url=_CC_BY_NC,
          expected=False,
      ),
      dict(
          testcase_name='cc_sampling_plus',
          license_url=_CC_SAMPLING_PLUS,
          expected=False,
      ),
      dict(testcase_name='empty_str', license_url='', expected=False),
      dict(testcase_name='none_value', license_url=None, expected=False),
      dict(
          testcase_name='unknown_url',
          license_url='http://example.com/unknown',
          expected=False,
      ),
  )
  def test_is_commercial_license(self, license_url, expected: bool):
    self.assertEqual(fsd50k.is_commercial_license(license_url), expected)

  def test_is_member_of_debug_true(self):
    for debug_id in ('37199', '175151', '329838', '1277'):
      self.assertTrue(fsd50k.is_member_of_debug(debug_id))
      self.assertTrue(fsd50k.is_member_of_debug(int(debug_id)))

  def test_is_member_of_debug_false(self):
    self.assertFalse(fsd50k.is_member_of_debug('253463'))
    self.assertFalse(fsd50k.is_member_of_debug('345111'))
    self.assertFalse(fsd50k.is_member_of_debug('160826'))
    self.assertFalse(fsd50k.is_member_of_debug('420945'))
    self.assertFalse(fsd50k.is_member_of_debug('non_existent_id'))
    self.assertFalse(fsd50k.is_member_of_debug(''))


if __name__ == '__main__':
  absltest.main()
