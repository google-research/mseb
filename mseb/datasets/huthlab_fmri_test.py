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

"""Tests for huthlab_fmri dataset loader."""

import email.message
import io
import os
import re
import struct
from unittest import mock
import urllib.request
import urllib.response

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import joblib
from mseb import types
from mseb import utils
from mseb.datasets import huthlab_fmri
import numpy as np
from scipy.io import wavfile


class _FakeUrlResponse(urllib.response.addinfourl):

  def __init__(self, body: bytes, url: str, code: int):
    hdrs = email.message.Message()
    super().__init__(io.BytesIO(body), hdrs, url, code=code)
    self.msg = 'OK'


class _ScriptedRangeHandler(
    urllib.request.HTTPHandler, urllib.request.HTTPSHandler
):

  def __init__(self, responder):
    urllib.request.HTTPHandler.__init__(self)
    urllib.request.HTTPSHandler.__init__(self)
    self._responder = responder
    self.ranges: list[str] = []

  def https_open(self, req: urllib.request.Request):
    self.ranges.append(req.get_header('Range') or '')
    return self._responder(req)


class HuthLabDatasetTest(parameterized.TestCase):
  """Tests for the HuthLabDataset class."""

  def setUp(self):
    super().setUp()
    self.testdata_dir = self.create_tempdir()
    self.base_path = self.testdata_dir.full_path

    # Create directory structure.
    os.makedirs(os.path.join(self.base_path, 'responses'))
    os.makedirs(os.path.join(self.base_path, 'stimuli'))

    self.n_voxels = 10
    self.n_trs_per_story = 50

    # The on-disk .jbl file is a dict mapping story_name -> (T, V) array.
    # File is named UT{subject}_responses.jbl (e.g. UTS03_responses.jbl).
    np.random.seed(42)
    self.mock_resp = {}
    for story in list(huthlab_fmri.TRAIN_STORIES['S03']) + list(
        huthlab_fmri.TEST_STORIES['S03']
    ):
      self.mock_resp[story] = np.random.randn(
          self.n_trs_per_story, self.n_voxels
      ).astype(np.float32)

    joblib.dump(
        self.mock_resp,
        os.path.join(self.base_path, 'responses', 'UTS03_responses.jbl'),
    )

  def _make_dataset(self, **kwargs):
    return huthlab_fmri.HuthLabDataset(base_path=self.base_path, **kwargs)

  # --- Init tests ---

  def test_init_with_base_path(self):
    dataset = self._make_dataset()
    self.assertEqual(dataset.base_path, self.base_path)
    self.assertEqual(dataset.subject, 'S03')

  def test_init_with_custom_subject(self):
    dataset = self._make_dataset(subject='S01')
    self.assertEqual(dataset.subject, 'S01')

  def test_init_default_tr_duration(self):
    dataset = self._make_dataset()
    self.assertEqual(dataset.tr_duration, huthlab_fmri.DEFAULT_TR_DURATION)

  def test_init_custom_tr_duration(self):
    dataset = self._make_dataset(tr_duration=1.5)
    self.assertEqual(dataset.tr_duration, 1.5)

  def test_init_requires_base_path(self):
    with flagsaver.flagsaver(dataset_basepath=None, huthlab_data_path=None):
      with mock.patch.object(huthlab_fmri, 'DEFAULT_BASE_PATH', None):
        with self.assertRaises(ValueError):
          huthlab_fmri.HuthLabDataset(base_path=None)

  def test_default_base_path_resolution(self):
    with flagsaver.flagsaver(dataset_basepath=None, huthlab_data_path=None):
      with mock.patch.object(
          huthlab_fmri, 'DEFAULT_BASE_PATH', self.base_path
      ):
        dataset = huthlab_fmri.HuthLabDataset(base_path=None)
        self.assertEqual(dataset.base_path, self.base_path)

  # --- load_responses tests ---

  def test_load_responses_returns_dict(self):
    dataset = self._make_dataset()
    resp = dataset.load_responses()
    self.assertIsInstance(resp, dict)

  def test_load_responses_contains_all_stories(self):
    dataset = self._make_dataset()
    resp = dataset.load_responses()
    all_stories = list(huthlab_fmri.TRAIN_STORIES['S03']) + list(
        huthlab_fmri.TEST_STORIES['S03']
    )
    for story in all_stories:
      self.assertIn(story, resp)

  def test_load_responses_values_are_arrays(self):
    dataset = self._make_dataset()
    resp = dataset.load_responses()
    for _, arr in resp.items():
      self.assertIsInstance(arr, np.ndarray)
      self.assertEqual(arr.shape, (self.n_trs_per_story, self.n_voxels))

  def test_load_responses_matches_mock(self):
    dataset = self._make_dataset()
    resp = dataset.load_responses()
    for story in self.mock_resp:
      np.testing.assert_array_equal(resp[story], self.mock_resp[story])

  def test_load_responses_cached(self):
    """The underlying data is loaded only once (self._resp is cached)."""
    dataset = self._make_dataset()
    resp1 = dataset.load_responses()
    resp2 = dataset.load_responses()
    self.assertIs(resp1, resp2)

  # --- load_train_responses / load_test_responses tests ---

  def test_load_train_responses_shape(self):
    dataset = self._make_dataset()
    resp = dataset.load_train_responses()
    train_stories = dataset.get_train_stories()
    expected_trs = len(train_stories) * self.n_trs_per_story
    self.assertEqual(resp.shape, (expected_trs, self.n_voxels))

  def test_load_train_responses_values(self):
    dataset = self._make_dataset()
    resp = dataset.load_train_responses()
    train_stories = dataset.get_train_stories()
    expected = np.vstack([self.mock_resp[s] for s in train_stories])
    np.testing.assert_array_equal(resp, expected)

  def test_load_test_responses_shape(self):
    dataset = self._make_dataset()
    resp = dataset.load_test_responses()
    test_stories = dataset.get_test_stories()
    # load_test_responses slices [40:] from each story.
    trs_per_test_story = self.n_trs_per_story - 40
    expected_trs = len(test_stories) * trs_per_test_story
    self.assertEqual(resp.shape, (expected_trs, self.n_voxels))

  def test_load_test_responses_values(self):
    dataset = self._make_dataset()
    resp = dataset.load_test_responses()
    test_stories = dataset.get_test_stories()
    # load_test_responses slices [40:] from each story.
    expected = np.vstack([self.mock_resp[s][40:] for s in test_stories])
    np.testing.assert_array_equal(resp, expected)

  # --- Story metadata tests ---

  def test_get_train_stories_s03(self):
    dataset = self._make_dataset()
    train_stories = dataset.get_train_stories()
    self.assertIs(train_stories, huthlab_fmri.TRAIN_STORIES['S03'])
    self.assertIn('adollshouse', train_stories)

  def test_get_train_stories_s01(self):
    dataset = self._make_dataset(subject='S01')
    train_stories = dataset.get_train_stories()
    self.assertIs(train_stories, huthlab_fmri.TRAIN_STORIES['S01'])

  def test_get_train_stories_per_subject(self):
    """TRAIN_STORIES is keyed by subject; S02 and S03 share the same list."""
    self.assertIs(
        huthlab_fmri.TRAIN_STORIES['S02'],
        huthlab_fmri.TRAIN_STORIES['S03'],
    )
    self.assertIsNot(
        huthlab_fmri.TRAIN_STORIES['S01'],
        huthlab_fmri.TRAIN_STORIES['S02'],
    )

  def test_s02_has_extra_stories(self):
    """S02/S03 have stories not in S01."""
    s02_only = set(huthlab_fmri.TRAIN_STORIES['S02']) - set(
        huthlab_fmri.TRAIN_STORIES['S01']
    )
    self.assertNotEmpty(s02_only)

  def test_get_test_stories_s03(self):
    dataset = self._make_dataset()
    test_stories = dataset.get_test_stories()
    self.assertIs(test_stories, huthlab_fmri.TEST_STORIES['S03'])
    self.assertIn('wheretheressmoke', test_stories)

  def test_get_test_stories_all_subjects_same(self):
    """All subjects share the same test stories."""
    for subject in huthlab_fmri.SUBJECTS:
      self.assertEqual(
          huthlab_fmri.TEST_STORIES[subject],
          ('wheretheressmoke', 'onapproachtopluto', 'fromboyhoodtofatherhood'),
      )

  def test_train_test_no_overlap(self):
    """Train and test stories should not overlap for any subject."""
    for subject in huthlab_fmri.SUBJECTS:
      train = set(huthlab_fmri.TRAIN_STORIES[subject])
      test = set(huthlab_fmri.TEST_STORIES[subject])
      self.assertEmpty(train & test)

  # --- Audio path tests ---

  def test_get_audio_path(self):
    dataset = self._make_dataset()
    path = dataset.get_audio_path('wheretheressmoke')
    expected = os.path.join(self.base_path, 'stimuli', 'wheretheressmoke.wav')
    self.assertEqual(path, expected)

  def test_get_sound(self):
    wav_path = os.path.join(self.base_path, 'stimuli', 'itsabox.wav')
    wavfile.write(wav_path, 16000, np.zeros(16000, dtype=np.int16))
    dataset = self._make_dataset()
    sound = dataset.get_sound('itsabox')
    self.assertIsInstance(sound, types.Sound)
    self.assertEqual(sound.context.id, 'itsabox')
    self.assertEqual(sound.context.sample_rate, 16000)
    self.assertEqual(sound.context.length, 16000)
    self.assertEqual(sound.context.waveform_end_second, 1.0)
    self.assertEqual(sound.context.language, 'en')
    self.assertEqual(sound.context.language_name, 'English')

  def test_get_sound_fetches_prefix_via_http_range_when_missing_locally(self):
    buf = io.BytesIO()
    full_audio = np.arange(16000 * 200, dtype=np.int16)
    wavfile.write(buf, 16000, full_audio)
    wav_bytes = buf.getvalue()

    def respond_range(req: urllib.request.Request):
      byte_range = req.get_header('Range') or ''
      match = re.match(r'bytes=(\d+)-(\d+)', byte_range)
      if match is None:
        raise ValueError(f'Unexpected Range header: {byte_range!r}')
      start = int(match.group(1))
      end = min(int(match.group(2)), len(wav_bytes) - 1)
      return _FakeUrlResponse(wav_bytes[start : end + 1], req.full_url, 206)

    handler = _ScriptedRangeHandler(respond_range)
    opener = urllib.request.build_opener(utils.SafeRedirectHandler(), handler)
    dataset = self._make_dataset(streaming=True)
    with mock.patch.object(
        utils, 'build_safe_url_opener', autospec=True, return_value=opener
    ):
      sound = dataset.get_sound('itsabox')

    # n_trs_per_story=50 -> max_seconds = (10 + 50) * 2.0 = 120s (< 200s full).
    self.assertEqual(sound.context.id, 'itsabox')
    self.assertEqual(sound.context.sample_rate, 16000)
    self.assertEqual(sound.context.length, 16000 * 120)
    self.assertEqual(
        handler.ranges, ['bytes=0-4095', f'bytes=44-{44 + 16000 * 120 * 2 - 1}']
    )

  def test_bundled_debug_responses_fixture_loads(self):
    with flagsaver.flagsaver(dataset_basepath=None, huthlab_data_path=None):
      ds = huthlab_fmri.HuthLabDataset(streaming=True)
      self.assertEqual(ds.base_path, huthlab_fmri._DEBUG_TESTDATA_PATH)
      responses = ds.load_responses()
      expected_stories = set(huthlab_fmri.DEBUG_TRAIN_STORIES) | set(
          huthlab_fmri.DEBUG_TEST_STORIES
      )
      self.assertEqual(set(responses.keys()), expected_stories)
      self.assertEqual(responses['itsabox'].shape, (10, 100))
      self.assertEqual(responses['odetostepfather'].shape, (10, 100))
      self.assertEqual(responses['wheretheressmoke'].shape, (50, 100))

  def test_fetch_wav_prefix_bytes_stops_at_first_data_chunk(self):
    buf = io.BytesIO()
    wavfile.write(buf, 16000, np.array([100, -100], dtype=np.int16))
    trailing_data = b'data' + struct.pack('<I', 4) + b'\x00\x00\x00\x00'
    wav_with_trailing = buf.getvalue() + trailing_data
    handler = _ScriptedRangeHandler(
        lambda req: _FakeUrlResponse(wav_with_trailing, req.full_url, 206)
    )
    opener = urllib.request.build_opener(utils.SafeRedirectHandler(), handler)
    out_bytes = huthlab_fmri._fetch_wav_prefix_bytes(
        'https://s3.amazonaws.com/openneuro.org/ds003020/stimuli/itsabox.wav',
        opener=opener,
    )
    _, samples = wavfile.read(io.BytesIO(out_bytes))
    np.testing.assert_array_equal(
        samples, np.array([100, -100], dtype=np.int16)
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='small_prefix_inside_probe',
          max_seconds=0.01,
          expected_samples=160,
          expected_requests=1,
      ),
      dict(
          testcase_name='fractional_sample_rounds_up',
          max_seconds=0.01 + 1e-6,
          expected_samples=161,
          expected_requests=1,
      ),
      dict(
          testcase_name='zero_duration_prefix',
          max_seconds=0.0,
          expected_samples=0,
          expected_requests=1,
      ),
  )
  def test_fetch_wav_prefix_bytes_single_probe_cases(
      self,
      max_seconds: float,
      expected_samples: int,
      expected_requests: int,
  ):
    buf = io.BytesIO()
    wavfile.write(buf, 16000, np.ones(1600, dtype=np.int16))
    wav_bytes = buf.getvalue()
    handler = _ScriptedRangeHandler(
        lambda req: _FakeUrlResponse(wav_bytes, req.full_url, 206)
    )
    opener = urllib.request.build_opener(utils.SafeRedirectHandler(), handler)
    out_bytes = huthlab_fmri._fetch_wav_prefix_bytes(
        'https://s3.amazonaws.com/openneuro.org/ds003020/stimuli/itsabox.wav',
        max_seconds=max_seconds,
        opener=opener,
    )
    self.assertEqual(struct.unpack('<I', out_bytes[4:8])[0], len(out_bytes) - 8)
    rate, samples = wavfile.read(io.BytesIO(out_bytes))
    self.assertEqual(rate, 16000)
    self.assertLen(samples, expected_samples)
    self.assertLen(handler.ranges, expected_requests)

  @parameterized.named_parameters(
      dict(
          testcase_name='default_debug_testdata',
          explicit_arg=None,
          huthlab_flag=None,
          dataset_flag=None,
          expect_streaming=True,
          expect_debug_path=True,
      ),
      dict(
          testcase_name='placer_default_disables_streaming',
          explicit_arg=None,
          huthlab_flag=None,
          dataset_flag='PLACER',
          expect_streaming=False,
          expect_debug_path=False,
      ),
      dict(
          testcase_name='huthlab_flag_override',
          explicit_arg=None,
          huthlab_flag='CUSTOM',
          dataset_flag=None,
          expect_streaming=True,
          expect_debug_path=False,
      ),
      dict(
          testcase_name='explicit_base_path_arg',
          explicit_arg='CUSTOM',
          huthlab_flag=None,
          dataset_flag='PLACER',
          expect_streaming=True,
          expect_debug_path=False,
      ),
  )
  def test_streaming_base_path_precedence(
      self,
      explicit_arg: str | None,
      huthlab_flag: str | None,
      dataset_flag: str | None,
      expect_streaming: bool,
      expect_debug_path: bool,
  ):
    placer_dir = self.create_tempdir().full_path
    custom_dir = self.create_tempdir().full_path
    resolved_explicit_arg = custom_dir if explicit_arg == 'CUSTOM' else None
    resolved_huthlab_flag = custom_dir if huthlab_flag == 'CUSTOM' else None
    resolved_dataset_flag = placer_dir if dataset_flag == 'PLACER' else None
    with (
        flagsaver.flagsaver(
            huthlab_data_path=resolved_huthlab_flag,
            dataset_basepath=resolved_dataset_flag,
        ),
        mock.patch.object(huthlab_fmri, 'DEFAULT_BASE_PATH', placer_dir),
    ):
      ds = huthlab_fmri.HuthLabDataset(
          base_path=resolved_explicit_arg, streaming=True
      )
      self.assertEqual(ds.streaming, expect_streaming)
      if expect_debug_path:
        self.assertEqual(ds.base_path, huthlab_fmri._DEBUG_TESTDATA_PATH)
      elif explicit_arg == 'CUSTOM' or huthlab_flag == 'CUSTOM':
        self.assertEqual(ds.base_path, custom_dir)
      else:
        self.assertEqual(ds.base_path, placer_dir)

  def test_get_sound_raises_when_not_streaming_and_missing_locally(self):
    dataset = self._make_dataset(streaming=False)
    with self.assertRaisesRegex(FileNotFoundError, 'itsabox.wav'):
      dataset.get_sound('itsabox')

  def test_get_sound_rejects_invalid_story_name(self):
    dataset = self._make_dataset(streaming=True)
    with self.assertRaisesRegex(ValueError, 'Invalid HuthLab story name'):
      dataset.get_sound('../traversal')

  @parameterized.named_parameters(
      dict(
          testcase_name='non_riff_header',
          body=b'NOT_A_WAV_FILE' * 4,
          error_regex='valid WAV header',
      ),
      dict(
          testcase_name='missing_data_chunk',
          body=b'RIFF' + struct.pack('<I', 48) + b'WAVEJUNK' + struct.pack(
              '<I', 32
          ) + b'\x00' * 32,
          error_regex='missing fmt or data chunk',
      ),
      dict(
          testcase_name='truncated_fmt_chunk',
          body=(
              b'RIFF'
              + struct.pack('<I', 48)
              + b'WAVEJUNK'
              + struct.pack('<I', 28)
              + b'\x00' * 28
              + b'fmt '
              + struct.pack('<I', 16)
          ),
          error_regex='missing fmt or data chunk',
      ),
      dict(
          testcase_name='oversized_data_chunk_in_header',
          body=(
              b'RIFF'
              + struct.pack('<I', 36)
              + b'WAVEfmt '
              + struct.pack('<IHHIIHH', 16, 1, 1, 16000, 32000, 2, 16)
              + b'data'
              + struct.pack('<I', huthlab_fmri._MAX_WAV_BYTES + 2)
          ),
          error_regex='exceeds limit',
      ),
  )
  def test_fetch_wav_prefix_bytes_rejects_invalid_wav_headers(
      self, body: bytes, error_regex: str
  ):
    handler = _ScriptedRangeHandler(
        lambda req: _FakeUrlResponse(body, req.full_url, 206)
    )
    opener = urllib.request.build_opener(utils.SafeRedirectHandler(), handler)
    with self.assertRaisesRegex(ValueError, error_regex):
      huthlab_fmri._fetch_wav_prefix_bytes(
          'https://s3.amazonaws.com/openneuro.org/ds003020/stimuli/itsabox.wav',
          opener=opener,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='non_206_status',
          status=200,
          body=b'RIFF' + b'\x00' * 40,
          error_regex='Expected HTTP 206 Partial Content',
      ),
      dict(
          testcase_name='oversized_range_response',
          status=206,
          body=b'\x00' * (huthlab_fmri._WAV_HEADER_PROBE_BYTES + 10),
          error_regex='exceeded requested',
      ),
      dict(
          testcase_name='truncated_pcm_payload',
          status=206,
          body=(
              b'RIFF'
              + struct.pack('<I', 8036)
              + b'WAVEfmt '
              + struct.pack('<IHHIIHH', 16, 1, 1, 16000, 32000, 2, 16)
              + b'data'
              + struct.pack('<I', 8000)
              + b'\x00' * 100
          ),
          error_regex='Truncated HTTP Range response',
      ),
  )
  def test_fetch_wav_prefix_bytes_rejects_invalid_http_range_responses(
      self, status: int, body: bytes, error_regex: str
  ):
    handler = _ScriptedRangeHandler(
        lambda req: _FakeUrlResponse(body, req.full_url, status)
    )
    opener = urllib.request.build_opener(utils.SafeRedirectHandler(), handler)
    with self.assertRaisesRegex(OSError, error_regex):
      huthlab_fmri._fetch_wav_prefix_bytes(
          'https://s3.amazonaws.com/openneuro.org/ds003020/stimuli/itsabox.wav',
          opener=opener,
      )

  @parameterized.named_parameters(
      dict(testcase_name='negative_start', start=-1, end=10),
      dict(testcase_name='end_before_start', start=10, end=5),
  )
  def test_read_bounded_range_rejects_invalid_bounds(
      self, start: int, end: int
  ):
    opener = urllib.request.build_opener(utils.SafeRedirectHandler())
    with self.assertRaisesRegex(ValueError, 'Invalid byte range'):
      huthlab_fmri._read_bounded_range(
          opener,
          'https://s3.amazonaws.com/openneuro.org/ds003020/stimuli/itsabox.wav',
          start,
          end,
      )

  @parameterized.named_parameters(
      dict(testcase_name='negative_seconds', max_seconds=-1.0),
      dict(testcase_name='nan_seconds', max_seconds=float('nan')),
  )
  def test_fetch_wav_prefix_bytes_rejects_invalid_max_seconds(
      self, max_seconds: float
  ):
    opener = urllib.request.build_opener(utils.SafeRedirectHandler())
    with self.assertRaisesRegex(ValueError, 'max_seconds'):
      huthlab_fmri._fetch_wav_prefix_bytes(
          'https://s3.amazonaws.com/openneuro.org/ds003020/stimuli/itsabox.wav',
          max_seconds=max_seconds,
          opener=opener,
      )

  def test_custom_train_and_test_stories(self):
    dataset = self._make_dataset(
        train_stories=huthlab_fmri.DEBUG_TRAIN_STORIES,
        test_stories=huthlab_fmri.DEBUG_TEST_STORIES,
    )
    self.assertEqual(
        dataset.get_train_stories(), huthlab_fmri.DEBUG_TRAIN_STORIES
    )
    self.assertEqual(
        dataset.get_test_stories(), huthlab_fmri.DEBUG_TEST_STORIES
    )

  # --- Constants tests ---

  def test_default_parameters(self):
    self.assertEqual(huthlab_fmri.DEFAULT_TR_DURATION, 2.0)
    self.assertEqual(huthlab_fmri.DEFAULT_DELAYS, (1, 2, 3, 4))

  def test_subjects(self):
    self.assertEqual(huthlab_fmri.SUBJECTS, ('S01', 'S02', 'S03'))

  def test_train_stories_is_dict(self):
    self.assertIsInstance(huthlab_fmri.TRAIN_STORIES, dict)
    for subject in huthlab_fmri.SUBJECTS:
      self.assertIn(subject, huthlab_fmri.TRAIN_STORIES)

  def test_test_stories_is_dict(self):
    self.assertIsInstance(huthlab_fmri.TEST_STORIES, dict)
    for subject in huthlab_fmri.SUBJECTS:
      self.assertIn(subject, huthlab_fmri.TEST_STORIES)


if __name__ == '__main__':
  absltest.main()
