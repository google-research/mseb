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

from collections.abc import Mapping
import email.message
import io
import os
import re
from unittest import mock
import urllib.error
import urllib.request
import urllib.response

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import test_pipeline
from apache_beam.testing import util as beam_util
from mseb import utils
from mseb.datasets import speech_massive
import numpy as np
import pandas as pd


class _FakeUrlResponse(urllib.response.addinfourl):
  """In-memory addinfourl response with an HTTP status message."""

  def __init__(
      self,
      fp: io.BytesIO,
      headers: email.message.Message,
      url: str,
      code: int,
  ):
    super().__init__(fp, headers, url, code=code)
    self.msg = "OK"


def _make_url_response(
    url: str,
    status: int,
    body: bytes = b"",
    headers: Mapping[str, str] | None = None,
) -> _FakeUrlResponse:
  msg = email.message.Message()
  if headers:
    for key, value in headers.items():
      msg[key] = value
  return _FakeUrlResponse(io.BytesIO(body), msg, url, code=status)


class _ScriptedHttpHandler(
    urllib.request.HTTPHandler, urllib.request.HTTPSHandler
):
  """In-memory urllib handler for testing redirects and Range requests."""

  def __init__(self, responder):
    urllib.request.HTTPHandler.__init__(self)
    urllib.request.HTTPSHandler.__init__(self)
    self._responder = responder
    self.requests: list[urllib.request.Request] = []

  def https_open(self, req: urllib.request.Request):
    self.requests.append(req)
    return self._responder(req)

  def http_open(self, req: urllib.request.Request):
    self.requests.append(req)
    return self._responder(req)


def _has_auth_header(req: urllib.request.Request) -> bool:
  return any(k.lower() == "authorization" for k, _ in req.header_items())


class SpeechMassiveTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.testdata_dir = self.create_tempdir()

    lang_dir = os.path.join(self.testdata_dir.full_path, "de-DE")
    os.makedirs(lang_dir)

    mock_data = dict(
        id=["2205"],
        locale=["de-DE"],
        partition=["test"],
        scenario=[10],
        scenario_str=["audio"],
        intent_idx=[46],
        intent_str=["audio_volume_mute"],
        utt=["stille für zwei stunden"],
        annot_utt=["stille für [time : zwei stunden]"],
        worker_id=["8"],
        slot_method=[{
            "slot": np.array(["time"]),
            "method": np.array(["translation"]),
        }],
        judgments=[{
            "worker_id": np.array(["27", "28", "8"]),
            "intent_score": np.array([1, 1, 1], dtype=np.int8),
            "slots_score": np.array([1, 1, 1], dtype=np.int8),
            "grammar_score": np.array([3, 4, 4], dtype=np.int8),
            "spelling_score": np.array([2, 2, 2], dtype=np.int8),
            "language_identification": np.array(["target", "target", "target"]),
        }],
        tokens=[np.array(["stille", "für", "zwei", "stunden"])],
        labels=[np.array(["Other", "Other", "time", "time"])],
        audio=[{
            "bytes": (
                b"RIFF$\x00\x00\x00WAVEfmt"
                b" \x10\x00\x00\x00\x01\x00\x01\x00@\x1f\x00\x00\x80>\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
            ),
            "path": "c15b5445ba46918a8d678e7b59b80aa6.wav",
        }],
        path=["test/c15b5445ba46918a8d678e7b59b80aa6.wav"],
        is_transcript_reported=[False],
        is_validated=[True],
        speaker_id=["5f32d5f107d49607c3f6cf7a"],
        speaker_sex=["Female"],
        speaker_age=["40"],
        speaker_ethnicity_simple=["White"],
        speaker_country_of_birth=["Germany"],
        speaker_country_of_residence=["Germany"],
        speaker_nationality=["Germany"],
        speaker_first_language=["German"],
    )
    pd.DataFrame(mock_data).to_parquet(
        os.path.join(lang_dir, "test-00000-of-00001.parquet")
    )

  @mock.patch("mseb.utils.download_from_hf")
  def test_loading_and_parsing(self, _):
    dataset = speech_massive.SpeechMassiveDataset(
        base_path=self.testdata_dir.full_path,
        filename="de-DE/test-00000-of-00001.parquet",
    )

    self.assertLen(dataset, 1)

    task_df = dataset.get_task_data()
    self.assertIsInstance(task_df, pd.DataFrame)
    self.assertLen(task_df, 1)
    self.assertNotIn("audio", task_df.columns)

    record = task_df.iloc[0]
    self.assertEqual(record.locale, "de-DE")
    self.assertEqual(record.partition, "test")
    self.assertEqual(record.speaker_id, "5f32d5f107d49607c3f6cf7a")
    self.assertEqual(record.speaker_sex, "Female")
    self.assertEqual(record.speaker_age, "40")
    self.assertEqual(record.intent_str, "audio_volume_mute")
    self.assertEqual(record.intent_idx, 46)

  def test_default_base_path_resolution(self):
    with flagsaver.flagsaver(dataset_basepath=None):
      with mock.patch.object(
          speech_massive, "DEFAULT_BASE_PATH", self.testdata_dir.full_path
      ):
        dataset = speech_massive.SpeechMassiveDataset(
            filename="de-DE/test-00000-of-00001.parquet"
        )
        self.assertEqual(dataset.base_path, self.testdata_dir.full_path)
        self.assertLen(dataset, 1)

  @mock.patch("mseb.utils.download_from_hf")
  def test_default_base_path_missing_file_raises_without_download(
      self, mock_download
  ):
    with flagsaver.flagsaver(dataset_basepath=None):
      with mock.patch.object(
          speech_massive, "DEFAULT_BASE_PATH", self.testdata_dir.full_path
      ):
        dataset = speech_massive.SpeechMassiveDataset(
            filename="de-DE/nonexistent.parquet"
        )
        with self.assertRaises(FileNotFoundError):
          _ = len(dataset)
        mock_download.assert_not_called()

  @mock.patch("mseb.utils.download_from_hf")
  def test_get_sound(self, _):
    dataset = speech_massive.SpeechMassiveDataset(
        base_path=self.testdata_dir.full_path,
        filename="de-DE/test-?????-of-?????.parquet",
    )
    record = dataset.get_task_data(with_audio=True).iloc[0]
    sound = dataset.get_sound(record)
    self.assertEqual(
        sound.context.id, "test/c15b5445ba46918a8d678e7b59b80aa6.wav"
    )
    self.assertEqual(sound.context.text, "stille für zwei stunden")
    self.assertEqual(sound.context.speaker_id, "5f32d5f107d49607c3f6cf7a")
    self.assertEqual(sound.context.speaker_age, 40)
    self.assertEqual(sound.context.speaker_gender, "Female")
    self.assertAlmostEqual(sound.context.waveform_end_second, 0.0)

  @mock.patch("mseb.utils.download_from_hf")
  def test_get_task_data_beam(self, _):
    dataset = speech_massive.SpeechMassiveDataset(
        base_path=self.testdata_dir.full_path,
        filename="de-DE/test-?????-of-?????.parquet",
    )
    with test_pipeline.TestPipeline() as p:
      pcoll = p | dataset.get_task_data_beam()

      expected_output = [{
          "locale": "de-DE",
          "partition": "test",
          "speaker_id": "5f32d5f107d49607c3f6cf7a",
          "text": "stille für zwei stunden",
      }]

      beam_util.assert_that(
          pcoll
          | "MapToSimpleDict"
          >> beam.Map(
              lambda x: {
                  "locale": x["locale"],
                  "partition": x["partition"],
                  "speaker_id": x["speaker_id"],
                  "text": x["sound"].context.text,
              }
          ),
          beam_util.equal_to(expected_output),
      )

  @mock.patch("mseb.utils.download_from_hf")
  def test_get_task_sounds_beam(self, _):
    dataset = speech_massive.SpeechMassiveDataset(
        base_path=self.testdata_dir.full_path,
        filename="de-DE/test-?????-of-?????.parquet",
    )
    with test_pipeline.TestPipeline() as p:
      pcoll = p | dataset.get_task_sounds_beam(locale="de_de")

      expected_output = [{
          "id": "test/c15b5445ba46918a8d678e7b59b80aa6.wav",
          "language": "de_de",
      }]

      beam_util.assert_that(
          pcoll
          | "MapToSimpleDict"
          >> beam.Map(
              lambda x: {
                  "id": x.context.id,
                  "language": x.context.language,
              }
          ),
          beam_util.equal_to(expected_output),
      )

  def test_streaming_selective_http_range_and_filter_fn(self):
    parquet_path = os.path.join(
        self.testdata_dir.full_path, "de-DE", "test-00000-of-00001.parquet"
    )
    with open(parquet_path, "rb") as f:
      base_df = pd.read_parquet(f)
    two_rows = pd.concat([base_df, base_df], ignore_index=True)
    two_rows.loc[0, "id"] = "6902"
    two_rows.loc[1, "id"] = "999999"
    buf = io.BytesIO()
    two_rows.to_parquet(buf, row_group_size=1)
    parquet_bytes = buf.getvalue()

    range_requests = []

    def respond_range(req: urllib.request.Request):
      byte_range = req.get_header("Range") or ""
      range_requests.append(byte_range)
      match = re.match(r"bytes=(\d+)-(\d+)", byte_range)
      if match is None:
        raise ValueError(f"Unexpected Range header: {byte_range!r}")
      start, end = int(match.group(1)), int(match.group(2))
      end = min(end, len(parquet_bytes) - 1)
      total = len(parquet_bytes)
      return _make_url_response(
          req.full_url,
          206,
          body=parquet_bytes[start : end + 1],
          headers={"Content-Range": f"bytes {start}-{end}/{total}"},
      )

    handler = _ScriptedHttpHandler(respond_range)
    opener = urllib.request.build_opener(
        utils.SafeRedirectHandler(), handler
    )
    empty_dir = self.create_tempdir().full_path
    with (
        flagsaver.flagsaver(dataset_basepath=None),
        mock.patch.object(speech_massive, "DEFAULT_BASE_PATH", empty_dir),
        mock.patch.object(
            utils,
            "list_hf_files",
            autospec=True,
            return_value=["de-DE/test-00000-of-00001.parquet"],
        ) as mock_list,
        mock.patch.object(
            utils,
            "build_safe_url_opener",
            autospec=True,
            return_value=opener,
        ),
    ):
      ds = speech_massive.SpeechMassiveDataset(
          filename="de-DE/test-?????-of-?????.parquet",
          streaming=True,
          token="test_token",
          filter_fn=speech_massive.is_member_of_debug,
      )
      self.assertIsNone(ds.base_path)
      self.assertTrue(ds.streaming)
      df = ds.get_task_data()
      mock_list.assert_called_once()
      self.assertEqual(list(df["id"]), ["6902"])
      self.assertNotIn("audio", df.columns)
      df_audio = ds.get_task_data(with_audio=True)
      self.assertEqual(list(df_audio["id"]), ["6902"])
      sound = ds.get_sound(df_audio.iloc[0].to_dict())
      self.assertEqual(
          sound.context.id, "test/c15b5445ba46918a8d678e7b59b80aa6.wav"
      )
      self.assertNotEmpty(range_requests)

  @parameterized.named_parameters(
      dict(
          testcase_name="explicit_base_path",
          use_explicit_arg=True,
          use_flag=False,
      ),
      dict(
          testcase_name="dataset_basepath_flag",
          use_explicit_arg=False,
          use_flag=True,
      ),
  )
  def test_streaming_honors_local_base_path(
      self, use_explicit_arg: bool, use_flag: bool
  ):
    local_dir = self.testdata_dir.full_path
    base_path = local_dir if use_explicit_arg else None
    flag_basepath = local_dir if use_flag else None
    with (
        flagsaver.flagsaver(dataset_basepath=flag_basepath),
        mock.patch.object(
            utils, "build_safe_url_opener", autospec=True
        ) as mock_build_opener,
    ):
      ds = speech_massive.SpeechMassiveDataset(
          filename="de-DE/test-?????-of-?????.parquet",
          base_path=base_path,
          streaming=True,
      )
      self.assertFalse(ds.streaming)
      self.assertEqual(ds.base_path, local_dir)
      self.assertLen(ds, 1)
      mock_build_opener.assert_not_called()

  @parameterized.parameters(401, 403)
  def test_http_range_file_raises_permission_error_on_auth_failure(
      self, status_code: int
  ):
    handler = _ScriptedHttpHandler(
        lambda req: _make_url_response(req.full_url, status_code)
    )
    opener = urllib.request.build_opener(
        utils.SafeRedirectHandler(), handler
    )
    with self.assertRaisesRegex(PermissionError, "HF_TOKEN"):
      speech_massive._HttpRangeFile(
          "https://huggingface.co/datasets/org/repo/resolve/main/a.parquet",
          repo_id="org/repo",
          opener=opener,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="cross_origin_strips_auth",
          redirect_url="https://cdn.example.com/a.parquet?sig=1",
          expect_redirect_auth=False,
      ),
      dict(
          testcase_name="same_origin_preserves_auth",
          redirect_url=(
              "https://huggingface.co/datasets/org/repo/resolve/rev1/a.parquet"
          ),
          expect_redirect_auth=True,
      ),
  )
  def test_http_range_file_redirect_authorization_handling(
      self, redirect_url: str, expect_redirect_auth: bool
  ):
    origin_url = (
        "https://huggingface.co/datasets/org/repo/resolve/main/a.parquet"
    )
    payload = b"abcd"

    def responder(req: urllib.request.Request):
      if req.full_url == origin_url:
        return _make_url_response(
            req.full_url, 302, headers={"Location": redirect_url}
        )
      byte_range = req.get_header("Range") or ""
      match = re.match(r"bytes=(\d+)-(\d+)", byte_range)
      if match is None:
        raise ValueError(f"Unexpected Range header: {byte_range!r}")
      start, end = int(match.group(1)), int(match.group(2))
      return _make_url_response(
          req.full_url,
          206,
          body=payload[start : end + 1],
          headers={"Content-Range": f"bytes {start}-{end}/{len(payload)}"},
      )

    handler = _ScriptedHttpHandler(responder)
    opener = urllib.request.build_opener(
        utils.SafeRedirectHandler(), handler
    )
    range_file = speech_massive._HttpRangeFile(
        origin_url,
        headers={"Authorization": "Bearer secret_token"},
        repo_id="org/repo",
        opener=opener,
    )
    out = bytearray(4)
    self.assertEqual(range_file.readinto(out), 4)
    self.assertEqual(bytes(out), payload)
    self.assertLen(handler.requests, 3)
    self.assertTrue(_has_auth_header(handler.requests[0]))
    self.assertEqual(
        _has_auth_header(handler.requests[1]), expect_redirect_auth
    )
    self.assertEqual(handler.requests[2].full_url, redirect_url)
    self.assertEqual(
        _has_auth_header(handler.requests[2]), expect_redirect_auth
    )

  def test_http_range_file_rejects_https_downgrade_redirect(self):
    origin_url = (
        "https://huggingface.co/datasets/org/repo/resolve/main/a.parquet"
    )
    handler = _ScriptedHttpHandler(
        lambda req: _make_url_response(
            req.full_url,
            302,
            headers={"Location": "http://cdn.example.com/a.parquet"},
        )
    )
    opener = urllib.request.build_opener(
        utils.SafeRedirectHandler(), handler
    )
    with self.assertRaisesRegex(
        urllib.error.URLError, "Refusing redirect from HTTPS"
    ):
      speech_massive._HttpRangeFile(
          origin_url,
          headers={"Authorization": "Bearer secret_token"},
          opener=opener,
      )

  def test_http_range_file_refreshes_and_recaches_expired_cdn_url(self):
    origin_url = (
        "https://huggingface.co/datasets/org/repo/resolve/main/a.parquet"
    )
    cdn_v1 = "https://cdn.example.com/a.parquet?sig=v1"
    cdn_v2 = "https://cdn.example.com/a.parquet?sig=v2"
    redirect_targets = iter([cdn_v1, cdn_v2])
    cdn_v1_calls = 0

    def responder(req: urllib.request.Request):
      nonlocal cdn_v1_calls
      if req.full_url == origin_url:
        return _make_url_response(
            req.full_url, 302, headers={"Location": next(redirect_targets)}
        )
      if req.full_url == cdn_v1:
        cdn_v1_calls += 1
        if cdn_v1_calls == 1:
          return _make_url_response(
              req.full_url,
              206,
              body=b"a",
              headers={"Content-Range": "bytes 0-0/4"},
          )
        return _make_url_response(req.full_url, 403)
      byte_range = req.get_header("Range") or ""
      if byte_range == "bytes=0-1":
        return _make_url_response(
            req.full_url,
            206,
            body=b"ab",
            headers={"Content-Range": "bytes 0-1/4"},
        )
      return _make_url_response(
          req.full_url,
          206,
          body=b"cd",
          headers={"Content-Range": "bytes 2-3/4"},
      )

    handler = _ScriptedHttpHandler(responder)
    opener = urllib.request.build_opener(
        utils.SafeRedirectHandler(), handler
    )
    range_file = speech_massive._HttpRangeFile(
        origin_url,
        headers={"Authorization": "Bearer secret_token"},
        repo_id="org/repo",
        opener=opener,
    )
    first_half = bytearray(2)
    second_half = bytearray(2)
    self.assertEqual(range_file.readinto(first_half), 2)
    self.assertEqual(range_file.readinto(second_half), 2)
    self.assertEqual(bytes(first_half + second_half), b"abcd")
    origin_hits = [r for r in handler.requests if r.full_url == origin_url]
    self.assertLen(origin_hits, 2)
    self.assertEqual(handler.requests[-1].full_url, cdn_v2)
    for req in handler.requests:
      if req.full_url.startswith("https://cdn.example.com/"):
        self.assertFalse(_has_auth_header(req))

  def test_http_range_file_content_length_fallback_and_eof(self):
    url = "https://huggingface.co/datasets/org/repo/resolve/main/a.parquet"
    call_count = 0

    def responder(req: urllib.request.Request):
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        return _make_url_response(
            req.full_url, 200, body=b"a", headers={"Content-Length": "4"}
        )
      return _make_url_response(
          req.full_url,
          206,
          body=b"abcd",
          headers={"Content-Range": "bytes 0-3/4"},
      )

    handler = _ScriptedHttpHandler(responder)
    opener = urllib.request.build_opener(
        utils.SafeRedirectHandler(), handler
    )
    range_file = speech_massive._HttpRangeFile(url, opener=opener)
    self.assertEqual(range_file.readinto(bytearray(0)), 0)
    self.assertEqual(call_count, 1)
    out = bytearray(4)
    self.assertEqual(range_file.readinto(out), 4)
    self.assertEqual(bytes(out), b"abcd")
    self.assertEqual(range_file.readinto(bytearray(4)), 0)
    self.assertEqual(call_count, 2)

  @parameterized.named_parameters(
      dict(
          testcase_name="unknown_star_total_on_206",
          status=206,
          headers={"Content-Range": "bytes 0-0/*", "Content-Length": "1"},
      ),
      dict(
          testcase_name="missing_size_headers_on_200",
          status=200,
          headers={},
      ),
  )
  def test_http_range_file_fetch_size_rejects_missing_total_size(
      self, status: int, headers: Mapping[str, str]
  ):
    url = "https://huggingface.co/datasets/org/repo/resolve/main/a.parquet"
    handler = _ScriptedHttpHandler(
        lambda req: _make_url_response(
            req.full_url, status, body=b"a", headers=headers
        )
    )
    opener = urllib.request.build_opener(
        utils.SafeRedirectHandler(), handler
    )
    with self.assertRaisesRegex(OSError, "total file size"):
      speech_massive._HttpRangeFile(url, opener=opener)

  @parameterized.named_parameters(
      dict(
          testcase_name="non_206_status",
          read_status=200,
          read_body=b"0123456789",
          error_regex="Expected HTTP 206 Partial Content",
      ),
      dict(
          testcase_name="oversized_range_body",
          read_status=206,
          read_body=b"01234567",
          error_regex="exceeded requested 4 bytes",
      ),
  )
  def test_http_range_file_readinto_rejects_invalid_range_response(
      self, read_status: int, read_body: bytes, error_regex: str
  ):
    url = "https://huggingface.co/datasets/org/repo/resolve/main/a.parquet"
    call_count = 0

    def responder(req: urllib.request.Request):
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        return _make_url_response(
            req.full_url,
            206,
            body=b"0",
            headers={"Content-Range": "bytes 0-0/10"},
        )
      return _make_url_response(
          req.full_url,
          read_status,
          body=read_body,
          headers={"Content-Range": "bytes 0-3/10"},
      )

    handler = _ScriptedHttpHandler(responder)
    opener = urllib.request.build_opener(
        utils.SafeRedirectHandler(), handler
    )
    range_file = speech_massive._HttpRangeFile(url, opener=opener)
    with self.assertRaisesRegex(OSError, error_regex):
      range_file.readinto(bytearray(4))

  def test_filter_fn_empty_shard_and_multishard_concat_and_audio_first_cache(
      self,
  ):
    shard_dir = self.create_tempdir().full_path
    lang_dir = os.path.join(shard_dir, "de-DE")
    os.makedirs(lang_dir)
    base_parquet = os.path.join(
        self.testdata_dir.full_path, "de-DE", "test-00000-of-00001.parquet"
    )
    base_df = pd.read_parquet(base_parquet)

    shard0 = base_df.copy()
    shard0["id"] = ["non_matching_id"]
    shard1 = base_df.copy()
    shard1["id"] = ["6902"]
    shard2 = base_df.copy()
    shard2["id"] = ["13734"]
    shard0_path = os.path.join(lang_dir, "test-00000-of-00003.parquet")
    shard1_path = os.path.join(lang_dir, "test-00001-of-00003.parquet")
    shard2_path = os.path.join(lang_dir, "test-00002-of-00003.parquet")
    shard0.to_parquet(shard0_path)
    shard1.to_parquet(shard1_path)
    shard2.to_parquet(shard2_path)

    empty_ds = speech_massive.SpeechMassiveDataset(
        base_path=shard_dir,
        filename="de-DE/test-00000-of-00003.parquet",
        filter_fn=speech_massive.is_member_of_debug,
    )
    empty_df = empty_ds.get_task_data()
    self.assertEmpty(empty_df)
    self.assertIn("id", empty_df.columns)
    self.assertNotIn("audio", empty_df.columns)

    multi_ds = speech_massive.SpeechMassiveDataset(
        base_path=shard_dir,
        filename="de-DE/test-?????-of-00003.parquet",
        filter_fn=speech_massive.is_member_of_debug,
    )
    unsorted_shards = [
        speech_massive.epath.Path(shard2_path),
        speech_massive.epath.Path(shard0_path),
        speech_massive.epath.Path(shard1_path),
    ]
    path_cls = type(speech_massive.epath.Path(lang_dir))
    with mock.patch.object(
        path_cls, "glob", autospec=True, return_value=iter(unsorted_shards)
    ):
      df_with_audio = multi_ds.get_task_data(with_audio=True)
    self.assertEqual(list(df_with_audio["id"]), ["6902", "13734"])
    self.assertEqual(list(df_with_audio.index), [0, 1])
    self.assertIn("audio", df_with_audio.columns)

    with mock.patch.object(
        speech_massive, "_read_parquet_file", autospec=True
    ) as mock_read_parquet:
      cached_df = multi_ds.get_task_data(with_audio=False)
      mock_read_parquet.assert_not_called()
    self.assertEqual(list(cached_df["id"]), ["6902", "13734"])
    self.assertNotIn("audio", cached_df.columns)

  def test_streaming_sorts_unsorted_hf_file_listing(self):
    base_parquet = os.path.join(
        self.testdata_dir.full_path, "de-DE", "test-00000-of-00001.parquet"
    )
    base_df = pd.read_parquet(base_parquet)
    shard0 = base_df.copy()
    shard0["id"] = ["6902"]
    shard1 = base_df.copy()
    shard1["id"] = ["13734"]
    buf0, buf1 = io.BytesIO(), io.BytesIO()
    shard0.to_parquet(buf0)
    shard1.to_parquet(buf1)
    shard_bytes = {
        "de-DE/test-00000-of-00002.parquet": buf0.getvalue(),
        "de-DE/test-00001-of-00002.parquet": buf1.getvalue(),
    }

    def fake_open_hf_parquet(repo_id, filename, token=None):
      del repo_id, token
      return io.BytesIO(shard_bytes[filename])

    with (
        flagsaver.flagsaver(dataset_basepath=None),
        mock.patch.object(
            utils,
            "list_hf_files",
            autospec=True,
            return_value=[
                "de-DE/test-00001-of-00002.parquet",
                "de-DE/test-00000-of-00002.parquet",
            ],
        ),
        mock.patch.object(
            speech_massive,
            "_open_hf_parquet",
            autospec=True,
            side_effect=fake_open_hf_parquet,
        ),
    ):
      ds = speech_massive.SpeechMassiveDataset(
          filename="de-DE/test-?????-of-00002.parquet",
          streaming=True,
      )
      df = ds.get_task_data()
    self.assertEqual(list(df["id"]), ["6902", "13734"])


class DebugAndCompactIdsTest(absltest.TestCase):

  def test_is_member_of_debug_true(self):
    self.assertTrue(speech_massive.is_member_of_debug("6902"))
    self.assertTrue(speech_massive.is_member_of_debug("13734"))
    self.assertTrue(speech_massive.is_member_of_debug("16603"))

  def test_is_member_of_debug_false(self):
    self.assertFalse(speech_massive.is_member_of_debug("non_existent_id"))
    self.assertFalse(speech_massive.is_member_of_debug(""))

  def test_is_member_of_compact_true(self):
    self.assertTrue(speech_massive.is_member_of_compact("0"))
    self.assertTrue(speech_massive.is_member_of_compact("10125"))

  def test_is_member_of_compact_false(self):
    self.assertFalse(speech_massive.is_member_of_compact("non_existent_id"))
    self.assertFalse(speech_massive.is_member_of_compact(""))

  def test_is_member_of_compact_size(self):
    count = 0
    for i in range(20_000):
      count += int(speech_massive.is_member_of_compact(str(i)))
    self.assertEqual(count, 500)


if __name__ == "__main__":
  absltest.main()
