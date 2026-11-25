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

"""HuthLab Encoding Model Scaling Laws dataset.

Loads fMRI responses and stimulus metadata from the HuthLab
encoding-model-scaling-laws dataset:

  Antonello, Turek, Vo, and Huth (2024). "Scaling in Speech and Language Models:
  A Path to Human-Level Performance?" arXiv:2401.10150.

  Data: https://utexas.box.com/v/EncodingModelScalingLaws

The data consists of:
  - fMRI BOLD responses from subjects listening to narrative stories.
  - Precomputed model features (downsampled to TR resolution).
  - Audio stimuli as WAV files.

Directory structure expected under `base_path`:
  base_path/
    responses/
      {subject}_test_stories_avg_resp.jbl   # (T_test, V) test fMRI responses
    stimuli/
    <-- story_audio/
      {story_name}.wav                  # Audio stimuli
"""

from collections.abc import Sequence
import io
import math
import os
import re
import struct
import urllib.request

from absl import flags
from absl import logging
from etils import epath
import joblib
from mseb import dataset
from mseb import types
from mseb import utils
import numpy as np

DEFAULT_BASE_PATH = None

_HUTHLAB_DATA_PATH = flags.DEFINE_string(
    'huthlab_data_path',
    None,
    'Path to the HuthLab encoding-model-scaling-laws data directory.',
)

_OPENNEURO_STIMULI_URL = (
    'https://s3.amazonaws.com/openneuro.org/ds003020/stimuli'
)
_DEBUG_TESTDATA_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), '..', 'testdata', 'debug', 'huthlab'
    )
)
_WAV_HEADER_PROBE_BYTES = 4096
_MAX_WAV_BYTES = 256 * 1024 * 1024
_STORY_NAME_RE = re.compile(r'^[A-Za-z0-9_-]+$')

# Default parameters from the paper and tutorial notebook.
DEFAULT_TR_DURATION = 2.0  # seconds
DEFAULT_DELAYS = (1, 2, 3, 4)  # FIR delays in TRs (2s, 4s, 6s, 8s)

# Subjects available in the dataset.
SUBJECTS = (
    'S01',
    'S02',
    'S03',
)

DEBUG_TRAIN_STORIES = (
    'itsabox',
    'odetostepfather',
)
DEBUG_TEST_STORIES = (
    'wheretheressmoke',
)

TRAIN_STORIES = {
    'S01': (
        'itsabox',
        'odetostepfather',
        'inamoment',
        'hangtime',
        'ifthishaircouldtalk',
        'goingthelibertyway',
        'golfclubbing',
        'thetriangleshirtwaistconnection',
        'igrewupinthewestborobaptistchurch',
        'tetris',
        'becomingindian',
        'canplanetearthfeedtenbillionpeoplepart1',
        'thetiniestbouquet',
        'swimmingwithastronauts',
        'lifereimagined',
        'forgettingfear',
        'stumblinginthedark',
        'backsideofthestorm',
        'food',
        'theclosetthatateeverything',
        'notontheusualtour',
        'exorcism',
        'adventuresinsayingyes',
        'thefreedomridersandme',
        'cocoonoflove',
        'waitingtogo',
        'thepostmanalwayscalls',
        'googlingstrangersandkentuckybluegrass',
        'mayorofthefreaks',
        'learninghumanityfromdogs',
        'shoppinginchina',
        'souls',
        'cautioneating',
        'comingofageondeathrow',
        'breakingupintheageofgoogle',
        'gpsformylostidentity',
        'eyespy',
        'treasureisland',
        'thesurprisingthingilearnedsailingsoloaroundtheworld',
        'theadvancedbeginner',
        'goldiethegoldfish',
        'life',
        'thumbsup',
        'seedpotatoesofleningrad',
        'theshower',
        'adollshouse',
        'canplanetearthfeedtenbillionpeoplepart2',
        'sloth',
        'howtodraw',
        'quietfire',
        'metsmagic',
        'penpal',
        'thecurse',
        'canadageeseandddp',
        'thatthingonmyarm',
        'buck',
        'wildwomenanddancingqueens',
        'againstthewind',
        'indianapolis',
        'alternateithicatom',
        'bluehope',
        'kiksuya',
        'afatherscover',
        'haveyoumethimyet',
        'firetestforlove',
        'catfishingstrangerstofindmyself',
        'christmas1940',
        'tildeath',
        'lifeanddeathontheoregontrail',
        'vixenandtheussr',
        'undertheinfluence',
        'beneaththemushroomcloud',
        'jugglingandjesus',
        'superheroesjustforeachother',
        'sweetaspie',
        'naked',
        'singlewomanseekingmanwich',
        'avatar',
        'whenmothersbullyback',
        'myfathershands',
        'reachingoutbetweenthebars',
        'theinterview',
        'stagefright',
        'legacy',
        'canplanetearthfeedtenbillionpeoplepart3',
        'listo',
        'gangstersandcookies',
        'birthofanation',
        'mybackseatviewofagreatromance',
        'lawsthatchokecreativity',
        'threemonths',
        'whyimustspeakoutaboutclimatechange',
        'leavingbaghdad',
    ),
    'S02': (
        'itsabox',
        'odetostepfather',
        'inamoment',
        'afearstrippedbare',
        'findingmyownrescuer',
        'hangtime',
        'ifthishaircouldtalk',
        'goingthelibertyway',
        'golfclubbing',
        'thetriangleshirtwaistconnection',
        'igrewupinthewestborobaptistchurch',
        'tetris',
        'becomingindian',
        'canplanetearthfeedtenbillionpeoplepart1',
        'thetiniestbouquet',
        'swimmingwithastronauts',
        'lifereimagined',
        'forgettingfear',
        'stumblinginthedark',
        'backsideofthestorm',
        'food',
        'theclosetthatateeverything',
        'escapingfromadirediagnosis',
        'notontheusualtour',
        'exorcism',
        'adventuresinsayingyes',
        'thefreedomridersandme',
        'cocoonoflove',
        'waitingtogo',
        'thepostmanalwayscalls',
        'googlingstrangersandkentuckybluegrass',
        'mayorofthefreaks',
        'learninghumanityfromdogs',
        'shoppinginchina',
        'souls',
        'cautioneating',
        'comingofageondeathrow',
        'breakingupintheageofgoogle',
        'gpsformylostidentity',
        'marryamanwholoveshismother',
        'eyespy',
        'treasureisland',
        'thesurprisingthingilearnedsailingsoloaroundtheworld',
        'theadvancedbeginner',
        'goldiethegoldfish',
        'life',
        'thumbsup',
        'seedpotatoesofleningrad',
        'theshower',
        'adollshouse',
        'canplanetearthfeedtenbillionpeoplepart2',
        'sloth',
        'howtodraw',
        'quietfire',
        'metsmagic',
        'penpal',
        'thecurse',
        'canadageeseandddp',
        'thatthingonmyarm',
        'buck',
        'thesecrettomarriage',
        'wildwomenanddancingqueens',
        'againstthewind',
        'indianapolis',
        'alternateithicatom',
        'bluehope',
        'kiksuya',
        'afatherscover',
        'haveyoumethimyet',
        'firetestforlove',
        'catfishingstrangerstofindmyself',
        'christmas1940',
        'tildeath',
        'lifeanddeathontheoregontrail',
        'vixenandtheussr',
        'undertheinfluence',
        'beneaththemushroomcloud',
        'jugglingandjesus',
        'superheroesjustforeachother',
        'sweetaspie',
        'naked',
        'singlewomanseekingmanwich',
        'avatar',
        'whenmothersbullyback',
        'myfathershands',
        'reachingoutbetweenthebars',
        'theinterview',
        'stagefright',
        'legacy',
        'canplanetearthfeedtenbillionpeoplepart3',
        'listo',
        'gangstersandcookies',
        'birthofanation',
        'mybackseatviewofagreatromance',
        'lawsthatchokecreativity',
        'threemonths',
        'whyimustspeakoutaboutclimatechange',
        'leavingbaghdad',
    ),
}
TRAIN_STORIES['S03'] = TRAIN_STORIES['S02']

TEST_STORIES = {
    'S01': ('wheretheressmoke', 'onapproachtopluto', 'fromboyhoodtofatherhood'),
    'S02': ('wheretheressmoke', 'onapproachtopluto', 'fromboyhoodtofatherhood'),
    'S03': ('wheretheressmoke', 'onapproachtopluto', 'fromboyhoodtofatherhood'),
}


def _read_bounded_range(
    opener: urllib.request.OpenerDirector, url: str, start: int, end: int
) -> bytes:
  """Reads an inclusive HTTP byte range with HTTP 206 and size validation."""
  if start < 0 or end < start:
    raise ValueError(f'Invalid byte range: start={start}, end={end}')
  expected_bytes = end - start + 1
  request = urllib.request.Request(url)
  request.add_header('Range', f'bytes={start}-{end}')
  with opener.open(
      request, timeout=utils.DEFAULT_HTTP_TIMEOUT_SECONDS
  ) as response:
    if response.status != 206:
      raise OSError(
          'Expected HTTP 206 Partial Content for Range request, got '
          f'{response.status}'
      )
    chunk = response.read(expected_bytes + 1)
  if len(chunk) > expected_bytes:
    raise OSError(
        f'HTTP Range response exceeded requested {expected_bytes} bytes'
    )
  return chunk


def _fetch_wav_prefix_bytes(
    url: str,
    max_seconds: float | None = None,
    opener: urllib.request.OpenerDirector | None = None,
) -> bytes:
  """Downloads a WAV stimulus (or prefix) via HTTP Range requests."""
  if max_seconds is not None and (
      not math.isfinite(max_seconds) or max_seconds < 0
  ):
    raise ValueError(
        f'max_seconds must be non-negative and finite, got {max_seconds}'
    )
  active_opener = opener or utils.build_safe_url_opener()
  header_probe = _read_bounded_range(
      active_opener, url, 0, _WAV_HEADER_PROBE_BYTES - 1
  )

  if (
      len(header_probe) < 44
      or header_probe[:4] != b'RIFF'
      or header_probe[8:12] != b'WAVE'
  ):
    raise ValueError('Remote stimulus does not start with a valid WAV header.')

  pos = 12
  byte_rate = 0
  block_align = 0
  data_header_pos = None
  orig_data_size = 0
  while pos + 8 <= len(header_probe):
    chunk_id = header_probe[pos : pos + 4]
    chunk_size = struct.unpack('<I', header_probe[pos + 4 : pos + 8])[0]
    if chunk_id == b'fmt ' and pos + 24 <= len(header_probe):
      byte_rate = struct.unpack('<I', header_probe[pos + 16 : pos + 20])[0]
      block_align = struct.unpack('<H', header_probe[pos + 20 : pos + 22])[0]
    elif chunk_id == b'data':
      data_header_pos = pos
      orig_data_size = chunk_size
      break
    pos += 8 + chunk_size + (chunk_size % 2)

  if data_header_pos is None or byte_rate == 0 or block_align == 0:
    raise ValueError('Remote WAV header is missing fmt or data chunk metadata.')

  data_offset = data_header_pos + 8
  target_data_bytes = orig_data_size
  if max_seconds is not None:
    wanted = int(math.ceil(max_seconds * byte_rate))
    wanted = ((wanted + block_align - 1) // block_align) * block_align
    target_data_bytes = min(orig_data_size, wanted)
  if target_data_bytes > _MAX_WAV_BYTES:
    raise ValueError(
        f'Requested WAV payload ({target_data_bytes} bytes) exceeds limit.'
    )

  if target_data_bytes == 0:
    pcm_bytes = b''
  elif data_offset + target_data_bytes <= len(header_probe):
    pcm_bytes = header_probe[data_offset : data_offset + target_data_bytes]
  else:
    pcm_bytes = _read_bounded_range(
        active_opener, url, data_offset, data_offset + target_data_bytes - 1
    )
  if len(pcm_bytes) != target_data_bytes:
    raise OSError(
        f'Truncated HTTP Range response: expected {target_data_bytes} bytes, '
        f'got {len(pcm_bytes)}'
    )

  prefix = bytearray(header_probe[:data_offset])
  prefix[4:8] = struct.pack('<I', data_offset - 8 + len(pcm_bytes))
  prefix[data_header_pos + 4 : data_header_pos + 8] = struct.pack(
      '<I', len(pcm_bytes)
  )
  return bytes(prefix) + pcm_bytes


class HuthLabDataset:
  """Loads HuthLab encoding model scaling laws data.

  This class handles loading fMRI responses, audio stimuli, and
  story-level metadata from the HuthLab dataset.
  """

  def __init__(
      self,
      base_path: str | None = None,
      subject: str = 'S03',
      tr_duration: float = DEFAULT_TR_DURATION,
      train_stories: Sequence[str] | None = None,
      test_stories: Sequence[str] | None = None,
      streaming: bool = False,
  ):
    """Initializes the HuthLabDataset.

    Args:
      base_path: Root path to the data. If None, uses the --huthlab_data_path
        flag, or falls back to --dataset_basepath / DEFAULT_BASE_PATH (or the
        bundled debug responses directory when `streaming` is True).
      subject: Subject identifier (e.g., 'S01', 'S02', 'S03').
      tr_duration: Duration of one fMRI TR in seconds.
      train_stories: Optional subset of training stories to use.
      test_stories: Optional subset of test stories to use.
      streaming: If True, defaults to bundled debug responses when no base path
        is provided and fetches missing stimulus WAV prefixes from OpenNeuro.
    """
    if base_path is None:
      base_path = _HUTHLAB_DATA_PATH.value
    default_basepath = (
        _DEBUG_TESTDATA_PATH if streaming else DEFAULT_BASE_PATH
    )
    self.base_path = dataset.get_base_path(
        base_path, default_basepath=default_basepath
    )
    if DEFAULT_BASE_PATH is not None and self.base_path == DEFAULT_BASE_PATH:
      streaming = False
    self.subject = subject
    self.tr_duration = tr_duration
    self.streaming = streaming
    self._train_stories = (
        tuple(train_stories) if train_stories is not None else None
    )
    self._test_stories = (
        tuple(test_stories) if test_stories is not None else None
    )
    self._resp = None

  def _path(self, *args: str) -> str:
    return os.path.join(self.base_path, *args)

  def load_responses(self) -> dict[str, np.ndarray]:
    """Loads fMRI responses (averaged across sessions).

    Returns:
      Dict of story name to fMRI responses, each of shape (T, V) where T is the
      total number of TRs (after trimming) across all stories for that split,
      and V is the number of voxels.
    """
    if self._resp is None:
      path = self._path('responses', f'UT{self.subject}_responses.jbl')
      logging.info('Loading responses from %s', path)
      # pylint: disable=g-unsafe-pickle-load
      self._resp = joblib.load(io.BytesIO(epath.Path(path).read_bytes()))
      # pylint: enable=g-unsafe-pickle-load
      logging.info(
          'Loaded responses for %d stories, first story dims: %s',
          len(self._resp),
          next(iter(self._resp.values())).shape if self._resp else 'N/A',
      )
    return self._resp

  def load_train_responses(self) -> np.ndarray:
    """Loads training fMRI responses.

    Returns:
      Array of shape (T_train, V) where T_train is the total number of
      train TRs (after trimming) across all train stories, and V is the
      number of voxels.
    """
    resp = self.load_responses()
    return np.vstack([resp[story] for story in self.get_train_stories()])

  def load_test_responses(self) -> np.ndarray:
    """Loads test fMRI responses (averaged across sessions).

    Returns:
      Array of shape (T_test, V) where T_test is the total number of
      test TRs (after trimming) across all test stories, and V is the
      number of voxels.
    """
    resp = self.load_responses()
    return np.vstack([resp[story][40:] for story in self.get_test_stories()])

  def get_train_stories(self) -> Sequence[str]:
    """Returns the list of training story names."""
    if self._train_stories is not None:
      return self._train_stories
    return TRAIN_STORIES[self.subject]

  def get_test_stories(self) -> Sequence[str]:
    """Returns the list of test story names."""
    if self._test_stories is not None:
      return self._test_stories
    return TEST_STORIES[self.subject]

  def get_audio_path(self, story: str) -> str:
    """Returns the path to the audio file for a story."""
    return self._path('stimuli', f'{story}.wav')

  def get_sound(self, story: str) -> types.Sound:
    """Loads the audio stimulus for a story as a Sound object."""
    if not _STORY_NAME_RE.fullmatch(story):
      raise ValueError(f'Invalid HuthLab story name: {story!r}')
    audio_path = self.get_audio_path(story)
    if epath.Path(audio_path).exists():
      waveform, sample_rate = utils.read_audio(audio_path)
    elif not self.streaming:
      raise FileNotFoundError(
          f'HuthLab audio file not found at {audio_path}.'
      )
    else:
      max_seconds = None
      if epath.Path(
          self._path('responses', f'UT{self.subject}_responses.jbl')
      ).exists():
        responses = self.load_responses()
        if story in responses:
          max_seconds = (10 + responses[story].shape[0]) * self.tr_duration
      url = f'{_OPENNEURO_STIMULI_URL}/{story}.wav'
      logging.info(
          'Downloading HuthLab stimulus %s from %s (max_seconds=%s)',
          story,
          url,
          max_seconds,
      )
      wav_bytes = _fetch_wav_prefix_bytes(url, max_seconds=max_seconds)
      waveform, sample_rate = utils.wav_bytes_to_waveform(wav_bytes)
    context = types.SoundContextParams(
        id=story,
        sample_rate=sample_rate,
        length=len(waveform),
        waveform_end_second=(
            len(waveform) / sample_rate if sample_rate > 0 else 0.0
        ),
        language='en',
        language_name='English',
    )
    return types.Sound(waveform=waveform, context=context)  # pyrefly: ignore[bad-argument-type]
