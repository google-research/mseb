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

"""Tests for the MSEB track registry and the shipped tracks.

Covers `mseb.tracks`: the registry itself, its lookup helpers, and the content
of the shipped ASR, generative and retrieval tracks. The track primitives
(`Size`, `Track`, `add_sized_variants`) live in `mseb.track` and are covered by
`mseb/track_test.py`.
"""

from absl.testing import absltest
from mseb import task as task_lib
from mseb import track as track_lib
from mseb import tracks
import mseb.tasks  # pylint: disable=unused-import  Registers all MSEB tasks.
from mseb.tracks import asr
from mseb.tracks import constants
from mseb.tracks import generative
from mseb.tracks import retrieval

_SHIPPED_TRACK_NAMES = ('asr', 'generative', 'retrieval')


class RegistryTest(absltest.TestCase):

  def test_registry_has_all_shipped_tracks(self):
    self.assertCountEqual(tracks.registry, _SHIPPED_TRACK_NAMES)

  def test_registry_keys_match_track_names(self):
    for name, mseb_track in tracks.registry.items():
      with self.subTest(track=name):
        self.assertEqual(name, mseb_track.name)

  def test_registry_values_are_tracks(self):
    for name, mseb_track in tracks.registry.items():
      with self.subTest(track=name):
        self.assertIsInstance(mseb_track, track_lib.Track)

  def test_module_constants_are_registered(self):
    for mseb_track in (asr.ASR, generative.GENERATIVE, retrieval.RETRIEVAL):
      with self.subTest(track=mseb_track.name):
        self.assertIs(tracks.registry[mseb_track.name], mseb_track)


class ListTracksTest(absltest.TestCase):

  def test_returns_sorted_registry_names(self):
    self.assertEqual(tracks.list_tracks(), sorted(tracks.registry))

  def test_returns_a_list(self):
    self.assertIsInstance(tracks.list_tracks(), list)


class GetTrackByNameTest(absltest.TestCase):

  def test_returns_the_named_track(self):
    self.assertIs(tracks.get_track_by_name('retrieval'), retrieval.RETRIEVAL)

  def test_round_trips_every_track(self):
    for name in tracks.list_tracks():
      with self.subTest(track=name):
        self.assertEqual(tracks.get_track_by_name(name).name, name)

  def test_unknown_track_raises(self):
    with self.assertRaises(ValueError):
      tracks.get_track_by_name('nonexistent')

  def test_unknown_track_error_lists_the_available_tracks(self):
    with self.assertRaisesRegex(ValueError, 'retrieval'):
      tracks.get_track_by_name('nonexistent')


class TrackMetadataTest(absltest.TestCase):

  def test_every_track_has_a_description(self):
    for name, mseb_track in tracks.registry.items():
      with self.subTest(track=name):
        self.assertNotEmpty(mseb_track.description)

  def test_every_track_has_three_sizes(self):
    for name, mseb_track in tracks.registry.items():
      with self.subTest(track=name):
        self.assertCountEqual(
            [size.name.lower() for size in mseb_track.tasks_by_size],
            ('debug', 'compact', 'full'),
        )

  def test_sizes_matches_tasks_by_size_keys(self):
    for name, mseb_track in tracks.registry.items():
      with self.subTest(track=name):
        self.assertCountEqual(
            mseb_track.sizes(), mseb_track.tasks_by_size.keys()
        )

  def test_describe_includes_the_track_name(self):
    for name, mseb_track in tracks.registry.items():
      with self.subTest(track=name):
        self.assertIn(name, mseb_track.describe())

  def test_describe_includes_all_size_labels(self):
    for name, mseb_track in tracks.registry.items():
      for size in ('debug', 'compact', 'full'):
        with self.subTest(track=name, size=size):
          self.assertIn(f'{size}:', mseb_track.describe())


class ListTasksTest(absltest.TestCase):
  """Covers `Track.list_tasks` as applied to the shipped tracks."""

  def test_returns_sorted_list(self):
    for name, mseb_track in tracks.registry.items():
      for size in mseb_track.sizes():
        with self.subTest(track=name, size=size):
          result = mseb_track.list_tasks(size)
          self.assertIsInstance(result, list)
          self.assertEqual(result, sorted(result))

  def test_default_size_is_full(self):
    for name, mseb_track in tracks.registry.items():
      with self.subTest(track=name):
        self.assertEqual(
            mseb_track.list_tasks(), mseb_track.list_tasks(track_lib.Size.FULL)
        )

  def test_has_no_duplicates(self):
    for name, mseb_track in tracks.registry.items():
      for size in mseb_track.sizes():
        with self.subTest(track=name, size=size):
          listed = mseb_track.list_tasks(size)
          self.assertLen(set(listed), len(listed))

  def test_result_is_always_a_subset_of_entries(self):
    for name, mseb_track in tracks.registry.items():
      for size, entries in mseb_track.tasks_by_size.items():
        with self.subTest(track=name, size=size):
          self.assertContainsSubset(mseb_track.list_tasks(size), entries)


class TrackContentTest(absltest.TestCase):
  """Spot-checks that each track contains the tasks it is supposed to."""

  def test_retrieval_contains_passage_in_lang_at_every_size(self):
    for size, expected in (
        (track_lib.Size.DEBUG, 'SVQEnUsPassageInLangRetrievalDebug'),
        (track_lib.Size.COMPACT, 'SVQEnUsPassageInLangRetrievalCompact'),
        (track_lib.Size.FULL, 'SVQEnUsPassageInLangRetrieval'),
    ):
      with self.subTest(size=size):
        self.assertIn(expected, retrieval.RETRIEVAL.list_tasks(size))

  def test_generative_contains_transcription_at_every_size(self):
    for size, expected in (
        (track_lib.Size.DEBUG, 'SVQEnUsSpeechTranscriptionDebug'),
        (track_lib.Size.COMPACT, 'SVQEnUsSpeechTranscriptionCompact'),
        (track_lib.Size.FULL, 'SVQEnUsSpeechTranscription'),
    ):
      with self.subTest(size=size):
        self.assertIn(expected, generative.GENERATIVE.list_tasks(size))

  def test_retrieval_has_no_transcription(self):
    # Transcription is a generative task; the retrieval track is for
    # embedding models.
    for size, entries in retrieval.RETRIEVAL.tasks_by_size.items():
      with self.subTest(size=size):
        self.assertEmpty([e for e in entries if 'SpeechTranscription' in e])

  def test_asr_is_transcription_only(self):
    for size, entries in asr.ASR.tasks_by_size.items():
      with self.subTest(size=size):
        for entry in entries:
          self.assertIn('SpeechTranscription', entry)

  def test_asr_is_contained_in_generative(self):
    # The ASR track is the transcription slice of the generative track.
    for size, entries in asr.ASR.tasks_by_size.items():
      with self.subTest(size=size):
        self.assertContainsSubset(
            entries, generative.GENERATIVE.tasks_by_size[size]
        )


class ConstantsTest(absltest.TestCase):

  def test_locale_lists_are_nonempty(self):
    for label, locales in (
        ('cross_lang', constants.SVQ_CROSS_LANG_LOCALES),
        ('in_lang', constants.SVQ_IN_LANG_LOCALES),
        ('svq', constants.SVQ_LOCALES),
        ('speech_massive', constants.SPEECH_MASSIVE_LOCALES),
    ):
      with self.subTest(locales=label):
        self.assertNotEmpty(locales)

  def test_locale_lists_are_tuples(self):
    # Membership tests and f-string interpolation work on any iterable, but a
    # tuple keeps iteration order stable across processes, which a set does
    # not. Every track entry list is built by iterating these.
    for label, locales in (
        ('cross_lang', constants.SVQ_CROSS_LANG_LOCALES),
        ('in_lang', constants.SVQ_IN_LANG_LOCALES),
        ('svq', constants.SVQ_LOCALES),
        ('speech_massive', constants.SPEECH_MASSIVE_LOCALES),
    ):
      with self.subTest(locales=label):
        self.assertIsInstance(locales, tuple)

  def test_locale_lists_are_sorted_and_deduplicated(self):
    for label, locales in (
        ('cross_lang', constants.SVQ_CROSS_LANG_LOCALES),
        ('in_lang', constants.SVQ_IN_LANG_LOCALES),
        ('svq', constants.SVQ_LOCALES),
        ('speech_massive', constants.SPEECH_MASSIVE_LOCALES),
    ):
      with self.subTest(locales=label):
        self.assertEqual(locales, tuple(sorted(set(locales))))

  def test_svq_locales_is_the_union(self):
    self.assertEqual(
        constants.SVQ_LOCALES,
        tuple(
            sorted(
                set(constants.SVQ_CROSS_LANG_LOCALES)
                | set(constants.SVQ_IN_LANG_LOCALES)
            )
        ),
    )


class RegistryConsistencyTest(absltest.TestCase):
  """Cross-checks every track entry against the real task registry.

  Track entries are plain strings and nothing validates them at definition
  time. Since `list_tasks` filters unknown names out, a wrong name template
  never fails a run -- it logs a warning and quietly shrinks a track. The
  shipped tracks are generated by interpolating locale lists into name
  templates, so one wrong template drops a whole family of tasks behind a line
  of log spam. These tests are the only guard against that, and they must
  therefore inspect `tasks_by_size` directly rather than `list_tasks` output.

  Assumes a fully populated registry. `mseb/tasks/__init__.py` wraps each task
  import in `try/except ImportError`, so a missing dependency would shrink the
  registry and surface here as a spurious failure; the BUILD dep on
  `//third_party/py/mseb/tasks` is what guarantees completeness.
  """

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.registered = frozenset(task_lib.get_name_to_task())

  def test_registry_is_populated(self):
    # Guards the tests below: an empty registry would make them vacuous.
    self.assertNotEmpty(self.registered)

  def test_every_entry_is_a_registered_task(self):
    for track_name, mseb_track in tracks.registry.items():
      for size, entries in mseb_track.tasks_by_size.items():
        with self.subTest(track=track_name, size=size):
          unknown = sorted(set(entries) - self.registered)
          self.assertEmpty(
              unknown,
              msg=(
                  f'Track {track_name!r} size {size!r} lists names that are'
                  f' not registered tasks: {unknown}.'
              ),
          )

  def test_every_size_resolves_to_at_least_one_task(self):
    # With filtering in place this is a real check: a size whose entries are
    # all phantom resolves to an empty task list and the run becomes a no-op.
    for track_name, mseb_track in tracks.registry.items():
      for size in mseb_track.sizes():
        with self.subTest(track=track_name, size=size):
          self.assertNotEmpty(
              mseb_track.list_tasks(size),
              msg=(
                  f'Track {track_name!r} size {size!r} resolves to no'
                  ' registered tasks at all.'
              ),
          )


if __name__ == '__main__':
  absltest.main()
