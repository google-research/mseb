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

"""Tests for the track module.

Covers the track primitives defined in `mseb.track`: the `Size` enum, the
`Track` dataclass and `add_sized_variants`. The shipped tracks and the track
registry live in `mseb.tracks` and are covered by `mseb/tracks/tracks_test.py`.

The tests here build synthetic tracks rather than using the shipped ones, so
that they exercise the primitives in isolation and do not fail when a track
definition changes. They do depend on the real task registry, because
`Track.list_tasks` filters its entries against it.
"""

import dataclasses

from absl.testing import absltest
from mseb import task as task_lib
from mseb import track
import mseb.tasks  # pylint: disable=unused-import  Registers all MSEB tasks.

# Track entries are plain strings that `Track.list_tasks` filters against the
# MSEB task registry, so exercising that filter needs names on both sides of
# it: two that are really registered, and one that is not.
_REGISTERED_ENTRIES = (
    'BirdsetHSNClassification',
    'SVQEnUsPassageInLangRetrieval',
)
_UNREGISTERED_ENTRY = 'NoSuchTask'

_FILTER_TRACK = track.Track(
    name='filtering',
    description='Synthetic track used to exercise registry filtering.',
    tasks_by_size={
        track.Size.FULL: tuple(
            sorted(_REGISTERED_ENTRIES + (_UNREGISTERED_ENTRY,))
        ),
    },
)

# One registered entry per size, so that size selection can be tested without
# the filter removing anything.
_SIZED_TRACK = track.Track(
    name='sized',
    description='Synthetic track with one registered entry per size.',
    tasks_by_size={
        track.Size.DEBUG: ('SVQEnUsPassageInLangRetrievalDebug',),
        track.Size.COMPACT: ('SVQEnUsPassageInLangRetrievalCompact',),
        track.Size.FULL: ('SVQEnUsPassageInLangRetrieval',),
    },
)


class RegistryPreconditionTest(absltest.TestCase):
  """Guards the assumptions the synthetic tracks are built on.

  Every filtering test below is vacuous if these names drift, so failures are
  reported here instead of as confusing assertion errors elsewhere.
  """

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.registered = frozenset(task_lib.get_name_to_task())

  def test_registry_is_populated(self):
    self.assertNotEmpty(self.registered)

  def test_expected_entries_are_registered(self):
    for entry in _REGISTERED_ENTRIES:
      with self.subTest(task=entry):
        self.assertIn(entry, self.registered)

  def test_sized_track_entries_are_registered(self):
    for size, entries in _SIZED_TRACK.tasks_by_size.items():
      for entry in entries:
        with self.subTest(size=size, task=entry):
          self.assertIn(entry, self.registered)

  def test_unregistered_entry_is_really_unregistered(self):
    self.assertNotIn(_UNREGISTERED_ENTRY, self.registered)


class SizeTest(absltest.TestCase):

  def test_has_the_three_documented_sizes(self):
    self.assertCountEqual(
        [size.name for size in track.Size], ('DEBUG', 'COMPACT', 'FULL')
    )

  def test_names_match_the_task_name_suffixes(self):
    # `add_sized_variants` builds suffixes out of these names, so their
    # capitalization is part of the contract.
    self.assertEqual(
        [size.name.lower().capitalize() for size in track.Size],
        ['Debug', 'Compact', 'Full'],
    )


class TrackDataclassTest(absltest.TestCase):

  def test_stores_name_and_description(self):
    self.assertEqual(_FILTER_TRACK.name, 'filtering')
    self.assertIn('Synthetic track', _FILTER_TRACK.description)

  def test_is_frozen(self):
    with self.assertRaises(dataclasses.FrozenInstanceError):
      _FILTER_TRACK.name = 'changed'

  def test_sizes_matches_tasks_by_size_keys(self):
    self.assertCountEqual(
        _SIZED_TRACK.sizes(), _SIZED_TRACK.tasks_by_size.keys()
    )

  def test_sizes_returns_a_list(self):
    self.assertIsInstance(_SIZED_TRACK.sizes(), list)


class ListTasksFilteringTest(absltest.TestCase):
  """Covers the registry filtering `list_tasks` applies to its entries."""

  def test_registered_entries_are_kept(self):
    result = _FILTER_TRACK.list_tasks(track.Size.FULL)
    for entry in _REGISTERED_ENTRIES:
      self.assertIn(entry, result)

  def test_unregistered_entries_are_dropped(self):
    # The name vanishes from the result rather than raising. This is why the
    # shipped-track consistency tests inspect `tasks_by_size` directly.
    result = _FILTER_TRACK.list_tasks(track.Size.FULL)
    self.assertIn(
        _UNREGISTERED_ENTRY, _FILTER_TRACK.tasks_by_size[track.Size.FULL]
    )
    self.assertNotIn(_UNREGISTERED_ENTRY, result)

  def test_dropped_entry_is_logged(self):
    with self.assertLogs(level='WARNING') as logs:
      _FILTER_TRACK.list_tasks(track.Size.FULL)
    self.assertTrue(
        any(_UNREGISTERED_ENTRY in line for line in logs.output),
        msg=(
            f'{_UNREGISTERED_ENTRY} was dropped without a warning: '
            f'{logs.output}'
        ),
    )

  def test_registered_entries_are_not_logged(self):
    with self.assertLogs(level='WARNING') as logs:
      _FILTER_TRACK.list_tasks(track.Size.FULL)
    for entry in _REGISTERED_ENTRIES:
      self.assertFalse(
          any(entry in line for line in logs.output),
          msg=f'Registered task {entry} should not warn: {logs.output}',
      )

  def test_result_is_exactly_the_registered_subset(self):
    self.assertEqual(
        _FILTER_TRACK.list_tasks(track.Size.FULL), sorted(_REGISTERED_ENTRIES)
    )

  def test_supplied_names_define_the_universe(self):
    # `select` replaces the registry outright, so a name that is not a real
    # task is admitted if the caller says it exists.
    result = _FILTER_TRACK.list_tasks(
        track.Size.FULL, select=[_UNREGISTERED_ENTRY]
    )
    self.assertEqual(result, [_UNREGISTERED_ENTRY])

  def test_supplied_names_only_narrow_the_track(self):
    # A selected name that the track does not list stays out of the result.
    result = _SIZED_TRACK.list_tasks(
        track.Size.FULL, select=['BirdsetHSNClassification']
    )
    self.assertEmpty(result)

  def test_supplied_names_can_filter_everything_out(self):
    result = _FILTER_TRACK.list_tasks(track.Size.FULL, select=['Unrelated'])
    self.assertEmpty(result)

  def test_empty_supplied_names_can_filter_everything_out(self):
    result = _FILTER_TRACK.list_tasks(track.Size.FULL, select=[])
    self.assertEmpty(result)


class ListTasksSizeTest(absltest.TestCase):

  def test_returns_the_entries_of_the_requested_size(self):
    for size, entries in _SIZED_TRACK.tasks_by_size.items():
      with self.subTest(size=size):
        self.assertEqual(_SIZED_TRACK.list_tasks(size), sorted(entries))

  def test_default_size_is_full(self):
    self.assertEqual(
        _SIZED_TRACK.list_tasks(), _SIZED_TRACK.list_tasks(track.Size.FULL)
    )

  def test_returns_sorted_list(self):
    result = _FILTER_TRACK.list_tasks(track.Size.FULL)
    self.assertIsInstance(result, list)
    self.assertEqual(result, sorted(result))

  def test_unknown_size_raises(self):
    # `_FILTER_TRACK` only defines FULL.
    with self.assertRaises(ValueError):
      _FILTER_TRACK.list_tasks(track.Size.DEBUG)

  def test_unknown_size_error_names_the_track(self):
    with self.assertRaisesRegex(ValueError, 'filtering'):
      _FILTER_TRACK.list_tasks(track.Size.DEBUG)


class DescribeTest(absltest.TestCase):

  def test_includes_name_and_description(self):
    desc = _SIZED_TRACK.describe()
    self.assertIn('sized', desc)
    self.assertIn(_SIZED_TRACK.description, desc)

  def test_includes_all_size_labels(self):
    desc = _SIZED_TRACK.describe()
    for size in ('debug', 'compact', 'full'):
      self.assertIn(f'{size}:', desc)

  def test_counts_raw_entries_not_resolved_tasks(self):
    # `describe` reports `len(tasks)` on the unfiltered entry tuple, so its
    # counts are an upper bound on what `list_tasks` returns. The label says
    # "task names" rather than "tasks" for exactly that reason.
    desc = _FILTER_TRACK.describe()
    raw = len(_FILTER_TRACK.tasks_by_size[track.Size.FULL])
    self.assertIn(f'full: {raw} task names', desc)
    self.assertLen(_FILTER_TRACK.list_tasks(track.Size.FULL), raw - 1)


class AddSizedVariantsTest(absltest.TestCase):
  """Tests for `add_sized_variants`.

  This is what lets a caller pass unsuffixed task names alongside `--size`:
  each input name is kept, and the `Debug`/`Compact` variants are appended
  when they exist in the registry. `list_tasks` then intersects the result
  with the requested track and size.
  """

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.registered = frozenset(task_lib.get_name_to_task())

  def test_appends_registered_sized_variants(self):
    base = 'SVQEnUsPassageInLangRetrieval'
    # Preconditions, so a registry change shows up as a clear failure here
    # rather than as a confusing assertion below.
    self.assertIn(base, self.registered)
    self.assertIn(f'{base}Debug', self.registered)
    self.assertIn(f'{base}Compact', self.registered)

    result = track.add_sized_variants([base])
    self.assertCountEqual(result, [base, f'{base}Debug', f'{base}Compact'])

  def test_original_names_are_always_kept(self):
    # Even a name with no sized variants, and even one that is not a task at
    # all: the function only ever adds.
    names = ['BirdsetHSNClassification', _UNREGISTERED_ENTRY]
    result = track.add_sized_variants(names)
    self.assertContainsSubset(names, result)

  def test_unregistered_name_gains_no_variants(self):
    self.assertEqual(
        track.add_sized_variants([_UNREGISTERED_ENTRY]), [_UNREGISTERED_ENTRY]
    )

  def test_empty_input(self):
    self.assertEmpty(track.add_sized_variants([]))

  def test_does_not_mutate_caller_list(self):
    names = ['SVQEnUsPassageInLangRetrieval']
    track.add_sized_variants(names)
    self.assertEqual(names, ['SVQEnUsPassageInLangRetrieval'])

  def test_accepts_a_tuple(self):
    self.assertIsInstance(
        track.add_sized_variants((_UNREGISTERED_ENTRY,)), list
    )

  def test_every_added_name_is_registered(self):
    names = [
        'SVQEnUsPassageInLangRetrieval',
        'BirdsetHSNClassification',
        _UNREGISTERED_ENTRY,
    ]
    added = set(track.add_sized_variants(names)) - set(names)
    self.assertContainsSubset(added, self.registered)

  def test_does_not_double_suffix(self):
    # The implementation appends to the list it is iterating, so the appended
    # variants are themselves visited and probed for a second suffix.
    doubly_suffixed = sorted(
        name
        for name in self.registered
        if any(
            name.endswith(f'{first}{second}')  # pylint: disable=g-complex-comprehension
            for first in ('Debug', 'Compact')
            for second in ('Debug', 'Compact')
        )
    )
    self.assertEmpty(
        doubly_suffixed,
        msg=(
            'add_sized_variants appends while iterating, so a task name'
            ' ending in two size suffixes would make it compound suffixes:'
            f' {doubly_suffixed}'
        ),
    )

  def test_feeds_list_tasks(self):
    # End-to-end shape of the intended call: unsuffixed names in, the track's
    # sized entries out.
    base = 'SVQEnUsPassageInLangRetrieval'
    result = _SIZED_TRACK.list_tasks(
        track.Size.COMPACT, select=track.add_sized_variants([base])
    )
    self.assertEqual(result, [f'{base}Compact'])


if __name__ == '__main__':
  absltest.main()
