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

"""Unit tests for the PEP 544 protocol-based lazy registry."""

import abc
import copy
import os
import pickle
import sys
import types
import typing
from typing import Any, Protocol, runtime_checkable

from absl.testing import absltest
from mseb import registry


@runtime_checkable
class SampleProcessor(Protocol):

  @property
  def name(self) -> str:
    ...

  def __call__(self, text: str) -> str:
    ...


class NonRuntimeProtocol(Protocol):

  def __call__(self, text: str) -> str:
    ...


class RegularClass:
  pass


class AbstractClass(abc.ABC):

  @abc.abstractmethod
  def __call__(self, text: str) -> str:
    ...


class ValidProcessor:
  """Satisfies SampleProcessor structurally without inheriting."""

  def __init__(
      self,
      prefix: str = '',
      *,
      lower: bool = True,
      data: list[dict[str, int]] | None = None,
  ) -> None:
    self._prefix, self._lower = prefix, lower
    self.data: list[dict[str, int]] = [] if data is None else data

  @property
  def name(self) -> str:
    return 'valid'

  def __call__(self, text: str) -> str:
    return f'{self._prefix}{text.lower() if self._lower else text}'


class AlternateProcessor:
  """Distinct valid processor class for conflict testing."""

  name = 'alt'

  def __call__(self, text: str) -> str:
    return text.upper()


class InvalidProcessor:
  """Missing __call__, so instances fail SampleProcessor validation."""

  name = 'invalid'


NOT_A_CLASS = 'constant_value'
VALID_SHORT_NAMES = (
    'processors',
    'p',
    '_private',
    'Whisper.V1',
    'parakeet_tdt_0.6b_v3',
    'wav2vec2-large-960h',
    'asr+llm#1@v2',
    '_local.v1+test',
)
INVALID_PLAIN_SHORT_NAMES = (
    '',
    '   ',
    ' Pad ',
    'a b',
    'foo\n',
    'µ',
    'a·b',
    '1bad',
    '0.6b',
    '-bad',
    '.bad',
    '+bad',
    '#bad',
    '@bad',
    '.',
    '---',
    'Bad)Name',
    'a/b',
    'a:b',
)
INVALID_SHORT_NAMES = INVALID_PLAIN_SHORT_NAMES + (
    'Preset.V1()',
    "Preset.V1(prefix='x:')",
)


class RegistryTest(absltest.TestCase):

  def test_public_exports_and_minimal_surface(self) -> None:
    self.assertEqual(registry.__all__, ['Registry'])

  def test_init_validates_name_and_runtime_checkable_protocol(self) -> None:
    for valid_name in VALID_SHORT_NAMES:
      self.assertEqual(
          registry.Registry(valid_name, SampleProcessor).entries(), []
      )
    for bad_name in INVALID_SHORT_NAMES:
      with self.assertRaises(ValueError):
        registry.Registry(bad_name, SampleProcessor)
    bad_val: Any = ValidProcessor()  # Bypass static check for runtime error.
    for proto in (NonRuntimeProtocol, RegularClass, AbstractClass, bad_val):
      with self.assertRaises(TypeError):
        registry.Registry('bad', proto)

  def test_lazy_import_deferred_until_get_or_build(self) -> None:
    temp_dir = self.create_tempdir().full_path
    mod_name = '_mseb_lazy_test_mod'
    with open(
        os.path.join(temp_dir, f'{mod_name}.py'), 'w', encoding='utf-8'
    ) as f:
      f.write(
          'class DiskProcessor:\n'
          '  @property\n  def name(self) -> str:\n    return "disk"\n'
          '  def __call__(self, text: str) -> str:\n    return text.strip()\n'
      )
    sys.path.insert(0, temp_dir)
    self.addCleanup(sys.path.remove, temp_dir)
    self.addCleanup(sys.modules.pop, mod_name, None)

    reg = registry.Registry('processors', SampleProcessor)
    reg.register('Disk.V1', f'{mod_name}.DiskProcessor')
    self.assertNotIn(mod_name, sys.modules)
    self.assertEqual(reg.entries(), ['Disk.V1'])
    self.assertNotIn(mod_name, sys.modules)

    cls = reg.get('Disk.V1')
    self.assertIn(mod_name, sys.modules)
    self.assertEqual(cls.__name__, 'DiskProcessor')
    proc = reg.build('Disk.V1')
    self.assertEqual((proc.name, proc('  hi  ')), ('disk', 'hi'))

  def test_get_returns_class_without_issubclass_and_guards_errors(self) -> None:
    reg = registry.Registry('processors', SampleProcessor)
    reg.register('Preset.V1', f"{__name__}.ValidProcessor(prefix='p:')")
    reg.register('Invalid.V1', f'{__name__}.InvalidProcessor')
    reg.register('MissingMod.V1', 'mseb._no_such_module_xyz.Cls')
    reg.register('MissingAttr.V1', f'{__name__}.NoSuchClass')
    reg.register('NonClass.V1', f'{__name__}.NOT_A_CLASS')

    cls = reg.get('Preset.V1')
    self.assertIs(cls, ValidProcessor)
    self.assertEqual(cls.__bases__, (object,))
    self.assertNotIn(SampleProcessor, cls.__mro__)
    self.assertIs(reg.get('Invalid.V1'), InvalidProcessor)

    for bad_name in INVALID_SHORT_NAMES:
      with self.assertRaises(KeyError):
        reg.get(bad_name)
    with self.assertRaisesRegex(KeyError, 'Invalid.V1.*Preset.V1'):
      reg.get('Unknown.V1')
    with self.assertRaises(ModuleNotFoundError):
      reg.get('MissingMod.V1')
    with self.assertRaises(AttributeError):
      reg.get('MissingAttr.V1')
    with self.assertRaises(TypeError):
      reg.get('NonClass.V1')

  def test_build_positional_kwargs_precedence_and_protocol_check(self) -> None:
    reg = registry.Registry('processors', SampleProcessor)
    reg.register(
        'Cfg.V1', f"{__name__}.ValidProcessor(prefix='preset:', lower=True)"
    )
    reg.register('Plain.V1', f'{__name__}.ValidProcessor')
    reg.register('Invalid.V1', f'{__name__}.InvalidProcessor')

    pos = reg.build('Plain.V1', 'pos:')
    self.assertIsInstance(pos, SampleProcessor)
    self.assertEqual(type(pos).__bases__, (object,))
    self.assertEqual(pos('AbC'), 'pos:abc')

    self.assertEqual(reg.build('Cfg.V1')('AbC'), 'preset:abc')
    self.assertEqual(
        reg.build("Cfg.V1(prefix='inline:', lower=False)")('AbC'), 'inline:AbC'
    )
    self.assertEqual(
        reg.build("Cfg.V1(\n  prefix='m:',  # inline comment\n)")('AbC'),
        'm:abc',
    )
    self.assertEqual(
        reg.build('Cfg.V1(\n  # comment only\n)')('AbC'), 'preset:abc'
    )
    self.assertEqual(
        reg.build("Cfg.V1(prefix='inline:', lower=False)", prefix='kw:')('AbC'),
        'kw:AbC',
    )
    reg.register('Mut.V1', f"{__name__}.ValidProcessor(data=[{{'k': 1}}])")
    first = reg.build('Mut.V1')
    assert isinstance(first, ValidProcessor)
    first.data[0]['k'] = 2
    first.data.append({'k': 3})
    second = reg.build('Mut.V1')
    assert isinstance(second, ValidProcessor)
    self.assertEqual(second.data, [{'k': 1}])
    with self.assertRaisesRegex(TypeError, 'SampleProcessor'):
      reg.build('Invalid.V1')
    unknown_calls = INVALID_PLAIN_SHORT_NAMES + (
        '(k=1)',
        'Cfg.V1 (k=1)',
        ' Cfg.V1(k=1)',
        '1bad(k=1)',
        '-bad(k=1)',
        'Cfg)V1(k=1)',
    )
    for unknown_call in unknown_calls:
      with self.assertRaises(KeyError):
        reg.build(unknown_call)
    malformed_calls = (
        'Cfg.V1(k=1) ',
        'Cfg.V1(',
        'Cfg.V1()()',
        "Cfg.V1('p')",
        'Cfg.V1(**d)',
        'Cfg.V1(k=v)',
        'Cfg.V1(a=1,a=2)',
    )
    for malformed in malformed_calls:
      with self.assertRaises(ValueError):
        reg.build(malformed)

  def test_pickle_and_deepcopy_registry_and_bound_build(self) -> None:
    reg = registry.Registry('processors', SampleProcessor)
    reg.register(
        'Cfg.V1', f"{__name__}.ValidProcessor(prefix='preset:', data=[])"
    )
    demo_module = types.ModuleType('demo_pickle_processors')
    reg.bind(demo_module)

    for target in (reg, reg.build, demo_module.build):
      self.assertNotEmpty(pickle.dumps(target))
    restored_reg = copy.deepcopy(reg)
    self.assertEqual(restored_reg.entries(), ['Cfg.V1'])
    self.assertEqual(restored_reg.build('Cfg.V1')('AbC'), 'preset:abc')
    restored_build = copy.deepcopy(reg.build)
    self.assertEqual(restored_build('Cfg.V1')('AbC'), 'preset:abc')
    restored_mod_build = copy.deepcopy(demo_module.build)
    self.assertEqual(restored_mod_build('Cfg.V1')('AbC'), 'preset:abc')

  def test_register_idempotent_vs_conflicting_and_validation(self) -> None:
    reg = registry.Registry('processors', SampleProcessor)
    spec_a = f"{__name__}.ValidProcessor(prefix='a:', lower=False)"
    reg.register('B.V1', f'{__name__}.AlternateProcessor')
    reg.register('A.V1', spec_a)
    reg.register('A.V1', spec_a)
    reg.register('A.V1', f"{__name__}.ValidProcessor(lower=False, prefix='a:')")
    reg.register(
        'Multi.V1',
        f"{__name__}.ValidProcessor(\n  prefix='ml:',  # preset comment\n)",
    )
    self.assertEqual(reg.entries(), ['A.V1', 'B.V1', 'Multi.V1'])
    self.assertEqual(reg.build('Multi.V1')('Ab'), 'ml:ab')
    with self.assertRaises(ValueError):
      reg.register('A.V1', f'{__name__}.AlternateProcessor')
    with self.assertRaises(ValueError):
      reg.register(
          'A.V1', f"{__name__}.ValidProcessor(prefix='b:', lower=False)"
      )
    valid_path = f'{__name__}.ValidProcessor'
    valid_reg = registry.Registry('valid_names', SampleProcessor)
    for valid_name in VALID_SHORT_NAMES:
      valid_reg.register(valid_name, valid_path)
      self.assertIs(valid_reg.get(valid_name), ValidProcessor)
      self.assertEqual(
          valid_reg.build(f"{valid_name}(prefix='x:')")('Ab'), 'x:ab'
      )
      valid_reg.unregister(valid_name)
    self.assertEqual(valid_reg.entries(), [])
    for bad_alias in INVALID_SHORT_NAMES:
      with self.assertRaises(ValueError):
        reg.register(bad_alias, valid_path)
    for bad_path in ('NoDot', '1a.B', 'a..B', 'a.1B', 'a-b.Cls', ''):
      with self.assertRaisesRegex(ValueError, 'class_path'):
        reg.register('Bad.V1', bad_path)
    for bad_call_path in ("a.B('p')", 'a.B(k=v)', 'a.B(a=1,a=2)'):
      with self.assertRaises(ValueError):
        reg.register('Bad.V1', bad_call_path)

  def test_unregister_lazy_reregister_and_validation(self) -> None:
    mod_name = 'mseb._unloaded_unregister_target'
    reg = registry.Registry('processors', SampleProcessor)
    with self.assertRaisesRegex(KeyError, '<empty>'):
      reg.unregister('Lazy.V1')
    reg.register('Keep.V1', f'{__name__}.ValidProcessor')
    reg.register('Lazy.V1', f'{mod_name}.MissingCls')
    reg.unregister('Lazy.V1')
    self.assertNotIn(mod_name, sys.modules)
    self.assertEqual(reg.entries(), ['Keep.V1'])
    for op in (reg.get, reg.build, reg.unregister):
      with self.assertRaisesRegex(KeyError, 'Keep.V1'):
        op('Lazy.V1')
    reg.register('Lazy.V1', f'{__name__}.AlternateProcessor')
    self.assertEqual(reg.build('Lazy.V1')('ab'), 'AB')
    reg.unregister('Lazy.V1')
    reg.register('Lazy.V1', f"{__name__}.ValidProcessor(prefix='v2:')")
    self.assertEqual(reg.build('Lazy.V1')('Ab'), 'v2:ab')
    for bad in INVALID_SHORT_NAMES:
      with self.assertRaises(KeyError):
        reg.unregister(bad)

  def test_bind_attaches_helpers_guards_getattr_and_asserts_types(self) -> None:
    reg = registry.Registry('processors', SampleProcessor)
    demo_module = types.ModuleType('demo_processors')
    getattr_fn = reg.bind(demo_module)

    expected = {'register', 'unregister', 'get', 'build', 'entries'}
    self.assertTrue(expected.issubset(dir(demo_module)))
    for method_name in expected:
      self.assertNotIn(method_name, demo_module.__dict__)
    raw_getattr: Any = getattr_fn  # Bypass @overload Literal for runtime check.
    for leaked in ('bind', '_entries', '_name', '_protocol', 'list'):
      self.assertFalse(hasattr(demo_module, leaked))
      with self.assertRaises(AttributeError):
        raw_getattr(leaked)

    register_fn = getattr_fn('register')
    unregister_fn = getattr_fn('unregister')
    get_fn, build_fn = getattr_fn('get'), getattr_fn('build')
    entries_fn = getattr_fn('entries')
    path = f'{__name__}.ValidProcessor'
    typing.assert_type(register_fn('Valid.V1', path), None)
    typing.assert_type(unregister_fn('Valid.V1'), None)
    typing.assert_type(register_fn(name='Valid.V1', class_path=path), None)
    typing.assert_type(unregister_fn(name='Valid.V1'), None)
    demo_module.register('Valid.V1', path)
    demo_module.register('Temp.V1', path)
    demo_module.unregister('Temp.V1')
    typing.assert_type(entries_fn(), list[str])
    typing.assert_type(get_fn('Valid.V1'), type[SampleProcessor])
    typing.assert_type(get_fn(name='Valid.V1'), type[SampleProcessor])
    proc_pos = build_fn('Valid.V1', '[pos] ')
    proc_kw = build_fn(name='Valid.V1', prefix='[kw] ', lower=False)
    typing.assert_type(proc_pos, SampleProcessor)
    typing.assert_type(proc_kw, SampleProcessor)

    self.assertEqual(demo_module.entries(), ['Valid.V1'])
    self.assertIs(demo_module.get('Valid.V1'), ValidProcessor)
    self.assertEqual((proc_pos('Hi'), proc_kw('Hi')), ('[pos] hi', '[kw] Hi'))


if __name__ == '__main__':
  absltest.main()
