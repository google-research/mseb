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

"""Lazy, protocol-validated component registry for MSEB.

Eagerly importing concrete component implementations (encoders, tasks,
datasets, and normalizers) at registration time pulls heavy transitive
dependencies into every process on import, inflating startup latency, memory
footprint, and build dependency graphs even when callers only list aliases or
instantiate a single component. Traditional base-class registries also couple
independent implementations to shared abstract base classes.

Design details:
  - Lazy loading: `register(name, class_path)` maps a short string alias to a
    `<module_path>.<ClassName>` path (with optional literal keyword presets).
    `entries()` and `unregister()` never import target modules, and
    `importlib.import_module` executes on demand only during `get()` or
    `build()`.
  - Structural PEP 544 `Protocol` validation: Each `Registry` is parameterized
    by a `@typing.runtime_checkable` `typing.Protocol` rather than a base
    class. Registered classes need not inherit from the protocol, and
    conformance is checked via `isinstance(instance, protocol)` in `build()`
    rather than `issubclass()` in `get()` (compatible with Python 3.12+
    protocols declaring `@property` members).
  - Safe 3-tier keyword arguments: Presets in `register('A', 'pkg.Cls(k=1)')`,
    inline build call-strings in `build('A(k=2)')`, and runtime `**kwargs` in
    `build('A', k=3)` merge from lowest to highest precedence. Call-strings
    allow only AST-evaluated literals (`ast.literal_eval`), and presets are
    deep-copied per `build()` so mutable defaults are never shared.
  - Uniform module binding: Calling
    `__getattr__ = REGISTRY.bind(sys.modules[__name__])` exposes `register`,
    `unregister`, `get`, `build`, and `entries` on a domain package module via
    an `@overload`-typed PEP 562 `__getattr__` and `__dir__` so consumers use
    the domain module directly with full static type inference and without
    importing `mseb.registry`.

Example:
  # In mseb/normalizers/__init__.py:
  REGISTRY = registry.Registry('normalizers', Normalizer)
  __getattr__ = REGISTRY.bind(sys.modules[__name__])
  REGISTRY.register('Whisper.V1', 'mseb.normalizers.whisper.WhisperNormalizer')

  # Callers use `mseb.normalizers` directly without importing `mseb.registry`:
  from mseb import normalizers
  names, cls = normalizers.entries(), normalizers.get('Whisper.V1')
  norm = normalizers.build('Whisper.V1', 'en-us')
  normalizers.register('Custom.V1', 'my_pkg.CustomNormalizer(lower=True)')
  normalizers.unregister('Custom.V1')
"""

import ast
from collections.abc import Callable
import copy
import dataclasses
import importlib
import re
import types
from typing import Any, Generic, Literal, Protocol, TypeVar, overload

__all__ = ['Registry']

T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)
_EntriesFn = Callable[[], list[str]]
_BOUND_METHODS = ('register', 'unregister', 'get', 'build', 'entries')


# Unlike zero-arg `_EntriesFn`, `Callable[[str, ...], ...]` parameters are
# anonymous/positional-only; `Protocol.__call__` preserves parameter names
# (`name`, `class_path`) so keyword calls type-check statically.
class _RegisterFn(Protocol):
  """Callable signature for bound `register`."""

  def __call__(self, name: str, class_path: str) -> None:
    ...


class _UnregisterFn(Protocol):
  """Callable signature for bound `unregister`."""

  def __call__(self, name: str) -> None:
    ...


class _GetFn(Protocol[T_co]):
  """Callable signature for bound `get`."""

  def __call__(self, name: str) -> type[T_co]:
    ...


# `Callable` cannot express required `name: str` followed by `*args, **kwargs`
# (`Callable[..., T_co]` would skip static checking of `name`).
class _BuildFn(Protocol[T_co]):
  """Callable signature for bound `build`."""

  def __call__(
      self,
      name: str,
      *args: Any,  # Untouched pass-through.
      **kwargs: Any,  # Untouched pass-through.
  ) -> T_co:
    ...


# `Callable` cannot express `@overload`s; `_ModuleGetAttr` maps each `Literal`
# attribute name to its typed helper and flags `module.<typo>` statically.
class _ModuleGetAttr(Protocol[T_co]):
  """Typed PEP 562 module `__getattr__` protocol returned by `Registry.bind`."""

  @overload
  def __call__(self, name: Literal['register']) -> _RegisterFn:
    ...

  @overload
  def __call__(self, name: Literal['unregister']) -> _UnregisterFn:
    ...

  @overload
  def __call__(self, name: Literal['get']) -> _GetFn[T_co]:
    ...

  @overload
  def __call__(self, name: Literal['build']) -> _BuildFn[T_co]:
    ...

  @overload
  def __call__(self, name: Literal['entries']) -> _EntriesFn:
    ...


@dataclasses.dataclass(frozen=True)
class _Entry:
  """Registered class path and preset keyword arguments."""

  class_path: str
  kwargs: dict[str, Any]  # Opaque literal values.


_SHORT_NAME_RE = re.compile(r'[A-Za-z_][A-Za-z0-9_.+#@-]*')


def _validate_short_name(name: str) -> None:
  """Validates `name` is an ASCII identifier allowing `-.+#@`."""
  if _SHORT_NAME_RE.fullmatch(name) is None:
    raise ValueError(
        f'Invalid short name {name!r}: must start with an ASCII letter or "_"'
        ' and contain only ASCII letters, digits, "_", or "-.+#@".'
    )


def _parse_call_string(
    spec: str,
) -> tuple[str, dict[str, Any]]:  # Opaque literal values.
  """Parses `target` or `target(k=v, ...)` into `(target, literal_kwargs)`."""
  if '(' not in spec:
    return spec, {}
  if not spec.endswith(')'):
    raise ValueError(f'Malformed call specification: {spec!r}')
  base_name, args_part = spec.split('(', 1)
  inner = args_part[:-1].strip()
  if not inner:
    return base_name, {}

  try:
    expr = ast.parse(f'_call(\n{inner}\n)', mode='eval')
  except SyntaxError as exc:
    raise ValueError(f'Invalid syntax in {spec!r}: {exc}') from exc
  if (
      not isinstance(expr.body, ast.Call)
      or not isinstance(expr.body.func, ast.Name)
      or expr.body.func.id != '_call'
  ):
    raise ValueError(f'Invalid call specification: {spec!r}')
  if expr.body.args:
    raise ValueError(f'Positional arguments are not supported in {spec!r}.')

  parsed_kwargs: dict[str, Any] = {}  # Opaque literal values.
  for kw in expr.body.keywords:
    if kw.arg is None:
      raise ValueError(f'Keyword unpacking is not supported in {spec!r}.')
    if kw.arg in parsed_kwargs:
      raise ValueError(f'Duplicate keyword argument {kw.arg!r} in {spec!r}.')
    try:
      parsed_kwargs[kw.arg] = ast.literal_eval(kw.value)
    except (ValueError, TypeError, SyntaxError) as exc:
      raise ValueError(
          f'Non-literal value for keyword {kw.arg!r} in {spec!r}: {exc}'
      ) from exc
  return base_name, parsed_kwargs


class Registry(Generic[T]):
  """A lazy-importing registry validated by a runtime-checkable Protocol."""

  def __init__(self, name: str, protocol: type[T]) -> None:
    """Initializes a protocol-validated registry for `name`."""
    _validate_short_name(name)
    if not isinstance(protocol, type) or not getattr(
        protocol, '_is_protocol', False
    ):
      raise TypeError(
          f'Registry protocol for {name!r} must be a typing.Protocol, '
          f'got {protocol!r}.'
      )
    if not getattr(protocol, '_is_runtime_protocol', False):
      raise TypeError(
          f'Registry protocol {protocol!r} for {name!r} must be decorated '
          'with @typing.runtime_checkable.'
      )
    self._name = name
    self._protocol = protocol
    self._entries: dict[str, _Entry] = {}

  def register(self, name: str, class_path: str) -> None:
    """Registers `<module>.<Class>` (with optional preset literal kwargs)."""
    _validate_short_name(name)
    target_path, preset_kwargs = _parse_call_string(class_path)
    parts = target_path.split('.')
    if len(parts) < 2 or not all(part.isidentifier() for part in parts):
      raise ValueError(
          'class_path must be a dotted path of valid Python identifiers, '
          f'got {class_path!r}.'
      )
    entry = _Entry(target_path, preset_kwargs)
    existing = self._entries.get(name)
    if existing is not None:
      if existing == entry:
        return
      raise ValueError(
          f'Conflicting registration for {name!r} in {self._name!r}: '
          f'{existing!r} != {entry!r}.'
      )
    self._entries[name] = entry

  def unregister(self, name: str) -> None:
    """Removes a registered alias without importing its target module."""
    if self._entries.pop(name, None) is None:
      raise self._unknown_error(name)

  def get(self, name: str) -> type[T]:
    """Lazily imports and returns the registered class for `name`."""
    entry = self._entries.get(name)
    if entry is None:
      raise self._unknown_error(name)
    module_path, attr_name = entry.class_path.rsplit('.', 1)
    target = getattr(importlib.import_module(module_path), attr_name)
    if not isinstance(target, type):
      raise TypeError(
          f'Target {entry.class_path!r} for {name!r} is not a class: {target!r}'
      )
    return target

  def build(
      self,
      name: str,
      *args: Any,  # Untouched pass-through.
      **kwargs: Any,  # Untouched pass-through.
  ) -> T:
    """Resolves, instantiates, and protocol-validates an entry by `name`."""
    base_name, inline_kwargs = _parse_call_string(name)
    cls: Callable[..., object] = self.get(base_name)
    entry = self._entries[base_name]
    preset_kwargs = {k: copy.deepcopy(v) for k, v in entry.kwargs.items()}
    instance = cls(*args, **{**preset_kwargs, **inline_kwargs, **kwargs})
    if not isinstance(instance, self._protocol):
      raise TypeError(
          f'Instantiated object {instance!r} from {entry.class_path!r} '
          f'does not satisfy protocol {self._protocol.__name__}.'
      )
    return instance

  def entries(self) -> list[str]:
    """Returns sorted registered alias names without importing targets."""
    return sorted(self._entries)

  def bind(self, module: types.ModuleType) -> _ModuleGetAttr[T]:
    """Binds `register`, `unregister`, `get`, `build`, and `entries`."""
    setattr(module, '__getattr__', self._module_getattr)
    setattr(
        module, '__dir__', lambda: sorted({*module.__dict__, *_BOUND_METHODS})
    )
    return self._module_getattr

  def _unknown_error(self, name: str) -> KeyError:
    available = ', '.join(sorted(self._entries)) or '<empty>'
    return KeyError(
        f'Unknown {self._name} entry {name!r}. Available: [{available}]'
    )

  @overload
  def _module_getattr(self, name: Literal['register']) -> _RegisterFn:
    ...

  @overload
  def _module_getattr(self, name: Literal['unregister']) -> _UnregisterFn:
    ...

  @overload
  def _module_getattr(self, name: Literal['get']) -> _GetFn[T]:
    ...

  @overload
  def _module_getattr(self, name: Literal['build']) -> _BuildFn[T]:
    ...

  @overload
  def _module_getattr(self, name: Literal['entries']) -> _EntriesFn:
    ...

  def _module_getattr(self, name: str) -> Any:  # Runtime @overload dispatch.
    if name in _BOUND_METHODS:
      return getattr(self, name)
    raise AttributeError(
        f'Registry {self._name!r} has no module attribute {name!r}.'
    )
