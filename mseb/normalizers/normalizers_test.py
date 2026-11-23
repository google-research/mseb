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

"""Unit tests for versioned MSEB text normalizers."""

import importlib
import sys
import typing

from absl.testing import absltest
from absl.testing import parameterized
from mseb import normalizers


class NormalizersTest(parameterized.TestCase):

  def test_lazy_import_deferred_until_get(self) -> None:
    target_names = ('mseb.normalizers', 'mseb.normalizers.whisper')
    saved_modules = {
        name: sys.modules.pop(name)
        for name in target_names
        if name in sys.modules
    }

    def _restore_modules() -> None:
      for name in target_names:
        sys.modules.pop(name, None)
      sys.modules.update(saved_modules)
      if 'mseb' in sys.modules and 'mseb.normalizers' in saved_modules:
        setattr(
            sys.modules['mseb'],
            'normalizers',
            saved_modules['mseb.normalizers'],
        )

    self.addCleanup(_restore_modules)

    reloaded = importlib.import_module('mseb.normalizers')
    self.assertNotIn('mseb.normalizers.whisper', sys.modules)
    self.assertEqual(reloaded.entries(), ['Whisper.V1'])
    self.assertNotIn('mseb.normalizers.whisper', sys.modules)
    cls = reloaded.get('Whisper.V1')
    self.assertIn('mseb.normalizers.whisper', sys.modules)
    self.assertEqual(cls.__name__, 'WhisperNormalizer')

  def test_static_types_and_structural_conformance(self) -> None:
    cls = normalizers.get('Whisper.V1')
    typing.assert_type(cls, type[normalizers.Normalizer])
    typing.assert_type(
        normalizers.get(name='Whisper.V1'), type[normalizers.Normalizer]
    )
    typing.assert_type(normalizers.entries(), list[str])
    self.assertIn('entries', dir(normalizers))
    self.assertNotIn('entries', normalizers.__dict__)

    obj = normalizers.build('Whisper.V1', 'en-us')
    typing.assert_type(obj, normalizers.Normalizer)
    typing.assert_type(
        normalizers.build(name='Whisper.V1', locale='en-us'),
        normalizers.Normalizer,
    )

    self.assertEqual(cls.__bases__, (object,))
    self.assertNotIn(normalizers.Normalizer, cls.__mro__)
    self.assertIsInstance(obj, cls)
    self.assertIsInstance(obj, normalizers.Normalizer)
    self.assertEqual(obj.locale, 'en-us')
    self.assertEqual(normalizers.build('Whisper.V1').locale, 'en')

  @parameterized.named_parameters(
      ('en', 'en', 'hello world 21'),
      ('en_us', 'en_us', 'hello world 21'),
      ('en_dash_us', 'en-US', 'hello world 21'),
      ('en_gb', 'en-gb', 'hello world 21'),
      ('de_de', 'de_de', 'hello world twenty one '),
      ('fr_fr', 'fr-FR', 'hello world twenty one '),
      ('es_es', 'es_es', 'hello world twenty one '),
      ('cmn_hans_cn', 'cmn_hans_cn', 'hello world twenty one '),
      ('empty', '', 'hello world twenty one '),
  )
  def test_whisper_v1_locale_dispatch(self, locale: str, expected: str) -> None:
    norm = normalizers.build('Whisper.V1', locale)
    self.assertEqual(norm.locale, locale)
    self.assertEqual(norm('Hello, World! Twenty-one.'), expected)

  def test_inline_locale_kwarg(self) -> None:
    norm = normalizers.build("Whisper.V1(locale='fr-fr')")
    self.assertEqual(norm.locale, 'fr-fr')
    self.assertEqual(
        norm('Hello, World! Twenty-one.'), 'hello world twenty one '
    )


if __name__ == '__main__':
  absltest.main()
