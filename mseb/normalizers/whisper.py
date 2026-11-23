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

"""Whisper text normalizers satisfying the Normalizer protocol."""

from collections.abc import Callable
import functools

from whisper.normalizers import basic
from whisper.normalizers import english


def _is_english_locale(locale: str) -> bool:
  """Returns True if `locale` has primary language subtag `'en'`."""
  return locale.strip().replace('-', '_').split('_', 1)[0].lower() == 'en'


@functools.cache
def _get_english_normalizer() -> english.EnglishTextNormalizer:
  return english.EnglishTextNormalizer()


@functools.cache
def _get_basic_normalizer() -> basic.BasicTextNormalizer:
  return basic.BasicTextNormalizer()


class WhisperNormalizer:
  """Locale-dispatching Whisper text normalizer (`Whisper.V1`)."""

  def __init__(self, locale: str = 'en') -> None:
    """Initializes the locale-dispatching Whisper normalizer.

    English (`'en'`) locales dispatch to `EnglishTextNormalizer`, while
    non-English locales dispatch to `BasicTextNormalizer`.

    Args:
      locale: BCP-47 or underscore-delimited locale tag (e.g. `'en-us'`).
    """
    self._locale = locale
    if _is_english_locale(locale):
      self._normalizer: Callable[[str], str] = _get_english_normalizer()
    else:
      self._normalizer = _get_basic_normalizer()

  @property
  def locale(self) -> str:
    return self._locale

  def __call__(self, text: str) -> str:
    return self._normalizer(text)


# TBD: Implement WhisperAANormalizer ('Whisper_AA.V1') once Artificial
# Analysis normalization rules are finalized.
