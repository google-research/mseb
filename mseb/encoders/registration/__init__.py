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

"""Registration for Encoders.

Import this to automatically add encoders to the encoder registry.
"""

import importlib
import pkgutil
from absl import logging


def _auto_discover_modules() -> None:
  for _, name, ispkg in pkgutil.walk_packages(
      __path__,
      prefix=f"{__name__}.",
  ):
    if not ispkg:
      try:
        importlib.import_module(name)
      except ModuleNotFoundError as e:
        logging.warning(
            "Did not import encoder module '%s' due to: %r. You need to "
            "install the missing dependency.", name, e
        )


_auto_discover_modules()
