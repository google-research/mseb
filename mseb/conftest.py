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

# pylint: skip-file
# Required so that absl flags get parsed before pytest tries to use them.

import sys
import absl.flags


def pytest_configure(config):
  # We must import absltest here to ensure its flags (like test_tmpdir)
  # are defined *before* we try to parse them.
  try:
    import absl.testing.absltest  # pylint: disable=g-import-not-at-top
  except ImportError:
    pass  # Handle case where absl-py isn't installed

  # This line tells absl.flags to parse the command line.
  # known_only=True ensures that it doesn't crash on
  # pytest-specific arguments (like -vv, -m, -n, etc.).
  absl.flags.FLAGS(sys.argv, known_only=True)
