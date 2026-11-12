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

"""MsgPack and Bagz Apache Beam sources and sinks for MSEB multimodal objects.

Example usage:

```python
import apache_beam as beam
from mseb import beam as mseb_beam
from mseb import types

items: list[types.MultiModalObject] = [...]

# Write MultiModalObjects to sharded Bagz files:
with beam.Pipeline() as pipeline:
  _ = (
      pipeline
      | beam.Create(items)
      | mseb_beam.WriteToBagz('/tmp/embeddings', num_shards=2)
  )

# Read MultiModalObjects back from sharded Bagz files:
with beam.Pipeline() as pipeline:
  objects = pipeline | mseb_beam.ReadFromBagz('/tmp/embeddings')
```
"""

import apache_beam as beam
from mseb import record
from mseb import types

from bagz.beam import bagzio


class ReadFromBagz(beam.PTransform):
  """Reads and deserializes MultiModalObjects from sharded Bagz files."""

  def __init__(self, file_pattern: str):
    if not file_pattern.endswith(('.bagz', '.bag')):
      file_pattern = f'{file_pattern}@*.bagz'
    self._file_pattern = file_pattern

  def expand(
      self, pbegin: beam.pvalue.PBegin
  ) -> beam.PCollection[types.MultiModalObject]:
    raw = pbegin | 'Read' >> bagzio.ReadFromBagz(self._file_pattern)
    return raw | 'Deserialize' >> beam.Map(record.decode)


class WriteToBagz(beam.PTransform):
  """Serializes MultiModalObjects and writes them to sharded Bagz files."""

  def __init__(self, prefix: str, num_shards: int = 1):
    self._prefix = prefix
    self._num_shards = max(1, num_shards)

  def expand(self, pcoll):
    encoded = (
        pcoll
        | 'Serialize' >> beam.Map(record.encode)
        | 'Reshuffle' >> beam.Reshuffle()  # Avoid fusing with write.
    )
    return encoded | 'Write' >> bagzio.WriteToBagz(
        f'{self._prefix}@{self._num_shards}.bagz'
    )
