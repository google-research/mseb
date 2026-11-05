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

"""Bagz and MsgPack serialization for MSEB multimodal objects."""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

from absl import logging
import bagz
from etils import epath
from mseb import types
import msgpack
import numpy as np
import tqdm

_SERIALIZABLE_TYPES = {
    cls.__name__: cls
    for cls in (
        types.EncodingStats,
        types.TextContextParams,
        types.SoundContextParams,
        types.ImageContextParams,
        types.PredictionContextParams,
        types.Text,
        types.TextWithTitleAndContext,
        types.Sound,
        types.SoundWithTitleAndContext,
        types.Image,
        types.SoundEmbedding,
        types.SoundEmbeddingWithTitleAndContext,
        types.TextEmbedding,
        types.ImageEmbedding,
        types.TextPrediction,
        types.SoundEmbeddingCollection,
        types.StreamingSpeechToTextEvent,
        types.StreamingStats,
    )
}


def _msgpack_default(obj: Any) -> Any:
  """Encodes NumPy arrays, scalars, and MSEB dataclasses for msgpack."""
  if isinstance(obj, np.ndarray):
    if obj.dtype.kind == 'O':
      return {'__ndarray_obj__': obj.tolist(), 'shape': obj.shape}
    return {
        '__ndarray__': obj.tobytes(),
        'dtype': obj.dtype.str,
        'shape': obj.shape,
    }
  if isinstance(obj, np.generic):
    return obj.item()
  cls = type(obj)
  if _SERIALIZABLE_TYPES.get(cls.__name__) is cls:
    return {
        '__type__': cls.__name__,
        'data': {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)},
    }
  raise TypeError(f'Unsupported type for msgpack serialization: {cls}')


def _msgpack_object_hook(d: dict[str, Any]) -> Any:
  """Decodes NumPy arrays and MSEB dataclasses from msgpack."""
  if '__ndarray__' in d:
    return (
        np.frombuffer(d['__ndarray__'], dtype=np.dtype(d['dtype']))
        .reshape(d['shape'])
        .copy()
    )
  if '__ndarray_obj__' in d:
    return np.array(d['__ndarray_obj__'], dtype=object).reshape(d['shape'])
  if '__type__' in d:
    cls = _SERIALIZABLE_TYPES.get(d['__type__'])
    if cls is None:
      raise ValueError(f"Unknown serialized type: {d['__type__']}")
    return cls(**d['data'])
  return d


def encode(obj: types.MultiModalObject) -> bytes:
  """Serializes an MSEB MultiModalObject to msgpack bytes."""
  return msgpack.packb(obj, default=_msgpack_default, use_bin_type=True)


def decode(data: bytes) -> types.MultiModalObject:
  """Deserializes an MSEB MultiModalObject from msgpack bytes."""
  return msgpack.unpackb(data, object_hook=_msgpack_object_hook, raw=False)


def etch(
    prefix: str,
    objects: (
        Mapping[str, types.MultiModalObject] | Sequence[types.MultiModalObject]
    ),
    num_shards: int = 1,
) -> None:
  """Serializes MSEB objects into sharded Bagz files on disk."""
  values = tuple(objects.values()) if isinstance(objects, Mapping) else objects
  logging.info(
      'Etching %d records to %s@%d.bagz',
      len(values),
      prefix,
      num_shards,
  )
  epath.Path(prefix).parent.mkdir(parents=True, exist_ok=True)
  for shard_id in range(num_shards):
    with bagz.Writer(
        f'{prefix}-{shard_id:05d}-of-{num_shards:05d}.bagz'
    ) as writer:
      for obj in values[shard_id::num_shards]:
        writer.write(encode(obj))


def play(prefix: str) -> dict[str, types.MultiModalObject]:
  """Reads MSEB objects written by `etch` into a dict keyed by context id."""
  shard_spec = f'{prefix}@*.bagz'
  logging.info('Playing records from %s', shard_spec)
  reader = bagz.Reader(shard_spec)
  objects = {}
  for raw in tqdm.tqdm(reader, total=len(reader), desc='Loading records'):
    obj = decode(raw)
    objects[obj.context.id] = obj
  logging.info('Loaded %d records from %s', len(objects), shard_spec)
  if not objects:
    raise FileNotFoundError(f'Bagz files {shard_spec} contained zero records')
  return objects
