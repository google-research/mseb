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

import dataclasses
from os import path

from absl.testing import absltest
import apache_beam as beam
import bagz
from mseb import record
from mseb import types
import numpy as np


class RecordTest(absltest.TestCase):

  def test_encode_and_decode_roundtrip(self):
    objects = [
        types.TextEmbedding(
            embedding=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            spans=np.array([[0, 5]], dtype=np.int32),
            context=types.TextContextParams(id='text_emb_1'),
            encoding_stats=types.EncodingStats(
                input_size_bytes=100, embedding_size_bytes=16
            ),
        ),
        types.TextWithTitleAndContext(
            text='hello world',
            title_text='Title',
            context_text='Context',
            context=types.TextContextParams(id='text_ctx_1'),
        ),
        types.Sound(
            waveform=np.array([0.1, -0.2, 0.3], dtype=np.float32),
            context=types.SoundContextParams(
                id='sound_1', sample_rate=16000, length=3
            ),
        ),
        types.SoundEmbedding(
            embedding=np.array(['transcript'], dtype=object),
            timestamps=np.array([[0.0, 1.0]], dtype=np.float32),
            context=types.SoundContextParams(
                id='asr_1', sample_rate=16000, length=16000
            ),
        ),
        types.SoundEmbeddingCollection(
            embeddings={
                'head': types.TextPrediction(
                    prediction='ok',
                    context=types.PredictionContextParams(id='bundle_1'),
                )
            },
            context=types.SoundContextParams(
                id='bundle_1', sample_rate=16000, length=3
            ),
        ),
    ]
    for obj in objects:
      decoded = record.decode(record.encode(obj))
      self.assertIs(type(decoded), type(obj))
      np.testing.assert_equal(
          dataclasses.asdict(decoded), dataclasses.asdict(obj)
      )

  def test_etch_and_play(self):
    basedir = self.create_tempdir().full_path
    embeddings = {
        'utt_1': types.TextEmbedding(
            embedding=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            spans=np.array([[0, -1]], dtype=np.int32),
            context=types.TextContextParams(id='utt_1'),
        ),
        'utt_2': types.TextEmbedding(
            embedding=np.array([[4.0, 5.0, 6.0]], dtype=np.float32),
            spans=np.array([[0, -1]], dtype=np.int32),
            context=types.TextContextParams(id='utt_2'),
        ),
        'utt_3': types.TextEmbedding(
            embedding=np.array([[7.0, 8.0, 9.0]], dtype=np.float32),
            spans=np.array([[0, -1]], dtype=np.int32),
            context=types.TextContextParams(id='utt_3'),
        ),
    }
    prefix = path.join(basedir, 'embeddings')
    record.etch(prefix, embeddings, num_shards=2)
    for shard, expected_ids in (
        ('00000-of-00002', ['utt_1', 'utt_3']),
        ('00001-of-00002', ['utt_2']),
    ):
      shard_ids = [
          record.decode(raw).context.id
          for raw in bagz.Reader(f'{prefix}-{shard}.bagz')
      ]
      self.assertEqual(shard_ids, expected_ids)
    loaded = record.play(prefix)

    self.assertCountEqual(loaded.keys(), embeddings.keys())
    for key, expected in embeddings.items():
      np.testing.assert_array_equal(loaded[key].embedding, expected.embedding)
      np.testing.assert_array_equal(loaded[key].spans, expected.spans)
      self.assertEqual(loaded[key].context, expected.context)

  def test_play_ignores_files_sharing_a_longer_prefix(self):
    basedir = self.create_tempdir().full_path
    text = types.Text(text='hello', context=types.TextContextParams(id='a'))
    other = types.Text(text='bye', context=types.TextContextParams(id='b'))
    record.etch(path.join(basedir, 'emb'), [text])
    record.etch(path.join(basedir, 'embeddings'), [other])
    self.assertCountEqual(record.play(path.join(basedir, 'emb')).keys(), ['a'])

  def test_play_missing_raises(self):
    basedir = self.create_tempdir().full_path
    with self.assertRaisesRegex(
        (FileNotFoundError, ValueError),
        'Failed to detect number of shards|No such file or directory',
    ):
      record.play(path.join(basedir, 'nonexistent'))

  def test_play_empty_raises(self):
    basedir = self.create_tempdir().full_path
    prefix = path.join(basedir, 'empty')
    record.etch(prefix, [], num_shards=1)
    with self.assertRaisesRegex(FileNotFoundError, 'contained zero records'):
      record.play(prefix)

  def test_write_to_bagz(self):
    basedir = self.create_tempdir().full_path
    prefix = path.join(basedir, 'beam_embeddings')
    items = [
        types.TextEmbedding(
            embedding=np.array([[1.0, 2.0]], dtype=np.float32),
            spans=np.array([[0, 5]], dtype=np.int32),
            context=types.TextContextParams(id='item_1'),
        ),
        types.TextEmbedding(
            embedding=np.array([[3.0, 4.0]], dtype=np.float32),
            spans=np.array([[0, 5]], dtype=np.int32),
            context=types.TextContextParams(id='item_2'),
        ),
    ]
    with beam.Pipeline(
        runner=beam.runners.direct.direct_runner.DirectRunner()
    ) as pipeline:
      _ = (
          pipeline
          | beam.Create(items)
          | record.WriteToBagz(prefix, num_shards=2)
      )
    loaded = record.play(prefix)
    self.assertCountEqual(loaded.keys(), ['item_1', 'item_2'])
    np.testing.assert_array_equal(
        loaded['item_1'].embedding, items[0].embedding
    )
    np.testing.assert_array_equal(
        loaded['item_2'].embedding, items[1].embedding
    )


if __name__ == '__main__':
  absltest.main()
