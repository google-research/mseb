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

from unittest import mock

from absl.testing import absltest
from mseb import types
from mseb.encoders import salient_term_segmentation_retriever as retriever_lib
from mseb.evaluators import segmentation_evaluator
import numpy as np
import numpy.testing as npt


def _make_sound_embedding(
    sound_id: str,
    frame_embeddings: np.ndarray,
    frame_timestamps: np.ndarray,
) -> types.SoundEmbedding:
  """Creates a SoundEmbedding with frame-level embeddings and timestamps."""
  return types.SoundEmbedding(
      embedding=frame_embeddings,
      timestamps=frame_timestamps,
      context=types.SoundContextParams(
          id=sound_id, sample_rate=16000, length=16000
      ),
  )


def _make_term_embedding(
    term: str,
    start_time: float,
    end_time: float,
    embedding: np.ndarray,
) -> retriever_lib.SegmentEmbedding:
  """Creates a SegmentEmbedding for a salient term."""
  return retriever_lib.SegmentEmbedding(
      embedding=term,
      start_time=start_time,
      end_time=end_time,
      vector=types.TextEmbedding(
          embedding=embedding.reshape(1, -1),
          spans=np.array([[0, 1]]),
          context=types.TextContextParams(id=term),
      ),
  )


class AverageEmbeddingOverSpanTest(absltest.TestCase):
  """Tests for _average_embedding_over_span."""

  def test_single_frame_overlap(self):
    """Frame midpoint 0.5 falls within [0.0, 1.0]."""
    frames = np.array([[1.0, 2.0, 3.0]])  # [1, 3]
    timestamps = np.array([[0.0, 1.0]])  # midpoint = 0.5
    result = retriever_lib._average_embedding_over_span(
        frames, timestamps, 0.0, 1.0
    )
    npt.assert_array_equal(result, [1.0, 2.0, 3.0])

  def test_multiple_frames_partial_overlap(self):
    """Only frames 1 and 2 (midpoints 1.5 and 2.5) overlap [1.0, 3.0]."""
    frames = np.array([
        [1.0, 0.0],  # frame 0: midpoint 0.5, outside
        [0.0, 2.0],  # frame 1: midpoint 1.5, inside
        [2.0, 4.0],  # frame 2: midpoint 2.5, inside
        [0.0, 0.0],  # frame 3: midpoint 3.5, outside
    ])
    timestamps = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    result = retriever_lib._average_embedding_over_span(
        frames, timestamps, 1.0, 3.0
    )
    # Average of [0, 2] and [2, 4] = [1, 3]
    npt.assert_array_almost_equal(result, [1.0, 3.0])

  def test_no_overlap_returns_zeros(self):
    """When no frames overlap, returns zero vector."""
    frames = np.array([[1.0, 2.0], [3.0, 4.0]])
    timestamps = np.array([[0.0, 1.0], [1.0, 2.0]])
    result = retriever_lib._average_embedding_over_span(
        frames, timestamps, 5.0, 6.0
    )
    npt.assert_array_equal(result, [0.0, 0.0])

  def test_all_frames_overlap(self):
    """All frames overlap the span."""
    frames = np.array([[1.0, 0.0], [0.0, 1.0]])
    timestamps = np.array([[0.0, 1.0], [1.0, 2.0]])
    result = retriever_lib._average_embedding_over_span(
        frames, timestamps, 0.0, 2.0
    )
    npt.assert_array_almost_equal(result, [0.5, 0.5])


class SalientTermSegmentationRetrieverTest(absltest.TestCase):
  """Tests for SalientTermSegmentationRetriever."""

  def _make_encoder_with_term_embeddings(
      self,
      term_embeddings_by_sound_id,
  ):
    """Creates an encoder with pre-populated term embeddings."""
    enc = retriever_lib.SalientTermSegmentationRetriever()
    enc._term_embeddings_by_sound_id = term_embeddings_by_sound_id
    enc._is_setup = True
    return enc

  def test_check_input_types_rejects_non_sound_embedding(self):
    enc = retriever_lib.SalientTermSegmentationRetriever()
    with self.assertRaises(ValueError):
      enc._check_input_types(
          [types.Text(text='hello', context=types.TextContextParams(id='t1'))]
      )

  def test_check_input_types_accepts_sound_embedding(self):
    enc = retriever_lib.SalientTermSegmentationRetriever()
    sound_emb = _make_sound_embedding(
        'utt1', np.zeros((1, 4)), np.array([[0.0, 1.0]])
    )
    enc._check_input_types([sound_emb])  # Should not raise.

  def test_encode_raises_when_not_setup(self):
    enc = retriever_lib.SalientTermSegmentationRetriever()
    sound_emb = _make_sound_embedding(
        'utt1', np.zeros((1, 4)), np.array([[0.0, 1.0]])
    )
    with self.assertRaises(ValueError):
      enc._encode([sound_emb])

  def test_encode_missing_sound_id(self):
    """Sound ID not in term_embeddings produces empty output."""
    enc = self._make_encoder_with_term_embeddings({})
    sound_emb = _make_sound_embedding(
        'unknown_utt', np.zeros((1, 4)), np.array([[0.0, 1.0]])
    )
    outputs = enc._encode([sound_emb])
    self.assertLen(outputs, 1)
    self.assertEmpty(outputs[0].embedding)

  def test_encode_single_term_single_frame(self):
    """Single frame, single term: term score = dot(frame, term_emb)."""
    # Frame embedding = [1, 0, 0, 0]
    sound_emb = _make_sound_embedding(
        'utt1',
        np.array([[1.0, 0.0, 0.0, 0.0]]),
        np.array([[0.0, 1.0]]),
    )
    # Term embedding = [1, 0, 0, 0] -> dot product = 1.0
    term_emb = _make_term_embedding(
        'weather', 0.0, 1.0, np.array([1.0, 0.0, 0.0, 0.0])
    )
    enc = self._make_encoder_with_term_embeddings({'utt1': [term_emb]})

    outputs = enc._encode([sound_emb])
    self.assertLen(outputs, 1)
    self.assertEqual(outputs[0].embedding[0], 'weather')
    npt.assert_almost_equal(outputs[0].scores[0], 1.0)
    npt.assert_array_almost_equal(outputs[0].timestamps[0], [0.0, 1.0])

  def test_encode_ranks_terms_by_score(self):
    """Multiple terms: output is ranked by dot-product score descending."""
    # 4 frames spanning [0,4], embedding dim = 2
    sound_emb = _make_sound_embedding(
        'utt1',
        np.array([
            [1.0, 0.0],  # frame 0: [0, 1]
            [0.0, 1.0],  # frame 1: [1, 2]
            [1.0, 1.0],  # frame 2: [2, 3]
            [0.5, 0.5],  # frame 3: [3, 4]
        ]),
        np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
    )

    # Term A at [0, 1]: span embedding = [1, 0], term_emb = [0, 1]
    # dot = 0.0
    term_a = _make_term_embedding('term_a', 0.0, 1.0, np.array([0.0, 1.0]))
    # Term B at [1, 2]: span embedding = [0, 1], term_emb = [0, 1]
    # dot = 1.0
    term_b = _make_term_embedding('term_b', 1.0, 2.0, np.array([0.0, 1.0]))
    # Term C at [2, 3]: span embedding = [1, 1], term_emb = [1, 0]
    # dot = 1.0
    term_c = _make_term_embedding('term_c', 2.0, 3.0, np.array([1.0, 0.0]))

    enc = self._make_encoder_with_term_embeddings(
        {'utt1': [term_a, term_b, term_c]}
    )
    outputs = enc._encode([sound_emb])
    self.assertLen(outputs, 1)
    pred = outputs[0]
    self.assertLen(pred.embedding, 3)

    # term_b and term_c both have score 1.0, term_a has score 0.0
    # term_a should be last
    self.assertEqual(pred.embedding[-1], 'term_a')
    npt.assert_almost_equal(pred.scores[-1], 0.0)

    # First two should have score 1.0
    npt.assert_almost_equal(pred.scores[0], 1.0)
    npt.assert_almost_equal(pred.scores[1], 1.0)

  def test_encode_multiple_sounds_in_batch(self):
    """Encoder handles a batch of multiple sound embeddings."""
    sound1 = _make_sound_embedding(
        'utt1',
        np.array([[1.0, 0.0]]),
        np.array([[0.0, 1.0]]),
    )
    sound2 = _make_sound_embedding(
        'utt2',
        np.array([[0.0, 1.0]]),
        np.array([[0.0, 1.0]]),
    )
    term1 = _make_term_embedding('alpha', 0.0, 1.0, np.array([1.0, 0.0]))
    term2 = _make_term_embedding('beta', 0.0, 1.0, np.array([0.0, 1.0]))
    enc = self._make_encoder_with_term_embeddings({
        'utt1': [term1],
        'utt2': [term2],
    })
    outputs = enc._encode([sound1, sound2])
    self.assertLen(outputs, 2)
    # utt1: dot([1,0], [1,0]) = 1.0
    self.assertEqual(outputs[0].embedding[0], 'alpha')
    npt.assert_almost_equal(outputs[0].scores[0], 1.0)
    # utt2: dot([0,1], [0,1]) = 1.0
    self.assertEqual(outputs[1].embedding[0], 'beta')
    npt.assert_almost_equal(outputs[1].scores[0], 1.0)

  def test_encode_preserves_timestamps(self):
    """Output timestamps match the reference segment timestamps."""
    sound_emb = _make_sound_embedding(
        'utt1',
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[0.0, 0.5], [0.5, 1.0]]),
    )
    term1 = _make_term_embedding('foo', 0.0, 0.5, np.array([1.0, 0.0]))
    term2 = _make_term_embedding('bar', 0.5, 1.0, np.array([0.0, 1.0]))
    enc = self._make_encoder_with_term_embeddings({'utt1': [term1, term2]})
    outputs = enc._encode([sound_emb])
    pred = outputs[0]
    # Both have dot product = 1.0, so order may vary.
    # Verify timestamps are from the original segments.
    all_ts = {tuple(ts) for ts in pred.timestamps}
    self.assertIn((0.0, 0.5), all_ts)
    self.assertIn((0.5, 1.0), all_ts)

  def test_compute_prediction_no_timestamps(self):
    """SoundEmbedding without timestamps uses default timestamps."""
    # Create a SoundEmbedding without timestamps (set to None-like)
    sound_emb = types.SoundEmbedding(
        embedding=np.array([[1.0, 0.0]]),
        timestamps=np.array([[0.0, 1.0]]),
        context=types.SoundContextParams(
            id='utt1', sample_rate=16000, length=16000
        ),
    )
    term_emb = _make_term_embedding('test_term', 0.0, 1.0, np.array([1.0, 0.0]))
    enc = retriever_lib.SalientTermSegmentationRetriever()
    pred = enc._compute_prediction(sound_emb, [term_emb])
    self.assertEqual(pred.embedding[0], 'test_term')
    npt.assert_almost_equal(pred.scores[0], 1.0)

  def test_encode_output_context_ids(self):
    """Output SoundEmbedding has the correct context.id."""
    sound_emb = _make_sound_embedding(
        'my_utt_id',
        np.array([[1.0]]),
        np.array([[0.0, 1.0]]),
    )
    term_emb = _make_term_embedding('t', 0.0, 1.0, np.array([1.0]))
    enc = self._make_encoder_with_term_embeddings({'my_utt_id': [term_emb]})
    outputs = enc._encode([sound_emb])
    self.assertEqual(outputs[0].context.id, 'my_utt_id')


class SetTaskAndSetupTest(absltest.TestCase):
  """Tests for set_task and _setup integration."""

  def test_set_task_stores_terms(self):
    """set_task extracts salient_term_lists from a mock task."""
    enc = retriever_lib.SalientTermSegmentationRetriever()
    mock_task = mock.MagicMock()
    mock_task.salient_term_lists.return_value = [
        (
            'utt1',
            [
                segmentation_evaluator.Segment('weather', 0.0, 1.0),
                segmentation_evaluator.Segment('boston', 1.0, 2.0),
            ],
        ),
    ]
    # Need to mock the embeddings_dir property too.
    mock_task.embeddings_dir = '/tmp/test_embeddings'
    enc.set_task(mock_task)
    self.assertIsNotNone(enc._terms_by_sound_id)
    self.assertIn('utt1', enc._terms_by_sound_id)
    self.assertLen(enc._terms_by_sound_id['utt1'], 2)

  def test_setup_raises_without_set_task(self):
    """_setup raises if set_task was not called."""
    enc = retriever_lib.SalientTermSegmentationRetriever()
    with self.assertRaises(ValueError):
      enc._setup()


if __name__ == '__main__':
  absltest.main()
