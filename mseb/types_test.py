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
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from mseb import types
import numpy as np
import numpy.testing as npt


class SoundContextParamsTest(parameterized.TestCase):

  def test_context_params_successful_instantiation(self):
    params = types.SoundContextParams(
        sample_rate=16000,
        length=88000,
        language="en",
        text="This is a test.",
        speaker_id="speaker_001",
        waveform_start_second=0.5,
        id="test",
    )
    self.assertEqual(params.sample_rate, 16000)
    self.assertEqual(params.length, 88000)
    self.assertEqual(params.language, "en")
    self.assertEqual(params.speaker_id, "speaker_001")
    self.assertIsNone(params.speaker_age)
    self.assertIsNone(params.speaker_gender)
    self.assertEqual(params.text, "This is a test.")
    self.assertEqual(params.waveform_start_second, 0.5)
    self.assertEqual(params.waveform_end_second, np.finfo(np.float32).max)

  def test_context_params_requires_sample_rate_and_length(self):
    with self.assertRaises(TypeError):
      types.SoundContextParams(text="This should fail.")

  def test_context_params_requires_sample_rate(self):
    with self.assertRaises(TypeError):
      types.SoundContextParams(
          length=80000,
          text="This should fail.",
          id="fail",
      )

  def test_context_params_requires_length(self):
    with self.assertRaises(TypeError):
      types.SoundContextParams(
          sample_rate=16000,
          text="This should fail.",
          id="fail",
      )

  def test_context_params_default_values(self):
    params = types.SoundContextParams(
        sample_rate=22050,
        length=80000,
        id="test",
    )
    assert params.sample_rate == 22050
    assert params.language is None
    assert params.text is None
    assert params.speaker_id is None
    assert params.waveform_start_second == 0.0
    assert params.waveform_end_second == np.finfo(np.float32).max

  def test_context_params_mutability(self):
    params = types.SoundContextParams(
        sample_rate=16000,
        length=80000,
        id="test",
    )
    assert params.text is None
    params.text = "New text value."
    assert params.text == "New text value."
    params.sample_rate = 48000
    assert params.sample_rate == 48000
    params.length = 84000
    assert params.length == 84000


class TextContextParamsTest(parameterized.TestCase):

  def test_context_params_successful_instantiation(self):
    params = types.TextContextParams(id="id", title="This is a title.")
    self.assertEqual(params.id, "id")
    self.assertEqual(params.title, "This is a title.")
    self.assertIsNone(params.context)

  def test_context_params_default_values(self):
    params = types.TextContextParams(id="id")
    self.assertEqual(params.id, "id")
    self.assertIsNone(params.title)
    self.assertIsNone(params.context)

  def test_context_params_mutability(self):
    params = types.TextContextParams(id="id", title="This is a title.")
    self.assertEqual(params.id, "id")
    self.assertEqual(params.title, "This is a title.")
    self.assertIsNone(params.context)
    params.context = "This is a context."
    self.assertEqual(params.title, "This is a title.")
    self.assertEqual(params.context, "This is a context.")


class TextTest(parameterized.TestCase):

  def test_text_successful_instantiation(self):
    text = types.Text(
        text="This is a test.",
        context=types.TextContextParams(id="text_id"),
    )
    self.assertEqual(text.text, "This is a test.")
    self.assertEqual(text.context.id, "text_id")
    self.assertIsNone(text.context.title)
    self.assertIsNone(text.context.context)


class TextEmbeddingTest(parameterized.TestCase):

  def test_text_embedding_successful_instantiation(self):
    embedding = types.TextEmbedding(
        embedding=np.array([1, 2, 3]),
        spans=np.array([[0, 1], [2, 3]]),
        context=types.TextContextParams(id="id"),
    )
    npt.assert_array_equal(embedding.embedding, np.array([1, 2, 3]))
    npt.assert_array_equal(embedding.spans, np.array([[0, 1], [2, 3]]))
    self.assertEqual(embedding.context.id, "id")


class TaskMetadataTest(parameterized.TestCase):

  def _get_valid_params(self) -> dict[str, Any]:
    return {
        "name": "MyTestTask",
        "description": "A test task for validation.",
        "reference": "https://example.com/reference",
        "dataset": types.Dataset(
            path="mteb/my-test-dataset",
            revision="1.0.0",
        ),
        "type": "Clustering",
        "category": "p2p",
        "eval_splits": ["test"],
        "eval_langs": ["eng-Latn"],
        "main_score": "v_measure",
        "revision": "v1.0.0",
        "scores": [
            types.Score(
                metric="v_measure",
                description="The V-Measure score.",
                value=0.5,
                min=0,
                max=1,
            )
        ],
        "domains": ["Legal", "Medical"],
        "task_subtypes": ["Political"],
    }

  def test_successful_instantiation(self):
    types.TaskMetadata(**self._get_valid_params())

  def test_is_frozen(self):
    metadata = types.TaskMetadata(**self._get_valid_params())
    with self.assertRaises(dataclasses.FrozenInstanceError):
      metadata.name = "A New Name"

  @parameterized.product(
      field=[
          "name",
          "description",
          "reference",
          "type",
          "category",
          "main_score",
          "revision"
      ],
      invalid_value=[None, ""],
  )
  def test_invalid_required_string_fields(self, field, invalid_value):
    params = self._get_valid_params()
    params[field] = invalid_value
    with self.assertRaisesRegex(
        TypeError,
        f"Metadata attribute '{field}' must be a non-empty string."
    ):
      types.TaskMetadata(**params)

  @parameterized.named_parameters(
      (
          "not_a_list",
          "eval_splits",
          "not-a-list",
          "must be a list"
      ),
      (
          "non_empty_list_fail",
          "eval_langs",
          [],
          "must be a non-empty list"
      ),
      (
          "non_string_item",
          "domains",
          ["Legal", None],
          "All items.*must be strings"
      ),
  )
  def test_invalid_list_fields(self, field, invalid_value, message):
    params = self._get_valid_params()
    params[field] = invalid_value
    with self.assertRaisesRegex(TypeError, message):
      types.TaskMetadata(**params)

  def test_dataset_instantiation_requires_path(self):
    with self.assertRaisesRegex(
        TypeError,
        "got an unexpected keyword argument 'wrong_key'"
    ):
      types.Dataset(wrong_key="some-path")  # type: ignore

  def test_main_score_must_be_in_scores_list(self):
    params = self._get_valid_params()
    params["main_score"] = "accuracy"
    with self.assertRaisesRegex(
        ValueError,
        "main_score 'accuracy' is not defined in the 'scores' list."
    ):
      types.TaskMetadata(**params)


class ListPredictionTest(parameterized.TestCase):

  def test_valid_list_prediction_init(self):
    prediction = types.ValidListPrediction([
        {"id": "bli", "score": 1.0},
        {"id": "bla", "score": 0.5},
        {"id": "blo", "score": 0.25},
    ])
    self.assertSequenceEqual(
        prediction.items,
        [
            {"id": "bli", "score": 1.0},
            {"id": "bla", "score": 0.5},
            {"id": "blo", "score": 0.25},
        ],
    )

  def test_invalid_answer_list_prediction_init(self):
    prediction = types.InvalidAnswerListPrediction()
    self.assertEqual(
        prediction.to_json(), types.ListPrediction.INVALID_ANSWER_STR
    )

  def test_no_response_list_prediction_init(self):
    prediction = types.NoResponseListPrediction()
    self.assertEqual(
        prediction.to_json(), types.ListPrediction.NO_RESPONSE_STR
    )

  def test_valid_list_prediction_to_json(self):
    prediction = types.ValidListPrediction([
        {"id": "bli", "score": 1.0},
        {"id": "bla", "score": 0.5},
        {"id": "blo", "score": 0.25},
    ])
    self.assertEqual(
        prediction.to_json(),
        '[{"id": "bli", "score": 1.0}, {"id": "bla", "score": 0.5}, {"id":'
        ' "blo", "score": 0.25}]',
    )

  def test_invalid_answer_list_prediction_to_json(self):
    prediction = types.InvalidAnswerListPrediction()
    self.assertEqual(
        prediction.to_json(),
        types.ListPrediction.INVALID_ANSWER_STR,
    )

  def test_no_response_list_prediction_to_json(self):
    prediction = types.NoResponseListPrediction()
    self.assertEqual(
        prediction.to_json(),
        types.ListPrediction.NO_RESPONSE_STR,
    )

  def test_valid_list_prediction_from_json(self):
    prediction = types.ValidListPrediction.from_json(
        '[{"id": "bli", "score": 1.0}, {"id": "bla", "score": 0.5}, {"id":'
        ' "blo", "score": 0.25}]',
    )
    self.assertEqual(
        prediction.items,
        [
            {"id": "bli", "score": 1.0},
            {"id": "bla", "score": 0.5},
            {"id": "blo", "score": 0.25},
        ],
    )

  def test_no_response_list_prediction_from_json(self):
    prediction = types.NoResponseListPrediction.from_json(
        types.ListPrediction.NO_RESPONSE_STR
    )
    self.assertEqual(
        prediction.to_json(), types.ListPrediction.NO_RESPONSE_STR
    )

  def test_invalid_answer_list_prediction_from_json(self):
    prediction = types.InvalidAnswerListPrediction.from_json(
        types.ListPrediction.INVALID_ANSWER_STR
    )
    self.assertEqual(
        prediction.to_json(), types.ListPrediction.INVALID_ANSWER_STR
    )

  def test_valid_list_prediction_normalize(self):
    prediction = types.ValidListPrediction([
        {"id": "blo", "score": 0.25},
        {"id": "bli", "score": 1.0},
        {"id": "bla", "score": 0.5},
    ])
    prediction.normalize(k=2)
    self.assertSequenceEqual(
        prediction.items,
        [{"id": "bli", "score": 1.0}, {"id": "bla", "score": 0.5}],
    )

  def test_valid_list_prediction_merge(self):
    prediction_1 = types.ValidListPrediction([
        {"id": "bli", "score": 1.0},
        {"id": "bla", "score": 0.5},
        {"id": "blo", "score": 0.25},
    ])
    prediction_2 = types.ValidListPrediction([
        {"id": "blu", "score": 1.0},
        {"id": "bla", "score": 0.5},
        {"id": "blo", "score": 0.25},
    ])
    prediction_1.merge(prediction_2, k=2)
    self.assertSequenceEqual(
        prediction_1.items,
        [{"id": "bli", "score": 1.0}, {"id": "blu", "score": 1.0}],
    )


if __name__ == "__main__":
  absltest.main()
