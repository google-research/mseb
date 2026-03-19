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

from absl import flags
from absl.testing import absltest
from mseb import types
from mseb.tasks.retrievals import utils

FLAGS = flags.FLAGS


class UtilsTest(absltest.TestCase):

  def test_get_empty_text_by_id(self):
    retrieved_items_str_list = [
        types.ValidListPrediction(
            items=[
                {"id": "doc_1"},
                {"id": "doc_2", "text": "my_text"},
            ]
        ).to_json(),
        types.ValidListPrediction(
            items=[
                {"id": "doc_3"},
                {"id": "doc_4", "text": "my_text"},
            ]
        ).to_json(),
    ]
    self.assertEqual(
        utils.BackFillRetrievedItemTexts.get_empty_text_by_id(
            retrieved_items_str_list
        ),
        {
            "doc_1": None,
            "doc_3": None,
        },
    )

  def test_backfill_retrieved_item_texts(self):
    documents = [
        types.Text(
            context=types.TextContextParams(id="doc_1"),
            text="text_1",
        ),
        types.Text(
            context=types.TextContextParams(id="doc_2"),
            text="text_2",
        ),
    ]
    backfill = utils.BackFillRetrievedItemTexts(
        documents, text_by_id={"doc_1": None}
    )
    retrieved_items_str = types.ValidListPrediction(
        items=[
            {"id": "doc_1"},
            {"id": "doc_2", "text": "my_text"},
        ]
    ).to_json()
    self.assertEqual(
        backfill.backfill(retrieved_items_str),
        types.ValidListPrediction(
            items=[
                {"id": "doc_1", "text": "text_1"},
                {"id": "doc_2", "text": "my_text"},
            ]
        ).to_json(),
    )

  def test_backfill_retrieved_item_texts_no_response(self):
    documents = [
        types.Text(
            context=types.TextContextParams(id="doc_1"),
            text="text_1",
        ),
        types.Text(
            context=types.TextContextParams(id="doc_2"),
            text="text_2",
        ),
    ]
    backfill = utils.BackFillRetrievedItemTexts(
        documents, text_by_id={"doc_1": None, "doc_2": None}
    )
    self.assertEqual(
        backfill.backfill(types.NoResponseListPrediction().to_json()),
        types.NoResponseListPrediction().to_json(),
    )


if __name__ == "__main__":
  absltest.main()
