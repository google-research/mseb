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

import os
from absl import flags
from absl.testing import absltest
from mseb import types
from mseb.tasks.retrievals import utils
import pyarrow as pa
import pyarrow.parquet as pq

FLAGS = flags.FLAGS


class UtilsTest(absltest.TestCase):

  def test_document_index_path(self):
    self.assertEqual(
        utils.document_index_path("/tmp/svq", "en"),
        "/tmp/svq/document_retrieval_cross_lang_en_index-?????-of-?????.parquet",
    )
    with self.assertRaisesRegex(
        ValueError, "Dataset base path is required for the document index"
    ):
      utils.document_index_path(None, "en")

  def test_documents_from_parquet(self):
    base_dir = self.create_tempdir().full_path
    pq.write_table(
        pa.table({"title": ["Doc 3"], "text": ["Text 3"]}),
        os.path.join(
            base_dir,
            "document_retrieval_cross_lang_en_index-00001-of-00002.parquet",
        ),
    )
    pq.write_table(
        pa.table({"title": ["Doc 1", "Doc 2"], "text": ["Text 1", "Text 2"]}),
        os.path.join(
            base_dir,
            "document_retrieval_cross_lang_en_index-00000-of-00002.parquet",
        ),
    )
    pattern = utils.document_index_path(base_dir, "en")
    documents = list(utils.documents_from_parquet(pattern, batch_size=1))
    self.assertEqual(
        [(d.context.id, d.context.title, d.text) for d in documents],
        [
            ("Doc 1", "Doc 1", "Text 1"),
            ("Doc 2", "Doc 2", "Text 2"),
            ("Doc 3", "Doc 3", "Text 3"),
        ],
    )

  def test_documents_from_parquet_missing_files(self):
    base_dir = self.create_tempdir().full_path
    pattern = utils.document_index_path(base_dir, "en")
    with self.assertRaisesRegex(
        FileNotFoundError, "No document index files match"
    ):
      list(utils.documents_from_parquet(pattern))

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

  def test_naive_token_count(self):
    token_count = utils.get_token_count("naive")
    self.assertIsInstance(token_count, utils.NaiveTokenCount)
    self.assertEqual(token_count("Hello world!"), 2)

  def test_list_prediction_truncation(self):
    truncation = utils.ListPredictionTruncation(
        max_tokens=8,
        tokenizer_name="naive",
    )

    self.assertEqual(
        truncation.maybe_truncate(
            types.ValidListPrediction(
                items=[
                    {"id": "doc_1", "text": "text_1"},
                    {"id": "doc_2", "text": "text_2"},
                ]
            ).to_json()
        ),
        types.ValidListPrediction(
            items=[
                {"id": "doc_1", "text": "text_1"},
                {"id": "doc_2", "text": "text_2"},
            ]
        ).to_json(),
    )

    self.assertEqual(
        truncation.maybe_truncate(
            types.ValidListPrediction(
                items=[
                    {"id": "doc_1", "text": "text 1"},
                    {"id": "doc_2", "text": "text 2"},
                ]
            ).to_json()
        ),
        types.ValidListPrediction(
            items=[
                {"id": "doc_1", "text": "text 1"},
            ]
        ).to_json(),
    )


if __name__ == "__main__":
  absltest.main()
