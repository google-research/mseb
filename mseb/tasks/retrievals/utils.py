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

"""Utility functions for retrieval tasks."""

from typing import Iterable, Mapping, Sequence
from mseb import types


class BackFillRetrievedItemTexts:
  """Backfills retrieved items with text from the documents."""

  def __init__(
      self,
      documents: Iterable[types.Text],
      text_by_id: Mapping[str, str | None],
      *,
      remove_scores: bool = True,
      top_k_retrieved_items: int = 10,
  ):
    self._text_by_id = {}
    if text_by_id:
      for document in documents:
        if (
            document.context.id in text_by_id
            and text_by_id[document.context.id] is None
        ):
          self._text_by_id[document.context.id] = document.text
    self._remove_scores = remove_scores
    self._top_k_retrieved_items = top_k_retrieved_items

  @staticmethod
  def get_empty_text_by_id(
      retrieved_items_str_list: Sequence[str | None],
  ) -> Mapping[str, str | None]:
    """Returns a mapping of document IDs to their texts."""
    text_by_id = {}
    for retrieved_items_str in retrieved_items_str_list:
      if retrieved_items_str:
        retrieved_items = types.ListPrediction.from_json(retrieved_items_str)
        if isinstance(retrieved_items, types.ValidListPrediction):
          for item in retrieved_items.items:
            if item.get('text') is None:
              text_by_id[item['id']] = None
    return text_by_id

  def backfill(self, retrieved_items_str: str | None) -> str | None:
    """Backfills retrieved items with text from the documents if not present.

    Args:
      retrieved_items_str: A JSON string representing a ListPrediction.

    Returns:
      A JSON string with the 'text' field populated for each item, or None if
      the input was None.
    """
    if not retrieved_items_str:
      return retrieved_items_str

    retrieved_items = types.ListPrediction.from_json(retrieved_items_str)
    if isinstance(retrieved_items, types.ValidListPrediction):
      retrieved_items.normalize(k=self._top_k_retrieved_items)
      items = retrieved_items.items
      for item in items:
        if 'text' not in item:
          item['text'] = self._text_by_id[item['id']]
        if self._remove_scores:
          item.pop('score', None)
    retrieved_items_str = retrieved_items.to_json()
    return retrieved_items_str
