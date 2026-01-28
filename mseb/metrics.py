# Copyright 2025 The MSEB Authors.
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

"""Common metrics for MSEB."""

from collections.abc import Sequence


def compute_reciprocal_rank(
    reference: str, predicted_neighbors: Sequence[str]
) -> float:
  """Computes the reciprocal rank for mean reciprocal rank (MRR)."""
  rank = 0
  for i, neighbor in enumerate(predicted_neighbors):
    if reference == neighbor:
      # Matched the reference at this position in the ranking.
      rank = i + 1
      break
  reciprocal_rank = 1 / rank if rank > 0 else 0
  return reciprocal_rank


def compute_exact_match(
    reference: str, predicted_neighbors: Sequence[str]
) -> float:
  """Computes the exact match for the first predicted neighbor."""
  if predicted_neighbors:
    return float(reference == predicted_neighbors[0])
  return 0.0
