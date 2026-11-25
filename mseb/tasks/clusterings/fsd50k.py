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

"""FSD50K sound event clustering tasks."""

import functools
from typing import Iterable

from mseb import types
from mseb.datasets import fsd50k
from mseb.evaluators import clustering_evaluator
from mseb.tasks import clustering

_DOCUMENTATION_FILE = (
    "https://github.com/google-research/mseb/blob/main/docs/fsd50k_clustering.md"
)
_DATASET_DOCUMENTATION_FILE = (
    "https://github.com/google-research/mseb/blob/main/docs/dataset_fsd50k.md"
)


_FSD50K_REFERENCE = """@article{fonseca2022fsd50k,
 author  = {Eduardo Fonseca and Xavier Favory and Jordi Pons and Frederic Font and Xavier Serra},
 title   = {{FSD50K}: an Open Dataset of Human-Labeled Sound Events},
 journal   = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
 volume  = {30},
 pages   = {829--852},
 year    = {2021},
 doi     = {10.1109/TASLP.2022.3149014}
}"""


def _build_metadata(
    *,
    name: str,
    description: str,
    revision: str,
    dataset_revision: str,
) -> types.TaskMetadata:
  """Builds TaskMetadata for an FSD50K sound event clustering task."""
  return types.TaskMetadata(
      name=name,
      description=description,
      reference=_FSD50K_REFERENCE,
      documentation_file=_DOCUMENTATION_FILE,
      dataset_documentation_file=_DATASET_DOCUMENTATION_FILE,
      type="Clustering",
      category="audio",
      main_score="VMeasure",
      revision=revision,
      dataset=types.Dataset(
          name="FSD50K",
          path="https://huggingface.co/datasets/Fhrozen/FSD50k",
          revision=dataset_revision,
      ),
      scores=[clustering_evaluator.vmeasure_score()],
      eval_splits=["test"],
      eval_langs=["und"],
      domains=["audio", "acoustic", "environmental"],
      task_subtypes=["clustering"],
  )


class FSD50KClustering(clustering.ClusteringTask):
  """Base class for sound event clustering on the FSD50K dataset."""

  split: str | None = None
  size: str | None = None
  commercial_only: bool = True

  @functools.cached_property
  def _fsd_dataset(self) -> fsd50k.FSD50KDataset:
    if self.split is None:
      raise ValueError("`split` must be set by a concrete task subclass.")
    filter_fn = None
    if self.size is not None:
      filter_fn = getattr(fsd50k, f"is_member_of_{self.size}")
    return fsd50k.FSD50KDataset(
        split=self.split,
        filter_fn=filter_fn,
        commercial_only=self.commercial_only,
    )

  @property
  def sub_tasks(self) -> list[str]:
    return ["sound_event"]

  def _get_label(self, record: dict[str, str]) -> str:
    """Gets the first label from the comma-separated labels string."""
    labels = record["labels"].split(",")
    return labels[0] if labels else "no_label"

  def multimodal_inputs(self) -> Iterable[types.Sound]:
    task_data = self._fsd_dataset.get_task_data()
    for record in task_data.to_dict("records"):
      yield self._fsd_dataset.get_sound(record)

  def examples(
      self, sub_task: str
  ) -> Iterable[clustering_evaluator.ClusteringExample]:
    task_data = self._fsd_dataset.get_task_data()
    for record in task_data.to_dict("records"):
      example_id = str(record["fname"])
      label = self._get_label(record)
      yield clustering_evaluator.ClusteringExample(example_id, label)


class FSD50KTestClustering(FSD50KClustering):
  split = "test"
  commercial_only = False
  metadata = _build_metadata(
      name="FSD50KTestClustering",
      description=(
          "Sound event clustering on the test split of the FSD50K dataset."
      ),
      revision="1.0.0",
      dataset_revision=fsd50k.FULL_VERSION,
  )


class FSD50KTestClusteringCommercial(FSD50KClustering):
  split = "test"
  commercial_only = True
  metadata = _build_metadata(
      name="FSD50KTestClusteringCommercial",
      description=(
          "Sound event clustering on the commercial-licensed test split of the"
          " FSD50K dataset."
      ),
      revision="2.0.0",
      dataset_revision=fsd50k.COMMERCIAL_VERSION,
  )


class FSD50KTestClusteringDebug(FSD50KClustering):
  split = "test"
  size = "debug"
  commercial_only = True
  metadata = _build_metadata(
      name="FSD50KTestClusteringDebug",
      description=(
          "Sound event clustering on the commercial-licensed debug test split"
          " of the FSD50K dataset."
      ),
      revision="2.0.0",
      dataset_revision=fsd50k.COMMERCIAL_VERSION,
  )
