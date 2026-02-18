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

from typing import Iterable, Type

from mseb import runner as runner_lib
from mseb import types
from mseb.datasets import fsd50k
from mseb.evaluators import clustering_evaluator
from mseb.tasks import clustering


class FSD50KClustering(clustering.ClusteringTask):
  """Base class for sound event clustering on the FSD50K dataset."""

  split: str | None = None
  _fsd_dataset: fsd50k.FSD50KDataset

  def _get_dataset(self) -> fsd50k.FSD50KDataset:
    if self.split is None:
      raise ValueError("`split` must be set by a concrete task subclass.")
    return fsd50k.FSD50KDataset(split=self.split)

  def setup(
      self, runner_cls: Type[runner_lib.EncoderRunner] | None = None, **kwargs
  ):
    self._fsd_dataset = self._get_dataset()

  @property
  def sub_tasks(self) -> list[str]:
    return ["sound_event"]

  def _get_label(self, record: dict[str, str]) -> str:
    """Gets the first label from the comma-separated labels string."""
    labels = record["labels"].split(",")
    return labels[0] if labels else "no_label"

  def sounds(self) -> Iterable[types.Sound]:
    for record in self._fsd_dataset.get_task_data().to_dict("records"):
      yield self._fsd_dataset.get_sound(record)

  def examples(
      self, sub_task: str
  ) -> Iterable[clustering_evaluator.ClusteringExample]:
    if self.split is None:
      raise ValueError("`split` must be set by a concrete task subclass.")
    for record in self._fsd_dataset.get_task_data().to_dict("records"):
      example_id = str(record["fname"])
      label = self._get_label(record)
      yield clustering_evaluator.ClusteringExample(example_id, label)


class FSD50KTestClustering(FSD50KClustering):
  split = "test"
  metadata = types.TaskMetadata(
      name="FSD50KTestClustering",
      description=(
          "Sound event clustering on the test split of the FSD50K dataset."
      ),
      reference="""@article{fonseca2022fsd50k,
 author  = {Eduardo Fonseca and Xavier Favory and Jordi Pons and Frederic Font and Xavier Serra},
 title   = {{FSD50K}: an Open Dataset of Human-Labeled Sound Events},
 journal   = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
 volume  = {30},
 pages   = {829--852},
 year    = {2021},
 doi     = {10.1109/TASLP.2022.3149014}
}""",
      documentation_file="fsd50k_clustering.md",
      dataset_documentation_file="dataset_fsd50k.md",
      type="Clustering",
      category="audio",
      main_score="VMeasure",
      revision="1.0.0",
      dataset=types.Dataset(
          path="https://huggingface.co/datasets/Fhrozen/FSD50k",
          revision="1.0.0",
      ),
      scores=[clustering_evaluator.vmeasure_score()],
      eval_splits=["test"],
      eval_langs=["und"],
      domains=["audio", "acoustic", "environmental"],
      task_subtypes=["clustering"],
  )
