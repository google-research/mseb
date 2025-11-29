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

"""FSD50K multi-label classification tasks."""

from typing import Iterable, Sequence

from mseb import types
from mseb.datasets import fsd50k
from mseb.evaluators import classification_evaluator
from mseb.tasks import classification


ReferenceType = classification_evaluator.MultiLabelClassificationReference


class FSD50KClassification(classification.ClassificationTask):
  """Base class for multi-label classification on the FSD50K dataset."""

  split: str | None = None

  def _get_dataset(self):
    return fsd50k.FSD50KDataset(split=self.split)

  @property
  def task_type(self) -> str:
    return "multi_label"

  @property
  def sub_tasks(self) -> list[str]:
    return ["classification"]

  def class_labels(self) -> Sequence[str]:
    # The class labels are the same regardless of the split.
    fsd_dataset = fsd50k.FSD50KDataset(split="test")
    return fsd_dataset.class_labels

  def sounds(self) -> Iterable[types.Sound]:
    fsd_dataset = self._get_dataset()
    for record in fsd_dataset.get_task_data().to_dict("records"):
      yield fsd_dataset.get_sound(record)

  def examples(self, sub_task: str) -> Iterable[ReferenceType]:
    if self.split is None:
      raise ValueError("`split` must be set by a concrete task subclass.")

    fsd_dataset = self._get_dataset()
    for _, record in enumerate(fsd_dataset.get_task_data().to_dict("records")):
      example_id = str(record["fname"])
      label_ids = record["labels"].split(",")
      yield classification_evaluator.MultiLabelClassificationReference(
          example_id=example_id,
          label_ids=label_ids,
      )


class FSD50KTestClassification(FSD50KClassification):
  split = "test"

  metadata = types.TaskMetadata(
      name="FSD50KTestClassification",
      description=(
          "Multi-label sound event classification on the test split of "
          "the FSD50K dataset."
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
      type="Classification",
      category="audio",
      main_score="mAP",
      revision="1.0.0",
      dataset=types.Dataset(
          path="https://huggingface.co/datasets/Fhrozen/FSD50k",
          revision="1.0.0",
      ),
      scores=[
          classification_evaluator.mean_average_precision(),
          classification_evaluator.micro_f1(),
          classification_evaluator.macro_f1(),
          classification_evaluator.hamming_loss(),
          classification_evaluator.subset_accuracy(),
      ],
      eval_splits=["test"],
      eval_langs=["und"],
      domains=["audio", "acoustic", "environmental"],
      task_subtypes=["classification"],
  )
