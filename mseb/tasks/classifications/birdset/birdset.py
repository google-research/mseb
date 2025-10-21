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

"""Birdset classification tasks."""

import os
from typing import Iterable

from mseb import types
from mseb.datasets import birdset
from mseb.evaluators import classification_evaluator
from mseb.tasks import classification


def _get_birdset_metadata(configuration: str) -> types.TaskMetadata:
  """Returns TaskMetadata for a Birdset classification task."""
  return types.TaskMetadata(
      name=f"Birdset{configuration}Classification",
      description=f"Ebird classification task on Birdset {configuration}.",
      reference="https://arxiv.org/abs/2403.10380",
      type="Classification",
      category="audio",
      main_score="mAP",
      revision="1.0.0",
      dataset=types.Dataset(
          path="https://huggingface.co/datasets/DBD-research-group/BirdSet",
          revision="1.0.0",
      ),
      scores=[
          classification_evaluator.mean_average_precision(),
          classification_evaluator.micro_f1(),
          classification_evaluator.macro_f1(),
          classification_evaluator.hamming_loss(),
          classification_evaluator.subset_accuracy(),
          classification_evaluator.balanced_accuracy(),
      ],
      eval_splits=["test"],
      eval_langs=["und"],
      domains=["bioacoustics"],
      task_subtypes=["classification"],
  )


class BirdsetClassification(classification.ClassificationTask):
  """Birdset classification task."""

  configuration: str = "HSN"  # Default configuration

  @property
  def task_type(self) -> str:
    return "multi_label"

  @property
  def weights_dir(self) -> str:
    return os.path.join(
        super().weights_dir,
        f"birdset_{self.configuration}_classification",
    )

  @property
  def sub_tasks(self) -> list[str]:
    return ["ebird_classification"]

  def _get_dataset(self) -> birdset.BirdsetDataset:
    return birdset.BirdsetDataset(
        split="test_5s", configuration=self.configuration
    )

  def sounds(self) -> Iterable[types.Sound]:
    dataset = self._get_dataset()
    for _, example in dataset.get_task_data().iterrows():
      yield dataset.get_sound(example)

  def examples(
      self, sub_task: str
  ) -> Iterable[classification_evaluator.MultiLabelClassificationReference]:
    dataset = self._get_dataset()
    for _, example in dataset.get_task_data().iterrows():
      yield classification_evaluator.MultiLabelClassificationReference(
          example_id=str(example["filepath"]),
          label_ids=list(example["ebird_code_multilabel"]),
      )

  def class_labels(self) -> Iterable[str]:
    dataset = self._get_dataset()
    unique_labels = set()
    for labels in dataset.get_task_data()["ebird_code_multilabel"]:
      for label in labels:
        unique_labels.add(label)
    all_labels = sorted(list(unique_labels))
    return all_labels


class BirdsetHSNClassification(BirdsetClassification):
  configuration = "HSN"
  metadata = _get_birdset_metadata(configuration)


class BirdsetNBPClassification(BirdsetClassification):
  configuration = "NBP"
  metadata = _get_birdset_metadata(configuration)


class BirdsetPOWClassification(BirdsetClassification):
  configuration = "POW"
  metadata = _get_birdset_metadata(configuration)


class BirdsetSSWClassification(BirdsetClassification):
  configuration = "SSW"
  metadata = _get_birdset_metadata(configuration)


class BirdsetSNEClassification(BirdsetClassification):
  configuration = "SNE"
  metadata = _get_birdset_metadata(configuration)


class BirdsetPERClassification(BirdsetClassification):
  configuration = "PER"
  metadata = _get_birdset_metadata(configuration)


class BirdsetNESClassification(BirdsetClassification):
  configuration = "NES"
  metadata = _get_birdset_metadata(configuration)


class BirdsetUHHClassification(BirdsetClassification):
  configuration = "UHH"
  metadata = _get_birdset_metadata(configuration)


class BirdsetXCMClassification(BirdsetClassification):
  configuration = "XCM"
  metadata = _get_birdset_metadata(configuration)


class BirdsetXCLClassification(BirdsetClassification):
  configuration = "XCL"
  metadata = _get_birdset_metadata(configuration)

