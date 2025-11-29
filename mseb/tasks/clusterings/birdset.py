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

"""Birdset clustering tasks."""

from typing import Any, Iterable, Type

from mseb import runner as runner_lib
from mseb import types
from mseb.datasets import birdset
from mseb.evaluators import clustering_evaluator
from mseb.tasks import clustering


def _get_birdset_clustering_metadata(configuration: str) -> types.TaskMetadata:
  """Returns TaskMetadata for a Birdset clustering task."""
  return types.TaskMetadata(
      name=f"BirdsetClustering{configuration}",
      description=f"Clustering task on Birdset {configuration} configuration.",
      reference="https://arxiv.org/abs/2403.10380",
      type="Clustering",
      category="audio",
      main_score="VMeasure",
      revision="1.0.0",
      dataset=types.Dataset(
          path="https://huggingface.co/datasets/DBD-research-group/BirdSet",
          revision="1.0.0",
      ),
      scores=[clustering_evaluator.vmeasure_score()],
      eval_splits=["test"],
      eval_langs=["en-US"],  # Assuming primary language, needs confirmation
      domains=["natural sounds"],
      task_subtypes=["clustering"],
  )


class BirdsetClustering(clustering.ClusteringTask):
  """Birdset clustering."""

  _birdset_dataset: birdset.BirdsetDataset
  configuration: str = "HSN"

  def _get_dataset(self) -> birdset.BirdsetDataset:
    return birdset.BirdsetDataset(
        split="test_5s", configuration=self.configuration
    )

  def setup(
      self, runner_cls: Type[runner_lib.EncoderRunner] | None = None, **kwargs
  ):
    self._birdset_dataset = self._get_dataset()

  def _task_data(self):
    return self._birdset_dataset.get_task_data()

  def _get_label(self, example: dict[str, Any]) -> str:
    """Get label from example."""
    # Pick only first label from multiple labels.
    if example["ebird_code_multilabel"]:
      return example["ebird_code_multilabel"][0]
    return "no_label"

  @property
  def sub_tasks(self) -> list[str]:
    return ["clustering"]

  def sounds(self) -> Iterable[types.Sound]:
    for example in self._task_data().to_dict("records"):
      yield self._birdset_dataset.get_sound(example)

  def examples(
      self, sub_task: str
  ) -> Iterable[clustering_evaluator.ClusteringExample]:
    """Get (utt_id, label) examples from Birdset dataset."""
    for example in self._task_data().to_dict("records"):
      yield clustering_evaluator.ClusteringExample(
          str(example["filepath"]), self._get_label(example)
      )


class BirdsetClusteringHSN(BirdsetClustering):
  configuration = "HSN"
  metadata = _get_birdset_clustering_metadata(configuration)


class BirdsetClusteringNBP(BirdsetClustering):
  configuration = "NBP"
  metadata = _get_birdset_clustering_metadata(configuration)


class BirdsetClusteringPOW(BirdsetClustering):
  configuration = "POW"
  metadata = _get_birdset_clustering_metadata(configuration)


class BirdsetClusteringSSW(BirdsetClustering):
  configuration = "SSW"
  metadata = _get_birdset_clustering_metadata(configuration)


class BirdsetClusteringSNE(BirdsetClustering):
  configuration = "SNE"
  metadata = _get_birdset_clustering_metadata(configuration)


class BirdsetClusteringPER(BirdsetClustering):
  configuration = "PER"
  metadata = _get_birdset_clustering_metadata(configuration)


class BirdsetClusteringNES(BirdsetClustering):
  configuration = "NES"
  metadata = _get_birdset_clustering_metadata(configuration)


class BirdsetClusteringUHH(BirdsetClustering):
  configuration = "UHH"
  metadata = _get_birdset_clustering_metadata(configuration)


class BirdsetClusteringXCM(BirdsetClustering):
  configuration = "XCM"
  metadata = _get_birdset_clustering_metadata(configuration)


class BirdsetClusteringXCL(BirdsetClustering):
  configuration = "XCL"
  metadata = _get_birdset_clustering_metadata(configuration)
