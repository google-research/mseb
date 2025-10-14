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

"""SVQ salient term segmentation tasks."""

from typing import Iterable

from mseb import types
from mseb.datasets import simple_voice_questions as svq
from mseb.evaluators import segmentation_evaluator
from mseb.tasks import segmentation


class SVQSalientTermSegmentation(segmentation.SegmentationTask):
  """Base class for salient term segmentation on the SVQ dataset."""

  locale: str | None = None

  @property
  def sub_tasks(self) -> list[str]:
    return ["salient_term"]

  def sounds(self) -> Iterable[types.Sound]:
    if self.locale is None:
      raise ValueError(
          "`locale` must be set by a concrete task subclass."
      )

    svq_dataset = svq.SimpleVoiceQuestionsDataset()
    for utt_id, record in svq_dataset.utt_id_to_record.items():
      if record["locale"] == self.locale:
        yield svq_dataset.get_sound_by_id(utt_id)

  def examples(
      self, sub_task: str
  ) -> Iterable[segmentation_evaluator.SegmentationReference]:
    if self.locale is None:
      raise ValueError(
          "`locale` must be set by a concrete task subclass."
      )

    svq_dataset = svq.SimpleVoiceQuestionsDataset()
    for utt_id, record in svq_dataset.utt_id_to_record.items():
      if record["locale"] == self.locale:
        terms = record.get("topk_salient_terms")
        timestamps = record.get("topk_salient_terms_timestamps")

        if not terms or not timestamps or len(terms) != len(timestamps):
          continue

        segments = [
            segmentation_evaluator.Segment(
                embedding=term,
                start_time=ts[0],
                end_time=ts[1],
            )
            for term, ts in zip(terms, timestamps)
        ]
        yield segmentation_evaluator.SegmentationReference(
            example_id=utt_id,
            segments=segments
        )


class SVQEnUsSalientTermSegmentation(SVQSalientTermSegmentation):
  locale = "en_us"
  metadata = types.TaskMetadata(
      name="SVQEnUsSalientTermSegmentation",
      description=(
          "Salient term segmentation task on the Simple Voice Questions (SVQ) "
          "dataset for en_us."
      ),
      reference="TODO",
      type="SalientTermSegmentation",
      category="speech",
      main_score="NDCG",
      revision="1.0.0",
      dataset=types.Dataset(
          path="https://huggingface.co/datasets/google/svq",
          revision="1.0.0",
      ),
      scores=[
          segmentation_evaluator.mean_average_precision(),
          segmentation_evaluator.normalized_discounted_cumulative_gain(),
          segmentation_evaluator.word_error_rate(),
          segmentation_evaluator.timestamps_accuracy(),
          segmentation_evaluator.embeddings_accuracy(),
          segmentation_evaluator.timestamps_and_embeddings_accuracy(),
      ],
      eval_splits=["test"],
      eval_langs=["en-US"],
      domains=["speech"],
      task_subtypes=["segmentation"],
  )
