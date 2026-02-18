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

"""SVQ salient term segmentation tasks."""

from typing import Iterable

from mseb import types
from mseb.datasets import simple_voice_questions as svq
from mseb.evaluators import segmentation_evaluator
from mseb.tasks import segmentation


_filter_fn_by_sub_task = {
    "salient_term": lambda x: True,
    "salient_term:clean": lambda x: x["environment"] == "clean",
    "salient_term:media_noise": lambda x: x["environment"] == "media_noise",
    "salient_term:traffic_noise": lambda x: x["environment"] == "traffic_noise",
    "salient_term:background_speech": (
        lambda x: x["environment"] == "background_speech"
    ),
}


def _base_sub_task(sub_task: str) -> str:
  return sub_task.split(":")[0]


class SVQSalientTermSegmentation(segmentation.SegmentationTask):
  """Base class for salient term segmentation on the SVQ dataset."""

  locale: str | None = None

  def _get_dataset(self) -> svq.SimpleVoiceQuestionsDataset:
    return svq.SimpleVoiceQuestionsDataset()

  @property
  def sub_tasks(self) -> list[str]:
    return list(_filter_fn_by_sub_task.keys())

  def sounds(self) -> Iterable[types.Sound]:
    if self.locale is None:
      raise ValueError("`locale` must be set by a concrete task subclass.")

    svq_dataset = self._get_dataset()
    for utt_id, record in svq_dataset.utt_id_to_record.items():
      if record["locale"] == self.locale:
        yield svq_dataset.get_sound({"utt_id": utt_id})

  def examples(
      self, sub_task: str
  ) -> Iterable[segmentation_evaluator.SegmentationReference]:
    if self.locale is None:
      raise ValueError("`locale` must be set by a concrete task subclass.")

    filter_fn = _filter_fn_by_sub_task[sub_task]
    svq_dataset = self._get_dataset()
    for utt_id, record in svq_dataset.utt_id_to_record.items():
      if record["locale"] == self.locale and filter_fn(record):
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
            example_id=utt_id, segments=segments
        )


class SVQArEgSalientTermSegmentation(SVQSalientTermSegmentation):
  locale = "ar_eg"
  metadata = types.TaskMetadata(
      name="SVQArEgSalientTermSegmentation",
      description=(
          "Salient term segmentation task on the Simple Voice Questions (SVQ) "
          "dataset for ar_eg."
      ),
      reference="https://huggingface.co/datasets/google/svq",
      documentation_file="svq_segmentation.md",
      dataset_documentation_file="dataset_svq.md",
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
      eval_langs=["ar-EG"],
      domains=["speech"],
      task_subtypes=["segmentation"],
  )


class SVQArXGulfSalientTermSegmentation(SVQSalientTermSegmentation):
  locale = "ar_x_gulf"
  metadata = types.TaskMetadata(
      name="SVQArXGulfSalientTermSegmentation",
      description=(
          "Salient term segmentation task on the Simple Voice Questions (SVQ) "
          "dataset for ar_x_gulf."
      ),
      reference="https://huggingface.co/datasets/google/svq",
      documentation_file="svq_segmentation.md",
      dataset_documentation_file="dataset_svq.md",
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
      eval_langs=["ar-x-gulf"],
      domains=["speech"],
      task_subtypes=["segmentation"],
  )


class SVQArXLevantSalientTermSegmentation(SVQSalientTermSegmentation):
  locale = "ar_x_levant"
  metadata = types.TaskMetadata(
      name="SVQArXLevantSalientTermSegmentation",
      description=(
          "Salient term segmentation task on the Simple Voice Questions (SVQ) "
          "dataset for ar_x_levant."
      ),
      reference="https://huggingface.co/datasets/google/svq",
      documentation_file="svq_segmentation.md",
      dataset_documentation_file="dataset_svq.md",
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
      eval_langs=["ar-x-levant"],
      domains=["speech"],
      task_subtypes=["segmentation"],
  )


class SVQArXMaghrebiSalientTermSegmentation(SVQSalientTermSegmentation):
  locale = "ar_x_maghrebi"
  metadata = types.TaskMetadata(
      name="SVQArXMaghrebiSalientTermSegmentation",
      description=(
          "Salient term segmentation task on the Simple Voice Questions (SVQ) "
          "dataset for ar_x_maghrebi."
      ),
      reference="https://huggingface.co/datasets/google/svq",
      documentation_file="svq_segmentation.md",
      dataset_documentation_file="dataset_svq.md",
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
      eval_langs=["ar-x-maghrebi"],
      domains=["speech"],
      task_subtypes=["segmentation"],
  )


class SVQBnBdSalientTermSegmentation(SVQSalientTermSegmentation):
  locale = "bn_bd"
  metadata = types.TaskMetadata(
      name="SVQBnBdSalientTermSegmentation",
      description=(
          "Salient term segmentation task on the Simple Voice Questions (SVQ) "
          "dataset for bn_bd."
      ),
      reference="https://huggingface.co/datasets/google/svq",
      documentation_file="svq_segmentation.md",
      dataset_documentation_file="dataset_svq.md",
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
      eval_langs=["bn-BD"],
      domains=["speech"],
      task_subtypes=["segmentation"],
  )


class SVQBnInSalientTermSegmentation(SVQSalientTermSegmentation):
  locale = "bn_in"
  metadata = types.TaskMetadata(
      name="SVQBnInSalientTermSegmentation",
      description=(
          "Salient term segmentation task on the Simple Voice Questions (SVQ) "
          "dataset for bn_in."
      ),
      reference="https://huggingface.co/datasets/google/svq",
      documentation_file="svq_segmentation.md",
      dataset_documentation_file="dataset_svq.md",
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
      eval_langs=["bn-IN"],
      domains=["speech"],
      task_subtypes=["segmentation"],
  )


class SVQEnAuSalientTermSegmentation(SVQSalientTermSegmentation):
  locale = "en_au"
  metadata = types.TaskMetadata(
      name="SVQEnAuSalientTermSegmentation",
      description=(
          "Salient term segmentation task on the Simple Voice Questions (SVQ) "
          "dataset for en_au."
      ),
      reference="https://huggingface.co/datasets/google/svq",
      documentation_file="svq_segmentation.md",
      dataset_documentation_file="dataset_svq.md",
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
      eval_langs=["en-AU"],
      domains=["speech"],
      task_subtypes=["segmentation"],
  )


class SVQEnGbSalientTermSegmentation(SVQSalientTermSegmentation):
  locale = "en_gb"
  metadata = types.TaskMetadata(
      name="SVQEnGbSalientTermSegmentation",
      description=(
          "Salient term segmentation task on the Simple Voice Questions (SVQ) "
          "dataset for en_gb."
      ),
      reference="https://huggingface.co/datasets/google/svq",
      documentation_file="svq_segmentation.md",
      dataset_documentation_file="dataset_svq.md",
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
      eval_langs=["en-GB"],
      domains=["speech"],
      task_subtypes=["segmentation"],
  )


class SVQEnInSalientTermSegmentation(SVQSalientTermSegmentation):
  locale = "en_in"
  metadata = types.TaskMetadata(
      name="SVQEnInSalientTermSegmentation",
      description=(
          "Salient term segmentation task on the Simple Voice Questions (SVQ) "
          "dataset for en_in."
      ),
      reference="https://huggingface.co/datasets/google/svq",
      documentation_file="svq_segmentation.md",
      dataset_documentation_file="dataset_svq.md",
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
      eval_langs=["en-IN"],
      domains=["speech"],
      task_subtypes=["segmentation"],
  )


class SVQEnPhSalientTermSegmentation(SVQSalientTermSegmentation):
  locale = "en_ph"
  metadata = types.TaskMetadata(
      name="SVQEnPhSalientTermSegmentation",
      description=(
          "Salient term segmentation task on the Simple Voice Questions (SVQ) "
          "dataset for en_ph."
      ),
      reference="https://huggingface.co/datasets/google/svq",
      documentation_file="svq_segmentation.md",
      dataset_documentation_file="dataset_svq.md",
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
      eval_langs=["en-PH"],
      domains=["speech"],
      task_subtypes=["segmentation"],
  )


class SVQEnUsSalientTermSegmentation(SVQSalientTermSegmentation):
  locale = "en_us"
  metadata = types.TaskMetadata(
      name="SVQEnUsSalientTermSegmentation",
      description=(
          "Salient term segmentation task on the Simple Voice Questions (SVQ) "
          "dataset for en_us."
      ),
      reference="https://huggingface.co/datasets/google/svq",
      documentation_file="svq_segmentation.md",
      dataset_documentation_file="dataset_svq.md",
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


class SVQFiFiSalientTermSegmentation(SVQSalientTermSegmentation):
  locale = "fi_fi"
  metadata = types.TaskMetadata(
      name="SVQFiFiSalientTermSegmentation",
      description=(
          "Salient term segmentation task on the Simple Voice Questions (SVQ) "
          "dataset for fi_fi."
      ),
      reference="https://huggingface.co/datasets/google/svq",
      documentation_file="svq_segmentation.md",
      dataset_documentation_file="dataset_svq.md",
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
      eval_langs=["fi-FI"],
      domains=["speech"],
      task_subtypes=["segmentation"],
  )


class SVQGuInSalientTermSegmentation(SVQSalientTermSegmentation):
  locale = "gu_in"
  metadata = types.TaskMetadata(
      name="SVQGuInSalientTermSegmentation",
      description=(
          "Salient term segmentation task on the Simple Voice Questions (SVQ) "
          "dataset for gu_in."
      ),
      reference="https://huggingface.co/datasets/google/svq",
      documentation_file="svq_segmentation.md",
      dataset_documentation_file="dataset_svq.md",
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
      eval_langs=["gu-IN"],
      domains=["speech"],
      task_subtypes=["segmentation"],
  )


class SVQHiInSalientTermSegmentation(SVQSalientTermSegmentation):
  locale = "hi_in"
  metadata = types.TaskMetadata(
      name="SVQHiInSalientTermSegmentation",
      description=(
          "Salient term segmentation task on the Simple Voice Questions (SVQ) "
          "dataset for hi_in."
      ),
      reference="https://huggingface.co/datasets/google/svq",
      documentation_file="svq_segmentation.md",
      dataset_documentation_file="dataset_svq.md",
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
      eval_langs=["hi-IN"],
      domains=["speech"],
      task_subtypes=["segmentation"],
  )


class SVQIdIdSalientTermSegmentation(SVQSalientTermSegmentation):
  locale = "id_id"
  metadata = types.TaskMetadata(
      name="SVQIdIdSalientTermSegmentation",
      description=(
          "Salient term segmentation task on the Simple Voice Questions (SVQ) "
          "dataset for id_id."
      ),
      reference="https://huggingface.co/datasets/google/svq",
      documentation_file="svq_segmentation.md",
      dataset_documentation_file="dataset_svq.md",
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
      eval_langs=["id-ID"],
      domains=["speech"],
      task_subtypes=["segmentation"],
  )


class SVQJaJpSalientTermSegmentation(SVQSalientTermSegmentation):
  locale = "ja_jp"
  metadata = types.TaskMetadata(
      name="SVQJaJpSalientTermSegmentation",
      description=(
          "Salient term segmentation task on the Simple Voice Questions (SVQ) "
          "dataset for ja_jp."
      ),
      reference="https://huggingface.co/datasets/google/svq",
      documentation_file="svq_segmentation.md",
      dataset_documentation_file="dataset_svq.md",
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
      eval_langs=["ja-JP"],
      domains=["speech"],
      task_subtypes=["segmentation"],
  )


class SVQKnInSalientTermSegmentation(SVQSalientTermSegmentation):
  locale = "kn_in"
  metadata = types.TaskMetadata(
      name="SVQKnInSalientTermSegmentation",
      description=(
          "Salient term segmentation task on the Simple Voice Questions (SVQ) "
          "dataset for kn_in."
      ),
      reference="https://huggingface.co/datasets/google/svq",
      documentation_file="svq_segmentation.md",
      dataset_documentation_file="dataset_svq.md",
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
      eval_langs=["kn-IN"],
      domains=["speech"],
      task_subtypes=["segmentation"],
  )


class SVQKoKrSalientTermSegmentation(SVQSalientTermSegmentation):
  locale = "ko_kr"
  metadata = types.TaskMetadata(
      name="SVQKoKrSalientTermSegmentation",
      description=(
          "Salient term segmentation task on the Simple Voice Questions (SVQ) "
          "dataset for ko_kr."
      ),
      reference="https://huggingface.co/datasets/google/svq",
      documentation_file="svq_segmentation.md",
      dataset_documentation_file="dataset_svq.md",
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
      eval_langs=["ko-KR"],
      domains=["speech"],
      task_subtypes=["segmentation"],
  )


class SVQMlInSalientTermSegmentation(SVQSalientTermSegmentation):
  locale = "ml_in"
  metadata = types.TaskMetadata(
      name="SVQMlInSalientTermSegmentation",
      description=(
          "Salient term segmentation task on the Simple Voice Questions (SVQ) "
          "dataset for ml_in."
      ),
      reference="https://huggingface.co/datasets/google/svq",
      documentation_file="svq_segmentation.md",
      dataset_documentation_file="dataset_svq.md",
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
      eval_langs=["ml-IN"],
      domains=["speech"],
      task_subtypes=["segmentation"],
  )


class SVQMrInSalientTermSegmentation(SVQSalientTermSegmentation):
  locale = "mr_in"
  metadata = types.TaskMetadata(
      name="SVQMrInSalientTermSegmentation",
      description=(
          "Salient term segmentation task on the Simple Voice Questions (SVQ) "
          "dataset for mr_in."
      ),
      reference="https://huggingface.co/datasets/google/svq",
      documentation_file="svq_segmentation.md",
      dataset_documentation_file="dataset_svq.md",
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
      eval_langs=["mr-IN"],
      domains=["speech"],
      task_subtypes=["segmentation"],
  )


class SVQRuRuSalientTermSegmentation(SVQSalientTermSegmentation):
  locale = "ru_ru"
  metadata = types.TaskMetadata(
      name="SVQRuRuSalientTermSegmentation",
      description=(
          "Salient term segmentation task on the Simple Voice Questions (SVQ) "
          "dataset for ru_ru."
      ),
      reference="https://huggingface.co/datasets/google/svq",
      documentation_file="svq_segmentation.md",
      dataset_documentation_file="dataset_svq.md",
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
      eval_langs=["ru-RU"],
      domains=["speech"],
      task_subtypes=["segmentation"],
  )


class SVQTaInSalientTermSegmentation(SVQSalientTermSegmentation):
  locale = "ta_in"
  metadata = types.TaskMetadata(
      name="SVQTaInSalientTermSegmentation",
      description=(
          "Salient term segmentation task on the Simple Voice Questions (SVQ) "
          "dataset for ta_in."
      ),
      reference="https://huggingface.co/datasets/google/svq",
      documentation_file="svq_segmentation.md",
      dataset_documentation_file="dataset_svq.md",
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
      eval_langs=["ta-IN"],
      domains=["speech"],
      task_subtypes=["segmentation"],
  )


class SVQTeInSalientTermSegmentation(SVQSalientTermSegmentation):
  locale = "te_in"
  metadata = types.TaskMetadata(
      name="SVQTeInSalientTermSegmentation",
      description=(
          "Salient term segmentation task on the Simple Voice Questions (SVQ) "
          "dataset for te_in."
      ),
      reference="https://huggingface.co/datasets/google/svq",
      documentation_file="svq_segmentation.md",
      dataset_documentation_file="dataset_svq.md",
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
      eval_langs=["te-IN"],
      domains=["speech"],
      task_subtypes=["segmentation"],
  )


class SVQUrInSalientTermSegmentation(SVQSalientTermSegmentation):
  locale = "ur_in"
  metadata = types.TaskMetadata(
      name="SVQUrInSalientTermSegmentation",
      description=(
          "Salient term segmentation task on the Simple Voice Questions (SVQ) "
          "dataset for ur_in."
      ),
      reference="https://huggingface.co/datasets/google/svq",
      documentation_file="svq_segmentation.md",
      dataset_documentation_file="dataset_svq.md",
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
      eval_langs=["ur-IN"],
      domains=["speech"],
      task_subtypes=["segmentation"],
  )


class SVQUrPkSalientTermSegmentation(SVQSalientTermSegmentation):
  locale = "ur_pk"
  metadata = types.TaskMetadata(
      name="SVQUrPkSalientTermSegmentation",
      description=(
          "Salient term segmentation task on the Simple Voice Questions (SVQ) "
          "dataset for ur_pk."
      ),
      reference="https://huggingface.co/datasets/google/svq",
      documentation_file="svq_segmentation.md",
      dataset_documentation_file="dataset_svq.md",
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
      eval_langs=["ur-PK"],
      domains=["speech"],
      task_subtypes=["segmentation"],
  )
