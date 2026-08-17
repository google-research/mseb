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

"""SVQ salient term segmentation selection tasks."""

from collections.abc import Sequence
import functools
import os
from typing import Any, Iterable

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


class SVQSalientTermSegmentationSelectionTask(
    segmentation.SegmentationSelectionTask
):
  """Base class for salient term segmentation selection on the SVQ dataset."""

  locale: str | None = None

  @functools.cached_property
  def svq_dataset(self) -> svq.SimpleVoiceQuestionsDataset:
    return svq.SimpleVoiceQuestionsDataset()

  def _task_data(self, task_data_key: str, dtype: dict[str, Any] | None = None):
    df = self.svq_dataset.get_task_data(task_data_key, dtype=dtype)
    if self.locale:
      df = df[df.locale == self.locale]
    return df

  @property
  def sub_tasks(self) -> list[str]:
    return list(_filter_fn_by_sub_task.keys())

  def multimodal_inputs(self) -> Iterable[types.Sound]:
    if self.locale is None:
      raise ValueError("`locale` must be set by a concrete task subclass.")

    df = self._task_data(
        "salient_term",
        dtype={
            "locale": str,
            "utt_id": str,
        },
    )
    for record in df.to_dict("records"):
      utt_id = record["utt_id"]
      yield self.svq_dataset.get_sound({"utt_id": utt_id})

  def examples(
      self, sub_task: str
  ) -> Iterable[segmentation_evaluator.SegmentationReference]:
    if self.locale is None:
      raise ValueError("`locale` must be set by a concrete task subclass.")

    filter_fn = _filter_fn_by_sub_task[sub_task]
    df = self._task_data(
        _base_sub_task(sub_task),
        dtype={
            "locale": str,
            "utt_id": str,
            "topk_salient_terms": Sequence[str],
            "topk_salient_terms_timestamps": Sequence[tuple[float, float]],
        },
    )
    for record in df.to_dict("records"):
      if filter_fn(record):
        utt_id = record["utt_id"]
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

  @property
  def embeddings_dir(self) -> str:
    assert self.locale is not None
    return os.path.join(
        super().embeddings_dir, f"svq_{self.locale}_salient_terms"
    )

  def salient_term_lists(
      self,
  ) -> Iterable[tuple[str, Sequence[segmentation_evaluator.Segment]]]:
    if self.locale is None:
      raise ValueError("`locale` must be set by a concrete task subclass.")

    svq_dataset = self.svq_dataset
    for record in svq_dataset.get_task_data(
        "salient_term",
        dtype={
            "locale": str,
            "utt_id": str,
            "candidate_salient_terms": Sequence[str],
            "candidate_salient_terms_timestamps": Sequence[tuple[float, float]],
        },
    ).to_dict("records"):
      if record["locale"] == self.locale:
        terms = record.get("candidate_salient_terms")
        timestamps = record.get("candidate_salient_terms_timestamps")
        if terms:
          yield (
              record["utt_id"],
              [
                  segmentation_evaluator.Segment(
                      embedding=term,
                      start_time=timestamp[0],
                      end_time=timestamp[1],
                  )
                  for term, timestamp in zip(terms, timestamps)  # pyrefly: ignore[bad-argument-type]
              ],
          )


# Locale -> (ClassName suffix, eval_lang)
_SVQ_LOCALES = {
    "ar_eg": ("ArEg", "ar-EG"),
    "ar_x_gulf": ("ArXGulf", "ar-x-gulf"),
    "ar_x_levant": ("ArXLevant", "ar-x-levant"),
    "ar_x_maghrebi": ("ArXMaghrebi", "ar-x-maghrebi"),
    "bn_bd": ("BnBd", "bn-BD"),
    "bn_in": ("BnIn", "bn-IN"),
    "en_au": ("EnAu", "en-AU"),
    "en_gb": ("EnGb", "en-GB"),
    "en_in": ("EnIn", "en-IN"),
    "en_ph": ("EnPh", "en-PH"),
    "en_us": ("EnUs", "en-US"),
    "fi_fi": ("FiFi", "fi-FI"),
    "gu_in": ("GuIn", "gu-IN"),
    "hi_in": ("HiIn", "hi-IN"),
    "id_id": ("IdId", "id-ID"),
    "ja_jp": ("JaJp", "ja-JP"),
    "kn_in": ("KnIn", "kn-IN"),
    "ko_kr": ("KoKr", "ko-KR"),
    "ml_in": ("MlIn", "ml-IN"),
    "mr_in": ("MrIn", "mr-IN"),
    "ru_ru": ("RuRu", "ru-RU"),
    "sw": ("Sw", "sw"),
    "ta_in": ("TaIn", "ta-IN"),
    "te_in": ("TeIn", "te-IN"),
    "ur_in": ("UrIn", "ur-IN"),
    "ur_pk": ("UrPk", "ur-PK"),
}


def _make_task_class(base_cls, locale, suffix, eval_lang, description):
  """Dynamically create a locale-specific task class."""
  class_name = f'SVQ{suffix}{base_cls.__name__[len("SVQ"):]}'
  cls = type(
      class_name,
      (base_cls,),
      {
          "locale": locale,
          "metadata": types.TaskMetadata(
              name=class_name,
              description=description,
              reference="https://huggingface.co/datasets/google/svq",
              documentation_file="svq_segmentation.md",
              dataset_documentation_file="dataset_svq.md",
              type="SalientTermSegmentation",
              category="speech",
              main_score="NDCG",
              revision="1.0.0",
              dataset=types.Dataset(
                  name="SVQ",
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
              eval_langs=[eval_lang],
              domains=["speech"],
              task_subtypes=["segmentation"],
          ),
      },
  )
  return cls


# Generate all locale-specific classes and register them in the module.
# Default size.
for _locale, (_suffix, _eval_lang) in _SVQ_LOCALES.items():
  _cls = _make_task_class(  # pylint: disable=invalid-name
      base_cls=SVQSalientTermSegmentationSelectionTask,
      locale=_locale,
      suffix=_suffix,
      eval_lang=_eval_lang,
      description="Salient term segmentation selection task.",
  )
  globals()[_cls.__name__] = _cls
