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

"""Speech Massive intent classification tasks."""

import functools
import os
import re
from typing import Any, Iterable

from mseb import types
from mseb.datasets import speech_massive
from mseb.evaluators import classification_evaluator
from mseb.tasks import classification


class SpeechMassiveIntentClassification(classification.ClassificationTask):
  """Speech Massive intent classification task."""

  locale: str | None = None
  filename: str | None = None

  @property
  def task_type(self) -> str:
    return "multi_class"

  @property
  def weights_dir(self) -> str:
    def _camel_to_snake(name: str) -> str:
      pattern = re.compile(r"(?<!^)(?=[A-Z])")
      return pattern.sub("_", name).lower()

    name = self.__class__.__name__
    return os.path.join(super().weights_dir, _camel_to_snake(name))

  @property
  def sub_tasks(self) -> list[str]:
    return ["intent_classification"]

  @functools.cached_property
  def speech_massive_dataset(self) -> speech_massive.SpeechMassiveDataset:
    return speech_massive.SpeechMassiveDataset(filename=self.filename)  # pyrefly: ignore[bad-argument-type]

  def _task_data(self, task_data_key: str, dtype: dict[str, Any] | None = None):
    df = self.speech_massive_dataset.get_task_data(task_data_key, dtype=dtype)
    if self.locale:
      df = df[df.locale == speech_massive.bcp47_by_locale[self.locale]]
    return df

  def multimodal_inputs(self) -> Iterable[types.Sound]:
    dataset = self.speech_massive_dataset
    for example in dataset.get_task_data(with_audio=True).to_dict("records"):
      yield dataset.get_sound(example)

  def multimodal_inputs_beam(self):
    return self.speech_massive_dataset.get_task_sounds_beam()

  def examples(
      self, sub_task: str
  ) -> Iterable[classification_evaluator.ClassificationReference]:
    dataset = self.speech_massive_dataset
    for example in dataset.get_task_data().to_dict("records"):
      yield classification_evaluator.ClassificationReference(
          example_id=example["path"],
          label_id=example["intent_str"],
      )

  def class_labels(self) -> Iterable[str]:
    return (
        "datetime_query",
        "iot_hue_lightchange",
        "transport_ticket",
        "takeaway_query",
        "qa_stock",
        "general_greet",
        "recommendation_events",
        "music_dislikeness",
        "iot_wemo_off",
        "cooking_recipe",
        "qa_currency",
        "transport_traffic",
        "general_quirky",
        "weather_query",
        "audio_volume_up",
        "email_addcontact",
        "takeaway_order",
        "email_querycontact",
        "iot_hue_lightup",
        "recommendation_locations",
        "play_audiobook",
        "lists_createoradd",
        "news_query",
        "alarm_query",
        "iot_wemo_on",
        "general_joke",
        "qa_definition",
        "social_query",
        "music_settings",
        "audio_volume_other",
        "calendar_remove",
        "iot_hue_lightdim",
        "calendar_query",
        "email_sendemail",
        "iot_cleaning",
        "audio_volume_down",
        "play_radio",
        "cooking_query",
        "datetime_convert",
        "qa_maths",
        "iot_hue_lightoff",
        "iot_hue_lighton",
        "transport_query",
        "music_likeness",
        "email_query",
        "play_music",
        "audio_volume_mute",
        "social_post",
        "alarm_set",
        "qa_factoid",
        "calendar_set",
        "play_game",
        "alarm_remove",
        "lists_remove",
        "transport_taxi",
        "recommendation_movies",
        "iot_coffee",
        "music_query",
        "play_podcasts",
        "lists_query",
    )


# Locale -> (ClassName suffix, eval_lang)
_SPEECH_MASSIVE_LOCALES = {
    "ar_sa": ("ArSa", "ar-SA"),
    "de_de": ("DeDe", "de-DE"),
    "es_es": ("EsEs", "es-ES"),
    "fr_fr": ("FrFr", "fr-FR"),
    "hu_hu": ("HuHu", "hu-HU"),
    "ko_kr": ("KoKr", "ko-KR"),
    "nl_nl": ("NlNl", "nl-NL"),
    "pl_pl": ("PlPl", "pl-PL"),
    "pt_pt": ("PtPt", "pt-PT"),
    "ru_ru": ("RuRu", "ru-RU"),
    "tr_tr": ("TrTr", "tr-TR"),
    "vi_vn": ("ViVn", "vi-VN"),
}


def _make_task_class(base_cls, locale, suffix, eval_lang, description):
  """Dynamically create a locale-specific task class."""
  class_name = (
      f'SpeechMassive{suffix}{base_cls.__name__[len("SpeechMassive"):]}'
  )
  cls = type(
      class_name,
      (base_cls,),
      {
          "locale": locale,
          "filename": f"{eval_lang}/test-?????-of-?????.parquet",
          "metadata": types.TaskMetadata(
              name=class_name,
              description=description,
              reference=(
                  "https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test"
              ),
              documentation_file="speech_massive_classification.md",
              dataset_documentation_file="dataset_speech_massive.md",
              type="IntentClassification",
              category="speech",
              main_score="Accuracy",
              revision="1.0.0",
              dataset=types.Dataset(
                  name="SpeechMassive",
                  path="https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test",
                  revision="2024.08.08",
              ),
              scores=[
                  classification_evaluator.accuracy(),
                  classification_evaluator.top_k_accuracy(k=5),
                  classification_evaluator.balanced_accuracy(),
                  classification_evaluator.weighted_f1(),
                  classification_evaluator.weighted_precision(),
                  classification_evaluator.weighted_recall(),
              ],
              eval_splits=["test"],
              eval_langs=[eval_lang],
              domains=["speech"],
              task_subtypes=["classification"],
          ),
      },
  )
  return cls


# Generate all locale-specific classes and register them in the module.
# Default size.
for _locale, (_suffix, _eval_lang) in _SPEECH_MASSIVE_LOCALES.items():
  _cls = _make_task_class(  # pylint: disable=invalid-name
      base_cls=SpeechMassiveIntentClassification,
      locale=_locale,
      suffix=_suffix,
      eval_lang=_eval_lang,
      description="Speech Massive intent classification task.",
  )
  globals()[_cls.__name__] = _cls
