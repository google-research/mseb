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

"""Speech Massive intent classification tasks."""

import os
from typing import Iterable

from mseb import types
from mseb.datasets import speech_massive
from mseb.evaluators import classification_evaluator
from mseb.tasks import classification


class SpeechMassiveIntentClassification(classification.ClassificationTask):
  """Speech Massive intent classification task."""

  locale: str | None = None

  @property
  def task_type(self) -> str:
    return "multi_class"

  @property
  def weights_dir(self) -> str:
    assert self.locale is not None
    return os.path.join(
        super().weights_dir,
        f"speech_massive_{self.locale}_intent_classification",
    )

  @property
  def sub_tasks(self) -> list[str]:
    return ["intent_classification"]

  def sounds(self) -> Iterable[types.Sound]:
    dataset = speech_massive.SpeechMassiveDataset(language=self.locale)
    for example in dataset.get_task_data().itertuples():
      yield dataset.get_sound(example)

  def examples(
      self, sub_task: str
  ) -> Iterable[classification_evaluator.ClassificationReference]:
    dataset = speech_massive.SpeechMassiveDataset(language=self.locale)
    for example in dataset.get_task_data().itertuples():
      yield classification_evaluator.ClassificationReference(
          example_id=example.path,
          label_id=example.intent_str,
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


class SpeechMassiveArSaIntentClassification(SpeechMassiveIntentClassification):
  locale = "ar_sa"
  metadata = types.TaskMetadata(
      name="SpeechMassiveArSaIntentClassification",
      description="Intent classification task.",
      reference="TODO",
      type="IntentClassification",
      category="speech",
      main_score="Accuracy",
      revision="1.0.0",
      dataset=types.Dataset(
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
      eval_langs=["ar-SA"],
      domains=["speech"],
      task_subtypes=["classification"],
  )


class SpeechMassiveDeDeIntentClassification(SpeechMassiveIntentClassification):
  locale = "de_de"
  metadata = types.TaskMetadata(
      name="SpeechMassiveDeDeIntentClassification",
      description="Intent classification task.",
      reference="TODO",
      type="IntentClassification",
      category="speech",
      main_score="Accuracy",
      revision="1.0.0",
      dataset=types.Dataset(
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
      eval_langs=["de-DE"],
      domains=["speech"],
      task_subtypes=["classification"],
  )


class SpeechMassiveEsEsIntentClassification(SpeechMassiveIntentClassification):
  locale = "es_es"
  metadata = types.TaskMetadata(
      name="SpeechMassiveEsEsIntentClassification",
      description="Intent classification task.",
      reference="TODO",
      type="IntentClassification",
      category="speech",
      main_score="Accuracy",
      revision="1.0.0",
      dataset=types.Dataset(
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
      eval_langs=["es-ES"],
      domains=["speech"],
      task_subtypes=["classification"],
  )


class SpeechMassiveFrFrIntentClassification(SpeechMassiveIntentClassification):
  locale = "fr_fr"
  metadata = types.TaskMetadata(
      name="SpeechMassiveFrFrIntentClassification",
      description="Intent classification task.",
      reference="TODO",
      type="IntentClassification",
      category="speech",
      main_score="Accuracy",
      revision="1.0.0",
      dataset=types.Dataset(
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
      eval_langs=["fr-FR"],
      domains=["speech"],
      task_subtypes=["classification"],
  )


class SpeechMassiveHuHuIntentClassification(SpeechMassiveIntentClassification):
  locale = "hu_hu"
  metadata = types.TaskMetadata(
      name="SpeechMassiveHuHuIntentClassification",
      description="Intent classification task.",
      reference="TODO",
      type="IntentClassification",
      category="speech",
      main_score="Accuracy",
      revision="1.0.0",
      dataset=types.Dataset(
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
      eval_langs=["hu-HU"],
      domains=["speech"],
      task_subtypes=["classification"],
  )


class SpeechMassiveKoKrIntentClassification(SpeechMassiveIntentClassification):
  locale = "ko_kr"
  metadata = types.TaskMetadata(
      name="SpeechMassiveKoKrIntentClassification",
      description="Intent classification task.",
      reference="TODO",
      type="IntentClassification",
      category="speech",
      main_score="Accuracy",
      revision="1.0.0",
      dataset=types.Dataset(
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
      eval_langs=["ko-KR"],
      domains=["speech"],
      task_subtypes=["classification"],
  )


class SpeechMassiveNlNlIntentClassification(SpeechMassiveIntentClassification):
  locale = "nl_nl"
  metadata = types.TaskMetadata(
      name="SpeechMassiveNlNlIntentClassification",
      description="Intent classification task.",
      reference="TODO",
      type="IntentClassification",
      category="speech",
      main_score="Accuracy",
      revision="1.0.0",
      dataset=types.Dataset(
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
      eval_langs=["nl-NL"],
      domains=["speech"],
      task_subtypes=["classification"],
  )


class SpeechMassivePlPlIntentClassification(SpeechMassiveIntentClassification):
  locale = "pl_pl"
  metadata = types.TaskMetadata(
      name="SpeechMassivePlPlIntentClassification",
      description="Intent classification task.",
      reference="TODO",
      type="IntentClassification",
      category="speech",
      main_score="Accuracy",
      revision="1.0.0",
      dataset=types.Dataset(
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
      eval_langs=["pl-PL"],
      domains=["speech"],
      task_subtypes=["classification"],
  )


class SpeechMassivePtPtIntentClassification(SpeechMassiveIntentClassification):
  locale = "pt_pt"
  metadata = types.TaskMetadata(
      name="SpeechMassivePtPtIntentClassification",
      description="Intent classification task.",
      reference="TODO",
      type="IntentClassification",
      category="speech",
      main_score="Accuracy",
      revision="1.0.0",
      dataset=types.Dataset(
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
      eval_langs=["pt-PT"],
      domains=["speech"],
      task_subtypes=["classification"],
  )


class SpeechMassiveRuRuIntentClassification(SpeechMassiveIntentClassification):
  locale = "ru_ru"
  metadata = types.TaskMetadata(
      name="SpeechMassiveRuRuIntentClassification",
      description="Intent classification task.",
      reference="TODO",
      type="IntentClassification",
      category="speech",
      main_score="Accuracy",
      revision="1.0.0",
      dataset=types.Dataset(
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
      eval_langs=["ru-RU"],
      domains=["speech"],
      task_subtypes=["classification"],
  )


class SpeechMassiveTrTrIntentClassification(SpeechMassiveIntentClassification):
  locale = "tr_tr"
  metadata = types.TaskMetadata(
      name="SpeechMassiveTrTrIntentClassification",
      description="Intent classification task.",
      reference="TODO",
      type="IntentClassification",
      category="speech",
      main_score="Accuracy",
      revision="1.0.0",
      dataset=types.Dataset(
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
      eval_langs=["tr-TR"],
      domains=["speech"],
      task_subtypes=["classification"],
  )


class SpeechMassiveViVnIntentClassification(SpeechMassiveIntentClassification):
  locale = "vi_vn"
  metadata = types.TaskMetadata(
      name="SpeechMassiveViVnIntentClassification",
      description="Intent classification task.",
      reference="TODO",
      type="IntentClassification",
      category="speech",
      main_score="Accuracy",
      revision="1.0.0",
      dataset=types.Dataset(
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
      eval_langs=["vi-VN"],
      domains=["speech"],
      task_subtypes=["classification"],
  )
