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

"""Speech Massive Speaker-gender classification tasks."""

import os
from typing import Iterable

from mseb import types
from mseb.datasets import speech_massive
from mseb.evaluators import classification_evaluator
from mseb.tasks import classification


class SpeechMassiveSpeakerGenderClassification(
    classification.ClassificationTask):
  """Speech Massive speaker-gender classification task."""

  locale: str | None = None
  filename: str | None = None

  @property
  def task_type(self) -> str:
    return "multi_class"

  @property
  def weights_dir(self) -> str:
    assert self.locale is not None
    return os.path.join(
        super().weights_dir,
        f"speech_massive_{self.locale}_speaker_gender_classification",
    )

  @property
  def sub_tasks(self) -> list[str]:
    return ["speaker_gender_classification"]

  def _get_dataset(self) -> speech_massive.SpeechMassiveDataset:
    return speech_massive.SpeechMassiveDataset(filename=self.filename)

  def sounds(self) -> Iterable[types.Sound]:
    dataset = self._get_dataset()
    for example in dataset.get_task_data().to_dict("records"):
      yield dataset.get_sound(example)

  def examples(
      self, sub_task: str
  ) -> Iterable[classification_evaluator.ClassificationReference]:
    dataset = self._get_dataset()
    class_labels = set(self.class_labels())
    for example in dataset.get_task_data().to_dict("records"):
      if example["speaker_sex"] not in class_labels:
        continue
      yield classification_evaluator.ClassificationReference(
          example_id=example["path"],
          label_id=example["speaker_sex"],
      )

  def class_labels(self) -> Iterable[str]:
    return (
        "Female",
        "Male",
    )


class SpeechMassiveArSaSpeakerGenderClassification(
    SpeechMassiveSpeakerGenderClassification):
  locale = "ar_sa"
  filename = "ar-SA/test-?????-of-?????.parquet"
  metadata = types.TaskMetadata(
      name="SpeechMassiveArSaSpeakerGenderClassification",
      description="Speaker-gender classification task.",
      reference="https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test",
      documentation_file="speech_massive_classification.md",
      dataset_documentation_file="dataset_speech_massive.md",
      type="SpeakerGenderClassification",
      category="speech",
      main_score="Accuracy",
      revision="1.0.0",
      dataset=types.Dataset(
          path="https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test",
          revision="2024.08.08",
      ),
      scores=[
          classification_evaluator.accuracy(),
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


class SpeechMassiveDeDeSpeakerGenderClassification(
    SpeechMassiveSpeakerGenderClassification):
  locale = "de_de"
  filename = "de-DE/test-?????-of-?????.parquet"
  metadata = types.TaskMetadata(
      name="SpeechMassiveDeDeSpeakerGenderClassification",
      description="Speaker-gender classification task.",
      reference="https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test",
      documentation_file="speech_massive_classification.md",
      dataset_documentation_file="dataset_speech_massive.md",
      type="SpeakerGenderClassification",
      category="speech",
      main_score="Accuracy",
      revision="1.0.0",
      dataset=types.Dataset(
          path="https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test",
          revision="2024.08.08",
      ),
      scores=[
          classification_evaluator.accuracy(),
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


class SpeechMassiveEsEsSpeakerGenderClassification(
    SpeechMassiveSpeakerGenderClassification):
  locale = "es_es"
  filename = "es-ES/test-?????-of-?????.parquet"
  metadata = types.TaskMetadata(
      name="SpeechMassiveEsEsSpeakerGenderClassification",
      description="Speaker-gender classification task.",
      reference="https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test",
      documentation_file="speech_massive_classification.md",
      dataset_documentation_file="dataset_speech_massive.md",
      type="SpeakerGenderClassification",
      category="speech",
      main_score="Accuracy",
      revision="1.0.0",
      dataset=types.Dataset(
          path="https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test",
          revision="2024.08.08",
      ),
      scores=[
          classification_evaluator.accuracy(),
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


class SpeechMassiveFrFrSpeakerGenderClassification(
    SpeechMassiveSpeakerGenderClassification):
  locale = "fr_fr"
  filename = "fr-FR/test-?????-of-?????.parquet"
  metadata = types.TaskMetadata(
      name="SpeechMassiveFrFrSpeakerGenderClassification",
      description="Speaker-gender classification task.",
      reference="https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test",
      documentation_file="speech_massive_classification.md",
      dataset_documentation_file="dataset_speech_massive.md",
      type="SpeakerGenderClassification",
      category="speech",
      main_score="Accuracy",
      revision="1.0.0",
      dataset=types.Dataset(
          path="https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test",
          revision="2024.08.08",
      ),
      scores=[
          classification_evaluator.accuracy(),
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


class SpeechMassiveHuHuSpeakerGenderClassification(
    SpeechMassiveSpeakerGenderClassification):
  locale = "hu_hu"
  filename = "hu-HU/test-?????-of-?????.parquet"
  metadata = types.TaskMetadata(
      name="SpeechMassiveHuHuSpeakerGenderClassification",
      description="Speaker-gender classification task.",
      reference="https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test",
      documentation_file="speech_massive_classification.md",
      dataset_documentation_file="dataset_speech_massive.md",
      type="SpeakerGenderClassification",
      category="speech",
      main_score="Accuracy",
      revision="1.0.0",
      dataset=types.Dataset(
          path="https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test",
          revision="2024.08.08",
      ),
      scores=[
          classification_evaluator.accuracy(),
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


class SpeechMassiveKoKrSpeakerGenderClassification(
    SpeechMassiveSpeakerGenderClassification):
  locale = "ko_kr"
  filename = "ko-KR/test-?????-of-?????.parquet"
  metadata = types.TaskMetadata(
      name="SpeechMassiveKoKrSpeakerGenderClassification",
      description="Speaker-gender classification task.",
      reference="https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test",
      documentation_file="speech_massive_classification.md",
      dataset_documentation_file="dataset_speech_massive.md",
      type="SpeakerGenderClassification",
      category="speech",
      main_score="Accuracy",
      revision="1.0.0",
      dataset=types.Dataset(
          path="https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test",
          revision="2024.08.08",
      ),
      scores=[
          classification_evaluator.accuracy(),
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


class SpeechMassiveNlNlSpeakerGenderClassification(
    SpeechMassiveSpeakerGenderClassification):
  locale = "nl_nl"
  filename = "nl-NL/test-?????-of-?????.parquet"
  metadata = types.TaskMetadata(
      name="SpeechMassiveNlNlSpeakerGenderClassification",
      description="Speaker-gender classification task.",
      reference="https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test",
      documentation_file="speech_massive_classification.md",
      dataset_documentation_file="dataset_speech_massive.md",
      type="SpeakerGenderClassification",
      category="speech",
      main_score="Accuracy",
      revision="1.0.0",
      dataset=types.Dataset(
          path="https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test",
          revision="2024.08.08",
      ),
      scores=[
          classification_evaluator.accuracy(),
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


class SpeechMassivePlPlSpeakerGenderClassification(
    SpeechMassiveSpeakerGenderClassification):
  locale = "pl_pl"
  filename = "pl-PL/test-?????-of-?????.parquet"
  metadata = types.TaskMetadata(
      name="SpeechMassivePlPlSpeakerGenderClassification",
      description="Speaker-gender classification task.",
      reference="https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test",
      documentation_file="speech_massive_classification.md",
      dataset_documentation_file="dataset_speech_massive.md",
      type="SpeakerGenderClassification",
      category="speech",
      main_score="Accuracy",
      revision="1.0.0",
      dataset=types.Dataset(
          path="https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test",
          revision="2024.08.08",
      ),
      scores=[
          classification_evaluator.accuracy(),
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


class SpeechMassivePtPtSpeakerGenderClassification(
    SpeechMassiveSpeakerGenderClassification):
  locale = "pt_pt"
  filename = "pt-PT/test-?????-of-?????.parquet"
  metadata = types.TaskMetadata(
      name="SpeechMassivePtPtSpeakerGenderClassification",
      description="Speaker-gender classification task.",
      reference="https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test",
      documentation_file="speech_massive_classification.md",
      dataset_documentation_file="dataset_speech_massive.md",
      type="SpeakerGenderClassification",
      category="speech",
      main_score="Accuracy",
      revision="1.0.0",
      dataset=types.Dataset(
          path="https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test",
          revision="2024.08.08",
      ),
      scores=[
          classification_evaluator.accuracy(),
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


class SpeechMassiveRuRuSpeakerGenderClassification(
    SpeechMassiveSpeakerGenderClassification):
  locale = "ru_ru"
  filename = "ru-RU/test-?????-of-?????.parquet"
  metadata = types.TaskMetadata(
      name="SpeechMassiveRuRuSpeakerGenderClassification",
      description="Speaker-gender classification task.",
      reference="https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test",
      documentation_file="speech_massive_classification.md",
      dataset_documentation_file="dataset_speech_massive.md",
      type="SpeakerGenderClassification",
      category="speech",
      main_score="Accuracy",
      revision="1.0.0",
      dataset=types.Dataset(
          path="https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test",
          revision="2024.08.08",
      ),
      scores=[
          classification_evaluator.accuracy(),
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


class SpeechMassiveTrTrSpeakerGenderClassification(
    SpeechMassiveSpeakerGenderClassification):
  locale = "tr_tr"
  filename = "tr-TR/test-?????-of-?????.parquet"
  metadata = types.TaskMetadata(
      name="SpeechMassiveTrTrSpeakerGenderClassification",
      description="Speaker-gender classification task.",
      reference="https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test",
      documentation_file="speech_massive_classification.md",
      dataset_documentation_file="dataset_speech_massive.md",
      type="SpeakerGenderClassification",
      category="speech",
      main_score="Accuracy",
      revision="1.0.0",
      dataset=types.Dataset(
          path="https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test",
          revision="2024.08.08",
      ),
      scores=[
          classification_evaluator.accuracy(),
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


class SpeechMassiveViVnSpeakerGenderClassification(
    SpeechMassiveSpeakerGenderClassification):
  locale = "vi_vn"
  filename = "vi-VN/test-?????-of-?????.parquet"
  metadata = types.TaskMetadata(
      name="SpeechMassiveViVnSpeakerGenderClassification",
      description="Speaker-gender classification task.",
      reference="https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test",
      documentation_file="speech_massive_classification.md",
      dataset_documentation_file="dataset_speech_massive.md",
      type="SpeakerGenderClassification",
      category="speech",
      main_score="Accuracy",
      revision="1.0.0",
      dataset=types.Dataset(
          path="https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test",
          revision="2024.08.08",
      ),
      scores=[
          classification_evaluator.accuracy(),
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
