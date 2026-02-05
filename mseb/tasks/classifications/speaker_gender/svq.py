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

"""Speech Massive Speaker-gender classification tasks."""

import os
from typing import Iterable

from mseb import types
from mseb.datasets import simple_voice_questions as svq
from mseb.evaluators import classification_evaluator
from mseb.tasks import classification


_filter_fn_by_sub_task = {
    'speaker_gender_classification': lambda x: True,
    'speaker_gender_classification:clean': (
        lambda x: x['environment'] == 'clean'
    ),
    'speaker_gender_classification:media_noise': (
        lambda x: x['environment'] == 'media_noise'
    ),
    'speaker_gender_classification:traffic_noise': (
        lambda x: x['environment'] == 'traffic_noise'
    ),
    'speaker_gender_classification:background_speech': (
        lambda x: x['environment'] == 'background_speech'
    ),
}


def _base_sub_task(sub_task: str) -> str:
  return sub_task.split(':')[0]


class SVQSpeakerGenderClassification(classification.ClassificationTask):
  """SVQ speaker-gender classification task."""

  locale: str | None = None

  @property
  def task_type(self) -> str:
    return 'multi_class'

  @property
  def weights_dir(self) -> str:
    assert self.locale is not None
    return os.path.join(
        super().weights_dir,
        f'svq_{self.locale}_speaker_gender_classification',
    )

  @property
  def sub_tasks(self) -> list[str]:
    return list(_filter_fn_by_sub_task.keys())

  def _get_dataset(self) -> svq.SimpleVoiceQuestionsDataset:
    return svq.SimpleVoiceQuestionsDataset()

  def sounds(self) -> Iterable[types.Sound]:
    dataset = self._get_dataset()
    for example in dataset.get_task_data('utt_index').to_dict('records'):
      if example['locale'] == self.locale:
        yield dataset.get_sound(example)

  def examples(
      self, sub_task: str
  ) -> Iterable[classification_evaluator.ClassificationReference]:
    filter_fn = _filter_fn_by_sub_task[sub_task]
    dataset = self._get_dataset()
    class_labels = set(self.class_labels())
    for example in dataset.get_task_data('utt_index').to_dict('records'):
      if (
          example['locale'] == self.locale
          and filter_fn(example)
          and example['speaker_gender'].capitalize() in class_labels
      ):
        yield classification_evaluator.ClassificationReference(
            example_id=example['utt_id'],
            label_id=example['speaker_gender'].capitalize(),
        )

  def class_labels(self) -> Iterable[str]:
    return (
        'Female',
        'Male',
    )


class SVQArEgSpeakerGenderClassification(SVQSpeakerGenderClassification):
  locale = 'ar_eg'
  metadata = types.TaskMetadata(
      name='SVQArEgSpeakerGenderClassification',
      description='Speaker-gender classification task.',
      reference='TODO',
      type='SpeakerGenderClassification',
      category='speech',
      main_score='Accuracy',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          classification_evaluator.accuracy(),
          classification_evaluator.balanced_accuracy(),
          classification_evaluator.weighted_f1(),
          classification_evaluator.weighted_precision(),
          classification_evaluator.weighted_recall(),
      ],
      eval_splits=['test'],
      eval_langs=['ar-EG'],
      domains=['speech'],
      task_subtypes=['classification'],
  )


class SVQArXGulfSpeakerGenderClassification(SVQSpeakerGenderClassification):
  locale = 'ar_x_gulf'
  metadata = types.TaskMetadata(
      name='SVQArXGulfSpeakerGenderClassification',
      description='Speaker-gender classification task.',
      reference='TODO',
      type='SpeakerGenderClassification',
      category='speech',
      main_score='Accuracy',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          classification_evaluator.accuracy(),
          classification_evaluator.balanced_accuracy(),
          classification_evaluator.weighted_f1(),
          classification_evaluator.weighted_precision(),
          classification_evaluator.weighted_recall(),
      ],
      eval_splits=['test'],
      eval_langs=['ar-x-gulf'],
      domains=['speech'],
      task_subtypes=['classification'],
  )


class SVQArXLevantSpeakerGenderClassification(SVQSpeakerGenderClassification):
  locale = 'ar_x_levant'
  metadata = types.TaskMetadata(
      name='SVQArXLevantSpeakerGenderClassification',
      description='Speaker-gender classification task.',
      reference='TODO',
      type='SpeakerGenderClassification',
      category='speech',
      main_score='Accuracy',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          classification_evaluator.accuracy(),
          classification_evaluator.balanced_accuracy(),
          classification_evaluator.weighted_f1(),
          classification_evaluator.weighted_precision(),
          classification_evaluator.weighted_recall(),
      ],
      eval_splits=['test'],
      eval_langs=['ar-x-levant'],
      domains=['speech'],
      task_subtypes=['classification'],
  )


class SVQArXMaghrebiSpeakerGenderClassification(SVQSpeakerGenderClassification):
  locale = 'ar_x_maghrebi'
  metadata = types.TaskMetadata(
      name='SVQArXMaghrebiSpeakerGenderClassification',
      description='Speaker-gender classification task.',
      reference='TODO',
      type='SpeakerGenderClassification',
      category='speech',
      main_score='Accuracy',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          classification_evaluator.accuracy(),
          classification_evaluator.balanced_accuracy(),
          classification_evaluator.weighted_f1(),
          classification_evaluator.weighted_precision(),
          classification_evaluator.weighted_recall(),
      ],
      eval_splits=['test'],
      eval_langs=['ar-x-maghrebi'],
      domains=['speech'],
      task_subtypes=['classification'],
  )


class SVQBnBdSpeakerGenderClassification(SVQSpeakerGenderClassification):
  locale = 'bn_bd'
  metadata = types.TaskMetadata(
      name='SVQBnBdSpeakerGenderClassification',
      description='Speaker-gender classification task.',
      reference='TODO',
      type='SpeakerGenderClassification',
      category='speech',
      main_score='Accuracy',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          classification_evaluator.accuracy(),
          classification_evaluator.balanced_accuracy(),
          classification_evaluator.weighted_f1(),
          classification_evaluator.weighted_precision(),
          classification_evaluator.weighted_recall(),
      ],
      eval_splits=['test'],
      eval_langs=['bn-BD'],
      domains=['speech'],
      task_subtypes=['classification'],
  )


class SVQBnInSpeakerGenderClassification(SVQSpeakerGenderClassification):
  locale = 'bn_in'
  metadata = types.TaskMetadata(
      name='SVQBnInSpeakerGenderClassification',
      description='Speaker-gender classification task.',
      reference='TODO',
      type='SpeakerGenderClassification',
      category='speech',
      main_score='Accuracy',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          classification_evaluator.accuracy(),
          classification_evaluator.balanced_accuracy(),
          classification_evaluator.weighted_f1(),
          classification_evaluator.weighted_precision(),
          classification_evaluator.weighted_recall(),
      ],
      eval_splits=['test'],
      eval_langs=['bn-IN'],
      domains=['speech'],
      task_subtypes=['classification'],
  )


class SVQEnAuSpeakerGenderClassification(SVQSpeakerGenderClassification):
  locale = 'en_au'
  metadata = types.TaskMetadata(
      name='SVQEnAuSpeakerGenderClassification',
      description='Speaker-gender classification task.',
      reference='TODO',
      type='SpeakerGenderClassification',
      category='speech',
      main_score='Accuracy',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          classification_evaluator.accuracy(),
          classification_evaluator.balanced_accuracy(),
          classification_evaluator.weighted_f1(),
          classification_evaluator.weighted_precision(),
          classification_evaluator.weighted_recall(),
      ],
      eval_splits=['test'],
      eval_langs=['en-AU'],
      domains=['speech'],
      task_subtypes=['classification'],
  )


class SVQEnGbSpeakerGenderClassification(SVQSpeakerGenderClassification):
  locale = 'en_gb'
  metadata = types.TaskMetadata(
      name='SVQEnGbSpeakerGenderClassification',
      description='Speaker-gender classification task.',
      reference='TODO',
      type='SpeakerGenderClassification',
      category='speech',
      main_score='Accuracy',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          classification_evaluator.accuracy(),
          classification_evaluator.balanced_accuracy(),
          classification_evaluator.weighted_f1(),
          classification_evaluator.weighted_precision(),
          classification_evaluator.weighted_recall(),
      ],
      eval_splits=['test'],
      eval_langs=['en-GB'],
      domains=['speech'],
      task_subtypes=['classification'],
  )


class SVQEnInSpeakerGenderClassification(SVQSpeakerGenderClassification):
  locale = 'en_in'
  metadata = types.TaskMetadata(
      name='SVQEnInSpeakerGenderClassification',
      description='Speaker-gender classification task.',
      reference='TODO',
      type='SpeakerGenderClassification',
      category='speech',
      main_score='Accuracy',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          classification_evaluator.accuracy(),
          classification_evaluator.balanced_accuracy(),
          classification_evaluator.weighted_f1(),
          classification_evaluator.weighted_precision(),
          classification_evaluator.weighted_recall(),
      ],
      eval_splits=['test'],
      eval_langs=['en-IN'],
      domains=['speech'],
      task_subtypes=['classification'],
  )


class SVQEnPhSpeakerGenderClassification(SVQSpeakerGenderClassification):
  locale = 'en_ph'
  metadata = types.TaskMetadata(
      name='SVQEnPhSpeakerGenderClassification',
      description='Speaker-gender classification task.',
      reference='TODO',
      type='SpeakerGenderClassification',
      category='speech',
      main_score='Accuracy',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          classification_evaluator.accuracy(),
          classification_evaluator.balanced_accuracy(),
          classification_evaluator.weighted_f1(),
          classification_evaluator.weighted_precision(),
          classification_evaluator.weighted_recall(),
      ],
      eval_splits=['test'],
      eval_langs=['en-PH'],
      domains=['speech'],
      task_subtypes=['classification'],
  )


class SVQEnUsSpeakerGenderClassification(SVQSpeakerGenderClassification):
  locale = 'en_us'
  metadata = types.TaskMetadata(
      name='SVQEnUsSpeakerGenderClassification',
      description='Speaker-gender classification task.',
      reference='TODO',
      type='SpeakerGenderClassification',
      category='speech',
      main_score='Accuracy',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          classification_evaluator.accuracy(),
          classification_evaluator.balanced_accuracy(),
          classification_evaluator.weighted_f1(),
          classification_evaluator.weighted_precision(),
          classification_evaluator.weighted_recall(),
      ],
      eval_splits=['test'],
      eval_langs=['en-US'],
      domains=['speech'],
      task_subtypes=['classification'],
  )


class SVQFiFiSpeakerGenderClassification(SVQSpeakerGenderClassification):
  locale = 'fi_fi'
  metadata = types.TaskMetadata(
      name='SVQFiFiSpeakerGenderClassification',
      description='Speaker-gender classification task.',
      reference='TODO',
      type='SpeakerGenderClassification',
      category='speech',
      main_score='Accuracy',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          classification_evaluator.accuracy(),
          classification_evaluator.balanced_accuracy(),
          classification_evaluator.weighted_f1(),
          classification_evaluator.weighted_precision(),
          classification_evaluator.weighted_recall(),
      ],
      eval_splits=['test'],
      eval_langs=['fi-FI'],
      domains=['speech'],
      task_subtypes=['classification'],
  )


class SVQGuInSpeakerGenderClassification(SVQSpeakerGenderClassification):
  locale = 'gu_in'
  metadata = types.TaskMetadata(
      name='SVQGuInSpeakerGenderClassification',
      description='Speaker-gender classification task.',
      reference='TODO',
      type='SpeakerGenderClassification',
      category='speech',
      main_score='Accuracy',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          classification_evaluator.accuracy(),
          classification_evaluator.balanced_accuracy(),
          classification_evaluator.weighted_f1(),
          classification_evaluator.weighted_precision(),
          classification_evaluator.weighted_recall(),
      ],
      eval_splits=['test'],
      eval_langs=['gu-IN'],
      domains=['speech'],
      task_subtypes=['classification'],
  )


class SVQHiInSpeakerGenderClassification(SVQSpeakerGenderClassification):
  locale = 'hi_in'
  metadata = types.TaskMetadata(
      name='SVQHiInSpeakerGenderClassification',
      description='Speaker-gender classification task.',
      reference='TODO',
      type='SpeakerGenderClassification',
      category='speech',
      main_score='Accuracy',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          classification_evaluator.accuracy(),
          classification_evaluator.balanced_accuracy(),
          classification_evaluator.weighted_f1(),
          classification_evaluator.weighted_precision(),
          classification_evaluator.weighted_recall(),
      ],
      eval_splits=['test'],
      eval_langs=['hi-IN'],
      domains=['speech'],
      task_subtypes=['classification'],
  )


class SVQIdIdSpeakerGenderClassification(SVQSpeakerGenderClassification):
  locale = 'id_id'
  metadata = types.TaskMetadata(
      name='SVQIdIdSpeakerGenderClassification',
      description='Speaker-gender classification task.',
      reference='TODO',
      type='SpeakerGenderClassification',
      category='speech',
      main_score='Accuracy',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          classification_evaluator.accuracy(),
          classification_evaluator.balanced_accuracy(),
          classification_evaluator.weighted_f1(),
          classification_evaluator.weighted_precision(),
          classification_evaluator.weighted_recall(),
      ],
      eval_splits=['test'],
      eval_langs=['id-ID'],
      domains=['speech'],
      task_subtypes=['classification'],
  )


class SVQJaJpSpeakerGenderClassification(SVQSpeakerGenderClassification):
  locale = 'ja_jp'
  metadata = types.TaskMetadata(
      name='SVQJaJpSpeakerGenderClassification',
      description='Speaker-gender classification task.',
      reference='TODO',
      type='SpeakerGenderClassification',
      category='speech',
      main_score='Accuracy',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          classification_evaluator.accuracy(),
          classification_evaluator.balanced_accuracy(),
          classification_evaluator.weighted_f1(),
          classification_evaluator.weighted_precision(),
          classification_evaluator.weighted_recall(),
      ],
      eval_splits=['test'],
      eval_langs=['ja-JP'],
      domains=['speech'],
      task_subtypes=['classification'],
  )


class SVQKnInSpeakerGenderClassification(SVQSpeakerGenderClassification):
  locale = 'kn_in'
  metadata = types.TaskMetadata(
      name='SVQKnInSpeakerGenderClassification',
      description='Speaker-gender classification task.',
      reference='TODO',
      type='SpeakerGenderClassification',
      category='speech',
      main_score='Accuracy',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          classification_evaluator.accuracy(),
          classification_evaluator.balanced_accuracy(),
          classification_evaluator.weighted_f1(),
          classification_evaluator.weighted_precision(),
          classification_evaluator.weighted_recall(),
      ],
      eval_splits=['test'],
      eval_langs=['kn-IN'],
      domains=['speech'],
      task_subtypes=['classification'],
  )


class SVQKoKrSpeakerGenderClassification(SVQSpeakerGenderClassification):
  locale = 'ko_kr'
  metadata = types.TaskMetadata(
      name='SVQKoKrSpeakerGenderClassification',
      description='Speaker-gender classification task.',
      reference='TODO',
      type='SpeakerGenderClassification',
      category='speech',
      main_score='Accuracy',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          classification_evaluator.accuracy(),
          classification_evaluator.balanced_accuracy(),
          classification_evaluator.weighted_f1(),
          classification_evaluator.weighted_precision(),
          classification_evaluator.weighted_recall(),
      ],
      eval_splits=['test'],
      eval_langs=['ko-KR'],
      domains=['speech'],
      task_subtypes=['classification'],
  )


class SVQMlInSpeakerGenderClassification(SVQSpeakerGenderClassification):
  locale = 'ml_in'
  metadata = types.TaskMetadata(
      name='SVQMlInSpeakerGenderClassification',
      description='Speaker-gender classification task.',
      reference='TODO',
      type='SpeakerGenderClassification',
      category='speech',
      main_score='Accuracy',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          classification_evaluator.accuracy(),
          classification_evaluator.balanced_accuracy(),
          classification_evaluator.weighted_f1(),
          classification_evaluator.weighted_precision(),
          classification_evaluator.weighted_recall(),
      ],
      eval_splits=['test'],
      eval_langs=['ml-IN'],
      domains=['speech'],
      task_subtypes=['classification'],
  )


class SVQMrInSpeakerGenderClassification(SVQSpeakerGenderClassification):
  locale = 'mr_in'
  metadata = types.TaskMetadata(
      name='SVQMrInSpeakerGenderClassification',
      description='Speaker-gender classification task.',
      reference='TODO',
      type='SpeakerGenderClassification',
      category='speech',
      main_score='Accuracy',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          classification_evaluator.accuracy(),
          classification_evaluator.balanced_accuracy(),
          classification_evaluator.weighted_f1(),
          classification_evaluator.weighted_precision(),
          classification_evaluator.weighted_recall(),
      ],
      eval_splits=['test'],
      eval_langs=['mr-IN'],
      domains=['speech'],
      task_subtypes=['classification'],
  )


class SVQRuRuSpeakerGenderClassification(SVQSpeakerGenderClassification):
  locale = 'ru_ru'
  metadata = types.TaskMetadata(
      name='SVQRuRuSpeakerGenderClassification',
      description='Speaker-gender classification task.',
      reference='TODO',
      type='SpeakerGenderClassification',
      category='speech',
      main_score='Accuracy',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          classification_evaluator.accuracy(),
          classification_evaluator.balanced_accuracy(),
          classification_evaluator.weighted_f1(),
          classification_evaluator.weighted_precision(),
          classification_evaluator.weighted_recall(),
      ],
      eval_splits=['test'],
      eval_langs=['ru-RU'],
      domains=['speech'],
      task_subtypes=['classification'],
  )


class SVQSwSpeakerGenderClassification(SVQSpeakerGenderClassification):
  locale = 'sw'
  metadata = types.TaskMetadata(
      name='SVQSwSpeakerGenderClassification',
      description='Speaker-gender classification task.',
      reference='TODO',
      type='SpeakerGenderClassification',
      category='speech',
      main_score='Accuracy',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          classification_evaluator.accuracy(),
          classification_evaluator.balanced_accuracy(),
          classification_evaluator.weighted_f1(),
          classification_evaluator.weighted_precision(),
          classification_evaluator.weighted_recall(),
      ],
      eval_splits=['test'],
      eval_langs=['sw'],
      domains=['speech'],
      task_subtypes=['classification'],
  )


class SVQTaInSpeakerGenderClassification(SVQSpeakerGenderClassification):
  locale = 'ta_in'
  metadata = types.TaskMetadata(
      name='SVQTaInSpeakerGenderClassification',
      description='Speaker-gender classification task.',
      reference='TODO',
      type='SpeakerGenderClassification',
      category='speech',
      main_score='Accuracy',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          classification_evaluator.accuracy(),
          classification_evaluator.balanced_accuracy(),
          classification_evaluator.weighted_f1(),
          classification_evaluator.weighted_precision(),
          classification_evaluator.weighted_recall(),
      ],
      eval_splits=['test'],
      eval_langs=['ta-IN'],
      domains=['speech'],
      task_subtypes=['classification'],
  )


class SVQTeInSpeakerGenderClassification(SVQSpeakerGenderClassification):
  locale = 'te_in'
  metadata = types.TaskMetadata(
      name='SVQTeInSpeakerGenderClassification',
      description='Speaker-gender classification task.',
      reference='TODO',
      type='SpeakerGenderClassification',
      category='speech',
      main_score='Accuracy',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          classification_evaluator.accuracy(),
          classification_evaluator.balanced_accuracy(),
          classification_evaluator.weighted_f1(),
          classification_evaluator.weighted_precision(),
          classification_evaluator.weighted_recall(),
      ],
      eval_splits=['test'],
      eval_langs=['te-IN'],
      domains=['speech'],
      task_subtypes=['classification'],
  )


class SVQUrInSpeakerGenderClassification(SVQSpeakerGenderClassification):
  locale = 'ur_in'
  metadata = types.TaskMetadata(
      name='SVQUrInSpeakerGenderClassification',
      description='Speaker-gender classification task.',
      reference='TODO',
      type='SpeakerGenderClassification',
      category='speech',
      main_score='Accuracy',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          classification_evaluator.accuracy(),
          classification_evaluator.balanced_accuracy(),
          classification_evaluator.weighted_f1(),
          classification_evaluator.weighted_precision(),
          classification_evaluator.weighted_recall(),
      ],
      eval_splits=['test'],
      eval_langs=['ur-IN'],
      domains=['speech'],
      task_subtypes=['classification'],
  )


class SVQUrPkSpeakerGenderClassification(SVQSpeakerGenderClassification):
  locale = 'ur_pk'
  metadata = types.TaskMetadata(
      name='SVQUrPkSpeakerGenderClassification',
      description='Speaker-gender classification task.',
      reference='TODO',
      type='SpeakerGenderClassification',
      category='speech',
      main_score='Accuracy',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          classification_evaluator.accuracy(),
          classification_evaluator.balanced_accuracy(),
          classification_evaluator.weighted_f1(),
          classification_evaluator.weighted_precision(),
          classification_evaluator.weighted_recall(),
      ],
      eval_splits=['test'],
      eval_langs=['ur-PK'],
      domains=['speech'],
      task_subtypes=['classification'],
  )
