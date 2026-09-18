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

"""SVQ Speaker-gender classification tasks."""

import functools
import os
from typing import Any, Iterable

from mseb import types
from mseb.datasets import simple_voice_questions as svq
from mseb.evaluators import classification_evaluator
from mseb.tasks import classification


def _get_environment(sub_task: str) -> str:
  """Returns the environment for the given sub_task.

  Examples:
    'speaker_gender_classification:clean' -> 'clean'
    'speaker_gender_classification' -> '*'

  Args:
    sub_task: The sub_task name.

  Returns:
    The environment for the given sub_task.
  """
  sub_task_parts = sub_task.split(':')
  if len(sub_task_parts) == 1:
    return '*'
  elif len(sub_task_parts) == 2:
    return sub_task_parts[1]
  else:
    raise ValueError(f'Invalid sub_task: {sub_task}')


class SVQSpeakerGenderClassification(classification.ClassificationTask):
  """SVQ speaker-gender classification task."""

  locale: str | None = None

  @functools.cached_property
  def svq_dataset(self) -> svq.SimpleVoiceQuestionsDataset:
    return svq.SimpleVoiceQuestionsDataset()

  def _task_data(self, task_data_key: str, dtype: dict[str, Any] | None = None):
    ds = self.svq_dataset.get_task_data(task_data_key, dtype=dtype)
    ds = ds[ds['classifications/speaker_gender']]
    return ds

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
    return [
        'speaker_gender_classification',
        'speaker_gender_classification:clean',
        'speaker_gender_classification:media_noise',
        'speaker_gender_classification:traffic_noise',
        'speaker_gender_classification:background_speech',
    ]

  def multimodal_inputs(self) -> Iterable[types.Sound]:
    if self.locale is None:
      raise ValueError('`locale` must be set by a concrete task subclass.')

    df = self._task_data(
        f'utts_{self.locale}_*',
        dtype={'locale': str, 'utt_id': str},
    )
    for example in df.to_dict('records'):
      yield self.svq_dataset.get_sound(example)

  def examples(
      self, sub_task: str
  ) -> Iterable[classification_evaluator.ClassificationReference]:
    if self.locale is None:
      raise ValueError('`locale` must be set by a concrete task subclass.')

    class_labels = set(self.class_labels())
    df = self._task_data(
        f'utts_{self.locale}_{_get_environment(sub_task)}',
        dtype={
            'locale': str,
            'utt_id': str,
            'speaker_gender': str,
        },
    )
    for example in df.to_dict('records'):
      gender = example.get('speaker_gender')
      if gender.capitalize() in class_labels:
        yield classification_evaluator.ClassificationReference(
            example_id=example['utt_id'],
            label_id=gender.capitalize(),
        )

  def class_labels(self) -> Iterable[str]:
    return (
        'Female',
        'Male',
    )


# Locale -> (ClassName suffix, eval_lang)
_SVQ_LOCALES = {
    'ar_eg': ('ArEg', 'ar-EG'),
    'ar_x_gulf': ('ArXGulf', 'ar-x-gulf'),
    'ar_x_levant': ('ArXLevant', 'ar-x-levant'),
    'ar_x_maghrebi': ('ArXMaghrebi', 'ar-x-maghrebi'),
    'bn_bd': ('BnBd', 'bn-BD'),
    'bn_in': ('BnIn', 'bn-IN'),
    'en_au': ('EnAu', 'en-AU'),
    'en_gb': ('EnGb', 'en-GB'),
    'en_in': ('EnIn', 'en-IN'),
    'en_ph': ('EnPh', 'en-PH'),
    'en_us': ('EnUs', 'en-US'),
    'fi_fi': ('FiFi', 'fi-FI'),
    'gu_in': ('GuIn', 'gu-IN'),
    'hi_in': ('HiIn', 'hi-IN'),
    'id_id': ('IdId', 'id-ID'),
    'ja_jp': ('JaJp', 'ja-JP'),
    'kn_in': ('KnIn', 'kn-IN'),
    'ko_kr': ('KoKr', 'ko-KR'),
    'ml_in': ('MlIn', 'ml-IN'),
    'mr_in': ('MrIn', 'mr-IN'),
    'ru_ru': ('RuRu', 'ru-RU'),
    'sw': ('Sw', 'sw'),
    'ta_in': ('TaIn', 'ta-IN'),
    'te_in': ('TeIn', 'te-IN'),
    'ur_in': ('UrIn', 'ur-IN'),
    'ur_pk': ('UrPk', 'ur-PK'),
}


def _make_task_class(base_cls, locale, suffix, eval_lang, description):
  """Dynamically create a locale-specific task class."""
  class_name = f'SVQ{suffix}{base_cls.__name__[len("SVQ"):]}'
  cls = type(
      class_name,
      (base_cls,),
      {
          'locale': locale,
          'metadata': types.TaskMetadata(
              name=class_name,
              description=description,
              reference='https://huggingface.co/datasets/google/svq',
              documentation_file='svq_classification.md',
              dataset_documentation_file='dataset_svq.md',
              type='SpeakerGenderClassification',
              category='speech',
              main_score='Accuracy',
              revision='1.0.0',
              dataset=types.Dataset(
                  name='SVQ',
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
              eval_langs=[eval_lang],
              domains=['speech'],
              task_subtypes=['classification'],
          ),
      },
  )
  return cls


# Generate all locale-specific classes and register them in the module.
for _locale, (_suffix, _eval_lang) in _SVQ_LOCALES.items():
  _cls = _make_task_class(  # pylint: disable=invalid-name
      base_cls=SVQSpeakerGenderClassification,
      locale=_locale,
      suffix=_suffix,
      eval_lang=_eval_lang,
      description='Speaker-gender classification task.',
  )
  globals()[_cls.__name__] = _cls
