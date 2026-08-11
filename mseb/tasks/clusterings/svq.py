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

"""SVQ clustering tasks."""

import functools
from typing import Iterable

from mseb import types
from mseb.datasets import simple_voice_questions as svq
from mseb.evaluators import clustering_evaluator
from mseb.tasks import clustering


class SVQClustering(clustering.ClusteringTask):
  """SVQ clustering."""

  locale: str | None = None

  @functools.cached_property
  def svq_dataset(self) -> svq.SimpleVoiceQuestionsDataset:
    return svq.SimpleVoiceQuestionsDataset()

  def _task_data(self):
    df = self.svq_dataset.get_task_data('utt_index')
    if self.locale:
      df = df[df.locale == self.locale]
    return df

  @property
  def sub_tasks(self) -> list[str]:
    return ['speaker_gender', 'speaker_age', 'speaker_id']

  def multimodal_inputs(self) -> Iterable[types.Sound]:
    for example in self._task_data().to_dict('records'):
      yield self.svq_dataset.get_sound(example)

  def multimodal_inputs_beam(self):
    return self.svq_dataset.get_task_sounds_beam(
        'utt_index', locale=self.locale
    )

  def examples(
      self, sub_task: str
  ) -> Iterable[clustering_evaluator.ClusteringExample]:
    """Get (utt_id, label) examples from svq dataset."""
    for example in self._task_data().to_dict('records'):
      yield clustering_evaluator.ClusteringExample(
          example['utt_id'], example[sub_task]
      )


class SVQClusteringAll(SVQClustering):
  locale = None


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
              documentation_file='svq_clustering.md',
              dataset_documentation_file='dataset_svq.md',
              type='Clustering',
              category='speech',
              main_score='VMeasure',
              revision='1.0.0',
              dataset=types.Dataset(
                  name='SVQ',
                  path='https://huggingface.co/datasets/google/svq',
                  revision='1.0.0',
              ),
              scores=[clustering_evaluator.vmeasure_score()],
              eval_splits=['test'],
              eval_langs=[eval_lang],
              domains=['speech'],
              task_subtypes=['clustering'],
          ),
      },
  )
  return cls


# Generate all locale-specific classes and register them in the module.
# Default size.
for _locale, (_suffix, _eval_lang) in _SVQ_LOCALES.items():
  _cls = _make_task_class(  # pylint: disable=invalid-name
      base_cls=SVQClustering,
      locale=_locale,
      suffix=_suffix,
      eval_lang=_eval_lang,
      description='Clustering task.',
  )
  globals()[_cls.__name__] = _cls
