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

from typing import Iterable, Type

import apache_beam as beam
from mseb import runner as runner_lib
from mseb import types
from mseb.datasets import simple_voice_questions as svq
from mseb.evaluators import clustering_evaluator
from mseb.tasks import clustering


class SVQClustering(clustering.ClusteringTask):
  """SVQ clustering."""

  _svq_dataset: svq.SimpleVoiceQuestionsDataset
  locale: str | None = None

  def _get_dataset(self) -> svq.SimpleVoiceQuestionsDataset:
    return svq.SimpleVoiceQuestionsDataset()

  def setup(
      self, runner_cls: Type[runner_lib.EncoderRunner] | None = None, **kwargs
  ):
    self._svq_dataset = self._get_dataset()

  def _task_data(self):
    df = self._svq_dataset.get_task_data('utt_index')
    if self.locale:
      df = df[df.locale == self.locale]
    return df

  @property
  def sub_tasks(self) -> list[str]:
    return ['speaker_gender', 'speaker_age', 'speaker_id']

  def sounds(self) -> Iterable[types.Sound]:
    for example in self._task_data().to_dict('records'):
      yield self._svq_dataset.get_sound(example)

  def sounds_beam(self):
    sounds = self._svq_dataset.get_task_sounds_beam('utt_index')
    if self.locale:
      sounds = sounds | f'FilterSoundsByLocale_{self.locale}' >> beam.Filter(
          lambda x: x.context.language == self.locale
      )
    return sounds

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


class SVQClusteringArEg(SVQClustering):
  locale = 'ar_eg'
  metadata = types.TaskMetadata(
      name='SVQClusteringArEg',
      description='Clustering task.',
      reference='TODO',
      type='Clustering',
      category='speech',
      main_score='VMeasure',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[clustering_evaluator.vmeasure_score()],
      eval_splits=['test'],
      eval_langs=['ar-EG'],
      domains=['speech'],
      task_subtypes=['clustering'],
  )


class SVQClusteringArXGulf(SVQClustering):
  locale = 'ar_x_gulf'
  metadata = types.TaskMetadata(
      name='SVQClusteringArXGulf',
      description='Clustering task.',
      reference='TODO',
      type='Clustering',
      category='speech',
      main_score='VMeasure',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[clustering_evaluator.vmeasure_score()],
      eval_splits=['test'],
      eval_langs=['ar-x-gulf'],
      domains=['speech'],
      task_subtypes=['clustering'],
  )


class SVQClusteringArXLevant(SVQClustering):
  locale = 'ar_x_levant'
  metadata = types.TaskMetadata(
      name='SVQClusteringArXLevant',
      description='Clustering task.',
      reference='TODO',
      type='Clustering',
      category='speech',
      main_score='VMeasure',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[clustering_evaluator.vmeasure_score()],
      eval_splits=['test'],
      eval_langs=['ar-x-levant'],
      domains=['speech'],
      task_subtypes=['clustering'],
  )


class SVQClusteringArXMaghrebi(SVQClustering):
  locale = 'ar_x_maghrebi'
  metadata = types.TaskMetadata(
      name='SVQClusteringArXMaghrebi',
      description='Clustering task.',
      reference='TODO',
      type='Clustering',
      category='speech',
      main_score='VMeasure',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[clustering_evaluator.vmeasure_score()],
      eval_splits=['test'],
      eval_langs=['ar-x-maghrebi'],
      domains=['speech'],
      task_subtypes=['clustering'],
  )


class SVQClusteringBnBd(SVQClustering):
  locale = 'bn_bd'
  metadata = types.TaskMetadata(
      name='SVQClusteringBnBd',
      description='Clustering task.',
      reference='TODO',
      type='Clustering',
      category='speech',
      main_score='VMeasure',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[clustering_evaluator.vmeasure_score()],
      eval_splits=['test'],
      eval_langs=['bn-BD'],
      domains=['speech'],
      task_subtypes=['clustering'],
  )


class SVQClusteringBnIn(SVQClustering):
  locale = 'bn_in'
  metadata = types.TaskMetadata(
      name='SVQClusteringBnIn',
      description='Clustering task.',
      reference='TODO',
      type='Clustering',
      category='speech',
      main_score='VMeasure',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[clustering_evaluator.vmeasure_score()],
      eval_splits=['test'],
      eval_langs=['bn-IN'],
      domains=['speech'],
      task_subtypes=['clustering'],
  )


class SVQClusteringEnAu(SVQClustering):
  locale = 'en_au'
  metadata = types.TaskMetadata(
      name='SVQClusteringEnAu',
      description='Clustering task.',
      reference='TODO',
      type='Clustering',
      category='speech',
      main_score='VMeasure',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[clustering_evaluator.vmeasure_score()],
      eval_splits=['test'],
      eval_langs=['en-AU'],
      domains=['speech'],
      task_subtypes=['clustering'],
  )


class SVQClusteringEnGb(SVQClustering):
  locale = 'en_gb'
  metadata = types.TaskMetadata(
      name='SVQClusteringEnGb',
      description='Clustering task.',
      reference='TODO',
      type='Clustering',
      category='speech',
      main_score='VMeasure',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[clustering_evaluator.vmeasure_score()],
      eval_splits=['test'],
      eval_langs=['en-GB'],
      domains=['speech'],
      task_subtypes=['clustering'],
  )


class SVQClusteringEnIn(SVQClustering):
  locale = 'en_in'
  metadata = types.TaskMetadata(
      name='SVQClusteringEnIn',
      description='Clustering task.',
      reference='TODO',
      type='Clustering',
      category='speech',
      main_score='VMeasure',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[clustering_evaluator.vmeasure_score()],
      eval_splits=['test'],
      eval_langs=['en-IN'],
      domains=['speech'],
      task_subtypes=['clustering'],
  )


class SVQClusteringEnPh(SVQClustering):
  locale = 'en_ph'
  metadata = types.TaskMetadata(
      name='SVQClusteringEnPh',
      description='Clustering task.',
      reference='TODO',
      type='Clustering',
      category='speech',
      main_score='VMeasure',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[clustering_evaluator.vmeasure_score()],
      eval_splits=['test'],
      eval_langs=['en-PH'],
      domains=['speech'],
      task_subtypes=['clustering'],
  )


class SVQClusteringEnUs(SVQClustering):
  locale = 'en_us'
  metadata = types.TaskMetadata(
      name='SVQClusteringEnUs',
      description='Clustering task.',
      reference='TODO',
      type='Clustering',
      category='speech',
      main_score='VMeasure',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[clustering_evaluator.vmeasure_score()],
      eval_splits=['test'],
      eval_langs=['en-US'],
      domains=['speech'],
      task_subtypes=['clustering'],
  )


class SVQClusteringFiFi(SVQClustering):
  locale = 'fi_fi'
  metadata = types.TaskMetadata(
      name='SVQClusteringFiFi',
      description='Clustering task.',
      reference='TODO',
      type='Clustering',
      category='speech',
      main_score='VMeasure',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[clustering_evaluator.vmeasure_score()],
      eval_splits=['test'],
      eval_langs=['fi-FI'],
      domains=['speech'],
      task_subtypes=['clustering'],
  )


class SVQClusteringGuIn(SVQClustering):
  locale = 'gu_in'
  metadata = types.TaskMetadata(
      name='SVQClusteringGuIn',
      description='Clustering task.',
      reference='TODO',
      type='Clustering',
      category='speech',
      main_score='VMeasure',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[clustering_evaluator.vmeasure_score()],
      eval_splits=['test'],
      eval_langs=['gu-IN'],
      domains=['speech'],
      task_subtypes=['clustering'],
  )


class SVQClusteringHiIn(SVQClustering):
  locale = 'hi_in'
  metadata = types.TaskMetadata(
      name='SVQClusteringHiIn',
      description='Clustering task.',
      reference='TODO',
      type='Clustering',
      category='speech',
      main_score='VMeasure',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[clustering_evaluator.vmeasure_score()],
      eval_splits=['test'],
      eval_langs=['hi-IN'],
      domains=['speech'],
      task_subtypes=['clustering'],
  )


class SVQClusteringIdId(SVQClustering):
  locale = 'id_id'
  metadata = types.TaskMetadata(
      name='SVQClusteringIdId',
      description='Clustering task.',
      reference='TODO',
      type='Clustering',
      category='speech',
      main_score='VMeasure',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[clustering_evaluator.vmeasure_score()],
      eval_splits=['test'],
      eval_langs=['id-ID'],
      domains=['speech'],
      task_subtypes=['clustering'],
  )


class SVQClusteringJaJp(SVQClustering):
  locale = 'ja_jp'
  metadata = types.TaskMetadata(
      name='SVQClusteringJaJp',
      description='Clustering task.',
      reference='TODO',
      type='Clustering',
      category='speech',
      main_score='VMeasure',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[clustering_evaluator.vmeasure_score()],
      eval_splits=['test'],
      eval_langs=['ja-JP'],
      domains=['speech'],
      task_subtypes=['clustering'],
  )


class SVQClusteringKnIn(SVQClustering):
  locale = 'kn_in'
  metadata = types.TaskMetadata(
      name='SVQClusteringKnIn',
      description='Clustering task.',
      reference='TODO',
      type='Clustering',
      category='speech',
      main_score='VMeasure',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[clustering_evaluator.vmeasure_score()],
      eval_splits=['test'],
      eval_langs=['kn-IN'],
      domains=['speech'],
      task_subtypes=['clustering'],
  )


class SVQClusteringKoKr(SVQClustering):
  locale = 'ko_kr'
  metadata = types.TaskMetadata(
      name='SVQClusteringKoKr',
      description='Clustering task.',
      reference='TODO',
      type='Clustering',
      category='speech',
      main_score='VMeasure',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[clustering_evaluator.vmeasure_score()],
      eval_splits=['test'],
      eval_langs=['ko-KR'],
      domains=['speech'],
      task_subtypes=['clustering'],
  )


class SVQClusteringMlIn(SVQClustering):
  locale = 'ml_in'
  metadata = types.TaskMetadata(
      name='SVQClusteringMlIn',
      description='Clustering task.',
      reference='TODO',
      type='Clustering',
      category='speech',
      main_score='VMeasure',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[clustering_evaluator.vmeasure_score()],
      eval_splits=['test'],
      eval_langs=['ml-IN'],
      domains=['speech'],
      task_subtypes=['clustering'],
  )


class SVQClusteringMrIn(SVQClustering):
  locale = 'mr_in'
  metadata = types.TaskMetadata(
      name='SVQClusteringMrIn',
      description='Clustering task.',
      reference='TODO',
      type='Clustering',
      category='speech',
      main_score='VMeasure',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[clustering_evaluator.vmeasure_score()],
      eval_splits=['test'],
      eval_langs=['mr-IN'],
      domains=['speech'],
      task_subtypes=['clustering'],
  )


class SVQClusteringRuRu(SVQClustering):
  locale = 'ru_ru'
  metadata = types.TaskMetadata(
      name='SVQClusteringRuRu',
      description='Clustering task.',
      reference='TODO',
      type='Clustering',
      category='speech',
      main_score='VMeasure',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[clustering_evaluator.vmeasure_score()],
      eval_splits=['test'],
      eval_langs=['ru-RU'],
      domains=['speech'],
      task_subtypes=['clustering'],
  )


class SVQClusteringSw(SVQClustering):
  locale = 'sw'
  metadata = types.TaskMetadata(
      name='SVQClusteringSw',
      description='Clustering task.',
      reference='TODO',
      type='Clustering',
      category='speech',
      main_score='VMeasure',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[clustering_evaluator.vmeasure_score()],
      eval_splits=['test'],
      eval_langs=['sw'],
      domains=['speech'],
      task_subtypes=['clustering'],
  )


class SVQClusteringTaIn(SVQClustering):
  locale = 'ta_in'
  metadata = types.TaskMetadata(
      name='SVQClusteringTaIn',
      description='Clustering task.',
      reference='TODO',
      type='Clustering',
      category='speech',
      main_score='VMeasure',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[clustering_evaluator.vmeasure_score()],
      eval_splits=['test'],
      eval_langs=['ta-IN'],
      domains=['speech'],
      task_subtypes=['clustering'],
  )


class SVQClusteringTeIn(SVQClustering):
  locale = 'te_in'
  metadata = types.TaskMetadata(
      name='SVQClusteringTeIn',
      description='Clustering task.',
      reference='TODO',
      type='Clustering',
      category='speech',
      main_score='VMeasure',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[clustering_evaluator.vmeasure_score()],
      eval_splits=['test'],
      eval_langs=['te-IN'],
      domains=['speech'],
      task_subtypes=['clustering'],
  )


class SVQClusteringUrIn(SVQClustering):
  locale = 'ur_in'
  metadata = types.TaskMetadata(
      name='SVQClusteringUrIn',
      description='Clustering task.',
      reference='TODO',
      type='Clustering',
      category='speech',
      main_score='VMeasure',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[clustering_evaluator.vmeasure_score()],
      eval_splits=['test'],
      eval_langs=['ur-IN'],
      domains=['speech'],
      task_subtypes=['clustering'],
  )


class SVQClusteringUrPk(SVQClustering):
  locale = 'ur_pk'
  metadata = types.TaskMetadata(
      name='SVQClusteringUrPk',
      description='Clustering task.',
      reference='TODO',
      type='Clustering',
      category='speech',
      main_score='VMeasure',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[clustering_evaluator.vmeasure_score()],
      eval_splits=['test'],
      eval_langs=['ur-PK'],
      domains=['speech'],
      task_subtypes=['clustering'],
  )
