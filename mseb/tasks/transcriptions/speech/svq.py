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

"""SVQ speech transcription tasks."""

from typing import Iterable, Sequence

from mseb import task as task_lib
from mseb import types
from mseb.datasets import simple_voice_questions as svq
from mseb.evaluators import transcription_evaluator
from mseb.tasks import transcription


_filter_fn_by_sub_task = {
    'speech_transcription': lambda x: True,
    'speech_transcription:clean': lambda x: x['environment'] == 'clean',
    'speech_transcription:media_noise': (
        lambda x: x['environment'] == 'media_noise'
    ),
    'speech_transcription:traffic_noise': (
        lambda x: x['environment'] == 'traffic_noise'
    ),
    'speech_transcription:background_speech': (
        lambda x: x['environment'] == 'background_speech'
    ),
}


def _base_sub_task(sub_task: str) -> str:
  return sub_task.split(':')[0]


class SVQSpeechTranscription(transcription.TranscriptionTask):
  """SVQ speech transcription task."""

  locale: str | None = None

  def _get_dataset(self) -> svq.SimpleVoiceQuestionsDataset:
    return svq.SimpleVoiceQuestionsDataset()

  @property
  def sub_tasks(self) -> list[str]:
    return list(_filter_fn_by_sub_task.keys())

  def sounds(self) -> Iterable[types.Sound]:
    svq_dataset = self._get_dataset()
    for example in svq_dataset.get_task_data(
        'speech_transcription',
        dtype={
            'locale': str,
            'utt_id': str,
            task_lib.TRANSCRIPT_KEY.value: str,
            transcription.CONTEXTUAL_BIAS_KEY.value: str,
        },
    ).to_dict('records'):
      if example['locale'] == self.locale:
        sound = svq_dataset.get_sound(example)
        sound.context.text = example[task_lib.TRANSCRIPT_KEY.value]
        if transcription.CONTEXTUAL_BIAS_KEY.value:
          sound = types.SoundWithTitleAndContext(
              waveform=sound.waveform,
              context=sound.context,
              context_text=example.get(transcription.CONTEXTUAL_BIAS_KEY.value),
          )
        yield sound

  def examples(
      self, sub_task: str
  ) -> Iterable[transcription_evaluator.TranscriptTruth]:
    filter_fn = _filter_fn_by_sub_task[sub_task]
    svq_dataset = self._get_dataset()
    for example in svq_dataset.get_task_data(
        _base_sub_task(sub_task),
        dtype={
            'locale': str,
            'utt_id': str,
            'transcript_truth': Sequence[str],
        },
    ).to_dict('records'):
      if example['locale'] == self.locale and filter_fn(example):
        yield transcription_evaluator.TranscriptTruth(
            sound_id=example['utt_id'],
            text=example['transcript_truth'],
            language=example['locale'],
        )


class SVQArEgSpeechTranscription(SVQSpeechTranscription):
  locale = 'ar_eg'
  metadata = types.TaskMetadata(
      name='SVQArEgSpeechTranscription',
      description='Speech transcription task.',
      reference='TODO',
      type='SpeechTranscription',
      category='speech',
      main_score='WER',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[transcription_evaluator.wer(), transcription_evaluator.ser()],
      eval_splits=['test'],
      eval_langs=['ar-EG'],
      domains=['speech'],
      task_subtypes=['transcription'],
  )


class SVQArXGulfSpeechTranscription(SVQSpeechTranscription):
  locale = 'ar_x_gulf'
  metadata = types.TaskMetadata(
      name='SVQArXGulfSpeechTranscription',
      description='Speech transcription task.',
      reference='TODO',
      type='SpeechTranscription',
      category='speech',
      main_score='WER',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[transcription_evaluator.wer(), transcription_evaluator.ser()],
      eval_splits=['test'],
      eval_langs=['ar-x-gulf'],
      domains=['speech'],
      task_subtypes=['transcription'],
  )


class SVQArXLevantSpeechTranscription(SVQSpeechTranscription):
  locale = 'ar_x_levant'
  metadata = types.TaskMetadata(
      name='SVQArXLevantSpeechTranscription',
      description='Speech transcription task.',
      reference='TODO',
      type='SpeechTranscription',
      category='speech',
      main_score='WER',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[transcription_evaluator.wer(), transcription_evaluator.ser()],
      eval_splits=['test'],
      eval_langs=['ar-x-levant'],
      domains=['speech'],
      task_subtypes=['transcription'],
  )


class SVQArXMaghrebiSpeechTranscription(SVQSpeechTranscription):
  locale = 'ar_x_maghrebi'
  metadata = types.TaskMetadata(
      name='SVQArXMaghrebiSpeechTranscription',
      description='Speech transcription task.',
      reference='TODO',
      type='SpeechTranscription',
      category='speech',
      main_score='WER',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[transcription_evaluator.wer(), transcription_evaluator.ser()],
      eval_splits=['test'],
      eval_langs=['ar-x-maghrebi'],
      domains=['speech'],
      task_subtypes=['transcription'],
  )


class SVQBnBdSpeechTranscription(SVQSpeechTranscription):
  locale = 'bn_bd'
  metadata = types.TaskMetadata(
      name='SVQBnBdSpeechTranscription',
      description='Speech transcription task.',
      reference='TODO',
      type='SpeechTranscription',
      category='speech',
      main_score='WER',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[transcription_evaluator.wer(), transcription_evaluator.ser()],
      eval_splits=['test'],
      eval_langs=['bn-BD'],
      domains=['speech'],
      task_subtypes=['transcription'],
  )


class SVQBnInSpeechTranscription(SVQSpeechTranscription):
  locale = 'bn_in'
  metadata = types.TaskMetadata(
      name='SVQBnInSpeechTranscription',
      description='Speech transcription task.',
      reference='TODO',
      type='SpeechTranscription',
      category='speech',
      main_score='WER',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[transcription_evaluator.wer(), transcription_evaluator.ser()],
      eval_splits=['test'],
      eval_langs=['bn-IN'],
      domains=['speech'],
      task_subtypes=['transcription'],
  )


class SVQEnAuSpeechTranscription(SVQSpeechTranscription):
  locale = 'en_au'
  metadata = types.TaskMetadata(
      name='SVQEnAuSpeechTranscription',
      description='Speech transcription task.',
      reference='TODO',
      type='SpeechTranscription',
      category='speech',
      main_score='WER',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[transcription_evaluator.wer(), transcription_evaluator.ser()],
      eval_splits=['test'],
      eval_langs=['en-AU'],
      domains=['speech'],
      task_subtypes=['transcription'],
  )


class SVQEnGbSpeechTranscription(SVQSpeechTranscription):
  locale = 'en_gb'
  metadata = types.TaskMetadata(
      name='SVQEnGbSpeechTranscription',
      description='Speech transcription task.',
      reference='TODO',
      type='SpeechTranscription',
      category='speech',
      main_score='WER',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[transcription_evaluator.wer(), transcription_evaluator.ser()],
      eval_splits=['test'],
      eval_langs=['en-GB'],
      domains=['speech'],
      task_subtypes=['transcription'],
  )


class SVQEnInSpeechTranscription(SVQSpeechTranscription):
  locale = 'en_in'
  metadata = types.TaskMetadata(
      name='SVQEnInSpeechTranscription',
      description='Speech transcription task.',
      reference='TODO',
      type='SpeechTranscription',
      category='speech',
      main_score='WER',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[transcription_evaluator.wer(), transcription_evaluator.ser()],
      eval_splits=['test'],
      eval_langs=['en-IN'],
      domains=['speech'],
      task_subtypes=['transcription'],
  )


class SVQEnPhSpeechTranscription(SVQSpeechTranscription):
  locale = 'en_ph'
  metadata = types.TaskMetadata(
      name='SVQEnPhSpeechTranscription',
      description='Speech transcription task.',
      reference='TODO',
      type='SpeechTranscription',
      category='speech',
      main_score='WER',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[transcription_evaluator.wer(), transcription_evaluator.ser()],
      eval_splits=['test'],
      eval_langs=['en-PH'],
      domains=['speech'],
      task_subtypes=['transcription'],
  )


class SVQEnUsSpeechTranscription(SVQSpeechTranscription):
  locale = 'en_us'
  metadata = types.TaskMetadata(
      name='SVQEnUsSpeechTranscription',
      description='Speech transcription task.',
      reference='TODO',
      type='SpeechTranscription',
      category='speech',
      main_score='WER',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[transcription_evaluator.wer(), transcription_evaluator.ser()],
      eval_splits=['test'],
      eval_langs=['en-US'],
      domains=['speech'],
      task_subtypes=['transcription'],
  )


class SVQFiFiSpeechTranscription(SVQSpeechTranscription):
  locale = 'fi_fi'
  metadata = types.TaskMetadata(
      name='SVQFiFiSpeechTranscription',
      description='Speech transcription task.',
      reference='TODO',
      type='SpeechTranscription',
      category='speech',
      main_score='WER',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[transcription_evaluator.wer(), transcription_evaluator.ser()],
      eval_splits=['test'],
      eval_langs=['fi-FI'],
      domains=['speech'],
      task_subtypes=['transcription'],
  )


class SVQGuInSpeechTranscription(SVQSpeechTranscription):
  locale = 'gu_in'
  metadata = types.TaskMetadata(
      name='SVQGuInSpeechTranscription',
      description='Speech transcription task.',
      reference='TODO',
      type='SpeechTranscription',
      category='speech',
      main_score='WER',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[transcription_evaluator.wer(), transcription_evaluator.ser()],
      eval_splits=['test'],
      eval_langs=['gu-IN'],
      domains=['speech'],
      task_subtypes=['transcription'],
  )


class SVQHiInSpeechTranscription(SVQSpeechTranscription):
  locale = 'hi_in'
  metadata = types.TaskMetadata(
      name='SVQHiInSpeechTranscription',
      description='Speech transcription task.',
      reference='TODO',
      type='SpeechTranscription',
      category='speech',
      main_score='WER',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[transcription_evaluator.wer(), transcription_evaluator.ser()],
      eval_splits=['test'],
      eval_langs=['hi-IN'],
      domains=['speech'],
      task_subtypes=['transcription'],
  )


class SVQIdIdSpeechTranscription(SVQSpeechTranscription):
  locale = 'id_id'
  metadata = types.TaskMetadata(
      name='SVQIdIdSpeechTranscription',
      description='Speech transcription task.',
      reference='TODO',
      type='SpeechTranscription',
      category='speech',
      main_score='WER',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[transcription_evaluator.wer(), transcription_evaluator.ser()],
      eval_splits=['test'],
      eval_langs=['id-ID'],
      domains=['speech'],
      task_subtypes=['transcription'],
  )


class SVQJaJpSpeechTranscription(SVQSpeechTranscription):
  locale = 'ja_jp'
  metadata = types.TaskMetadata(
      name='SVQJaJpSpeechTranscription',
      description='Speech transcription task.',
      reference='TODO',
      type='SpeechTranscription',
      category='speech',
      main_score='WER',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[transcription_evaluator.wer(), transcription_evaluator.ser()],
      eval_splits=['test'],
      eval_langs=['ja-JP'],
      domains=['speech'],
      task_subtypes=['transcription'],
  )


class SVQKnInSpeechTranscription(SVQSpeechTranscription):
  locale = 'kn_in'
  metadata = types.TaskMetadata(
      name='SVQKnInSpeechTranscription',
      description='Speech transcription task.',
      reference='TODO',
      type='SpeechTranscription',
      category='speech',
      main_score='WER',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[transcription_evaluator.wer(), transcription_evaluator.ser()],
      eval_splits=['test'],
      eval_langs=['kn-IN'],
      domains=['speech'],
      task_subtypes=['transcription'],
  )


class SVQKoKrSpeechTranscription(SVQSpeechTranscription):
  locale = 'ko_kr'
  metadata = types.TaskMetadata(
      name='SVQKoKrSpeechTranscription',
      description='Speech transcription task.',
      reference='TODO',
      type='SpeechTranscription',
      category='speech',
      main_score='WER',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[transcription_evaluator.wer(), transcription_evaluator.ser()],
      eval_splits=['test'],
      eval_langs=['ko-KR'],
      domains=['speech'],
      task_subtypes=['transcription'],
  )


class SVQMlInSpeechTranscription(SVQSpeechTranscription):
  locale = 'ml_in'
  metadata = types.TaskMetadata(
      name='SVQMlInSpeechTranscription',
      description='Speech transcription task.',
      reference='TODO',
      type='SpeechTranscription',
      category='speech',
      main_score='WER',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[transcription_evaluator.wer(), transcription_evaluator.ser()],
      eval_splits=['test'],
      eval_langs=['ml-IN'],
      domains=['speech'],
      task_subtypes=['transcription'],
  )


class SVQMrInSpeechTranscription(SVQSpeechTranscription):
  locale = 'mr_in'
  metadata = types.TaskMetadata(
      name='SVQMrInSpeechTranscription',
      description='Speech transcription task.',
      reference='TODO',
      type='SpeechTranscription',
      category='speech',
      main_score='WER',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[transcription_evaluator.wer(), transcription_evaluator.ser()],
      eval_splits=['test'],
      eval_langs=['mr-IN'],
      domains=['speech'],
      task_subtypes=['transcription'],
  )


class SVQRuRuSpeechTranscription(SVQSpeechTranscription):
  locale = 'ru_ru'
  metadata = types.TaskMetadata(
      name='SVQRuRuSpeechTranscription',
      description='Speech transcription task.',
      reference='TODO',
      type='SpeechTranscription',
      category='speech',
      main_score='WER',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[transcription_evaluator.wer(), transcription_evaluator.ser()],
      eval_splits=['test'],
      eval_langs=['ru-RU'],
      domains=['speech'],
      task_subtypes=['transcription'],
  )


class SVQSwSpeechTranscription(SVQSpeechTranscription):
  locale = 'sw'
  metadata = types.TaskMetadata(
      name='SVQSwSpeechTranscription',
      description='Speech transcription task.',
      reference='TODO',
      type='SpeechTranscription',
      category='speech',
      main_score='WER',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[transcription_evaluator.wer(), transcription_evaluator.ser()],
      eval_splits=['test'],
      eval_langs=['sw'],
      domains=['speech'],
      task_subtypes=['transcription'],
  )


class SVQTaInSpeechTranscription(SVQSpeechTranscription):
  locale = 'ta_in'
  metadata = types.TaskMetadata(
      name='SVQTaInSpeechTranscription',
      description='Speech transcription task.',
      reference='TODO',
      type='SpeechTranscription',
      category='speech',
      main_score='WER',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[transcription_evaluator.wer(), transcription_evaluator.ser()],
      eval_splits=['test'],
      eval_langs=['ta-IN'],
      domains=['speech'],
      task_subtypes=['transcription'],
  )


class SVQTeInSpeechTranscription(SVQSpeechTranscription):
  locale = 'te_in'
  metadata = types.TaskMetadata(
      name='SVQTeInSpeechTranscription',
      description='Speech transcription task.',
      reference='TODO',
      type='SpeechTranscription',
      category='speech',
      main_score='WER',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[transcription_evaluator.wer(), transcription_evaluator.ser()],
      eval_splits=['test'],
      eval_langs=['te-IN'],
      domains=['speech'],
      task_subtypes=['transcription'],
  )


class SVQUrInSpeechTranscription(SVQSpeechTranscription):
  locale = 'ur_in'
  metadata = types.TaskMetadata(
      name='SVQUrInSpeechTranscription',
      description='Speech transcription task.',
      reference='TODO',
      type='SpeechTranscription',
      category='speech',
      main_score='WER',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[transcription_evaluator.wer(), transcription_evaluator.ser()],
      eval_splits=['test'],
      eval_langs=['ur-IN'],
      domains=['speech'],
      task_subtypes=['transcription'],
  )


class SVQUrPkSpeechTranscription(SVQSpeechTranscription):
  locale = 'ur_pk'
  metadata = types.TaskMetadata(
      name='SVQUrPkSpeechTranscription',
      description='Speech transcription task.',
      reference='TODO',
      type='SpeechTranscription',
      category='speech',
      main_score='WER',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[transcription_evaluator.wer(), transcription_evaluator.ser()],
      eval_splits=['test'],
      eval_langs=['ur-PK'],
      domains=['speech'],
      task_subtypes=['transcription'],
  )
