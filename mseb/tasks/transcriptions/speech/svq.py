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

import functools
from typing import Any, Iterable

from mseb import task as task_lib
from mseb import types
from mseb.datasets import simple_voice_questions as svq
from mseb.evaluators import transcription_evaluator
from mseb.tasks import transcription


def _get_environment(sub_task: str) -> str:
  """Returns the environment for the given sub_task.

  Examples:
    'speech_transcription:clean' -> 'clean'
    'speech_transcription' -> '*'

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


class SVQSpeechTranscription(transcription.TranscriptionTask):
  """SVQ speech transcription task."""

  locale: str | None = None
  size: str | None = None

  @functools.cached_property
  def svq_dataset(self) -> svq.SimpleVoiceQuestionsDataset:
    return svq.SimpleVoiceQuestionsDataset()

  @property
  def sub_tasks(self) -> list[str]:
    return [
        'speech_transcription',
        'speech_transcription:clean',
        'speech_transcription:media_noise',
        'speech_transcription:traffic_noise',
        'speech_transcription:background_speech',
    ]

  def _task_data(self, task_data_key: str, dtype: dict[str, Any] | None = None):
    df = self.svq_dataset.get_task_data(task_data_key, dtype=dtype)
    if self.locale:
      df = df[df.locale == self.locale]
    if self.size is not None:
      candidate_cols = [
          'span_context_id_cross_lang',
          'span_context_id_in_lang',
          'passage_id_cross_lang',
          'passage_id_in_lang',
      ]
      available_cols = [col for col in candidate_cols if col in df.columns]
      assert available_cols
      coalesced = df[available_cols].bfill(axis=1).iloc[:, 0]
      mask = coalesced.map(getattr(svq, f'is_member_of_{self.size}'))
      df = df[mask]
    return df

  def multimodal_inputs(self) -> Iterable[types.Sound]:
    df = self._task_data(
        f'utts_{self.locale}_*',
        dtype={  # pyrefly: ignore[bad-argument-type]
            'locale': str,
            'utt_id': str,
            task_lib.TRANSCRIPT_KEY.value: str,
            transcription.CONTEXTUAL_BIAS_KEY.value: str,  # pyrefly: ignore[bad-assignment]
        },
    )
    for example in df.to_dict('records'):
      sound = self.svq_dataset.get_sound(example)
      sound.context.text = example[task_lib.TRANSCRIPT_KEY.value]
      if transcription.CONTEXTUAL_BIAS_KEY.value:
        sound = types.SoundWithTitleAndContext(
            waveform=sound.waveform,
            context=sound.context,
            context_text=example.get(transcription.CONTEXTUAL_BIAS_KEY.value),
        )
      yield sound

  def multimodal_inputs_beam(self):
    return self.svq_dataset.get_task_sounds_beam(
        f'utts_{self.locale}_*', locale=self.locale
    )

  def examples(
      self, sub_task: str
  ) -> Iterable[transcription_evaluator.TranscriptTruth]:
    df = self._task_data(
        f'utts_{self.locale}_{_get_environment(sub_task)}',
        dtype={
            'locale': str,
            'utt_id': str,
            'transcript_truth': str,
        },
    )
    for example in df.to_dict('records'):
      yield transcription_evaluator.TranscriptTruth(
          sound_id=example['utt_id'],
          text=example['transcript_truth'],
          language=example['locale'],
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


def _make_task_class(
    base_cls,
    locale,
    size,
    suffix,
    eval_lang,
    description,
):
  """Dynamically create a locale-specific task class."""
  class_name = f'SVQ{suffix}{base_cls.__name__[len("SVQ"):]}'
  if size is not None:
    class_name += size.capitalize()
  cls = type(
      class_name,
      (base_cls,),
      {
          'locale': locale,
          'size': size,
          'metadata': types.TaskMetadata(
              name=class_name,
              description=description,
              reference='https://huggingface.co/datasets/google/svq',
              documentation_file='svq_retrieval.md',
              dataset_documentation_file='dataset_svq.md',
              type='SpeechTranscription',
              category='speech',
              main_score='WER',
              revision='1.0.0',
              dataset=types.Dataset(
                  name='SVQ',
                  path='https://huggingface.co/datasets/google/svq',
                  revision='1.0.0',
              ),
              scores=[
                  transcription_evaluator.wer(),
                  transcription_evaluator.ser(),
              ],
              eval_splits=['test'],
              eval_langs=[eval_lang],
              domains=['speech'],
              task_subtypes=['transcription'],
          ),
      },
  )
  return cls


# Generate all locale-specific classes and register them in the module.
# Default size.
for _locale, (_suffix, _eval_lang) in _SVQ_LOCALES.items():
  _cls = _make_task_class(  # pylint: disable=invalid-name
      base_cls=SVQSpeechTranscription,
      locale=_locale,
      size=None,
      suffix=_suffix,
      eval_lang=_eval_lang,
      description='Speech transcription task.',
  )
  globals()[_cls.__name__] = _cls


# Compact size.
for _locale, (_suffix, _eval_lang) in _SVQ_LOCALES.items():
  _cls = _make_task_class(  # pylint: disable=invalid-name
      base_cls=SVQSpeechTranscription,
      locale=_locale,
      size='compact',
      suffix=_suffix,
      eval_lang=_eval_lang,
      description='Speech transcription task.',
  )
  globals()[_cls.__name__] = _cls

# Debug size.
for _locale, (_suffix, _eval_lang) in {
    'en_us': ('EnUs', 'en-US'),
    'fi_fi': ('FiFi', 'fi-FI'),
}.items():
  _cls = _make_task_class(  # pylint: disable=invalid-name
      base_cls=SVQSpeechTranscription,
      locale=_locale,
      size='debug',
      suffix=_suffix,
      eval_lang=_eval_lang,
      description='Speech transcription task.',
  )
  globals()[_cls.__name__] = _cls
