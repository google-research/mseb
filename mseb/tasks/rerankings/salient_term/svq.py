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

"""SVQ salient term reranking tasks."""

import functools
import hashlib
import os
import random
from typing import Any, Iterable, Mapping, Sequence

from absl import flags
from mseb import task as task_lib
from mseb import types
from mseb.datasets import simple_voice_questions as svq
from mseb.evaluators import reranking_evaluator
from mseb.tasks import reranking

_RANDOMIZE_CANDIDATE_SALIENT_TERMS = flags.DEFINE_bool(
    'randomize_candidate_salient_terms',
    True,
    'Whether to randomize the salient term reranking candidates to avoid any '
    'potential bias in the order of the candidates.',
)

_filter_fn_by_sub_task = {
    'salient_term_reranking': lambda x: True,
    'salient_term_reranking:clean': lambda x: x['environment'] == 'clean',
    'salient_term_reranking:media_noise': (
        lambda x: x['environment'] == 'media_noise'
    ),
    'salient_term_reranking:traffic_noise': (
        lambda x: x['environment'] == 'traffic_noise'
    ),
    'salient_term_reranking:background_speech': (
        lambda x: x['environment'] == 'background_speech'
    ),
}


def _seed_from_candidates(candidates: Sequence[str]) -> int:
  sha_hash = hashlib.sha256('\n'.join(candidates).encode('utf-8')).hexdigest()
  return int(sha_hash, 16)


def _get_context_text(candidates: Sequence[str], randomize: bool) -> str:
  if randomize:
    candidates = list(candidates)
    random.seed(_seed_from_candidates(candidates))
    random.shuffle(candidates)
    random.seed()

  return types.ValidListPrediction(
      items=[{'id': i, 'text': c} for i, c in enumerate(candidates)]
  ).to_json()


def _get_rank_by_id(
    candidates: Sequence[str], randomize: bool
) -> Mapping[int, int] | None:
  if not randomize:
    return None

  random.seed(_seed_from_candidates(candidates))
  rank_by_id = list(range(len(candidates)))
  random.shuffle(rank_by_id)
  rank_by_id = {i: r for i, r in enumerate(rank_by_id)}
  random.seed()
  return rank_by_id


class SVQSalientTermReranking(reranking.RerankingTask):
  """SVQ salient term reranking."""

  locale: str | None = None

  @functools.cached_property
  def svq_dataset(self) -> svq.SimpleVoiceQuestionsDataset:
    return svq.SimpleVoiceQuestionsDataset()

  @property
  def embeddings_dir(self) -> str:
    assert self.locale is not None
    name = f'svq_{self.locale}_salient_term_reranking'
    return os.path.join(super().embeddings_dir, name)

  def _task_data(self, task_data_key: str, dtype: dict[str, Any] | None = None):
    df = self.svq_dataset.get_task_data(task_data_key, dtype=dtype)
    if self.locale:
      df = df[df.locale == self.locale]
    return df

  @property
  def sub_tasks(self) -> list[str]:
    return list(_filter_fn_by_sub_task.keys())

  def multimodal_inputs(self) -> Iterable[types.SoundWithTitleAndContext]:
    df = self._task_data(
        'salient_term',
        dtype={
            'locale': str,
            'utt_id': str,
            task_lib.TRANSCRIPT_KEY.value: str,
            'candidate_salient_terms': Sequence[str],
        },
    )
    for example in df.to_dict('records'):
      sound = self.svq_dataset.get_sound(example)
      sound.context.text = example[task_lib.TRANSCRIPT_KEY.value]
      context_text = _get_context_text(
          example['candidate_salient_terms'],
          randomize=_RANDOMIZE_CANDIDATE_SALIENT_TERMS.value,
      )
      sound = types.SoundWithTitleAndContext(
          waveform=sound.waveform,
          context=sound.context,
          context_text=context_text,
      )
      yield sound

  def examples(
      self, sub_task: str
  ) -> Iterable[reranking_evaluator.RerankingCandidates]:
    filter_fn = _filter_fn_by_sub_task[sub_task]
    df = self._task_data(
        'salient_term',
        dtype={
            'locale': str,
            'utt_id': str,
            'candidate_salient_terms': Sequence[str],
            'topk_salient_terms': Sequence[str],
        },
    )
    for example in df.to_dict('records'):
      if filter_fn(example):
        rank_by_id = _get_rank_by_id(
            example['candidate_salient_terms'],
            randomize=_RANDOMIZE_CANDIDATE_SALIENT_TERMS.value,
        )
        yield reranking_evaluator.RerankingCandidates(
            sound_id=example['utt_id'],
            texts=example['topk_salient_terms'],
            language=example['locale'],
            rank_by_id=rank_by_id,
        )

  def candidate_lists(self) -> Iterable[tuple[str, Sequence[types.Text]]]:
    df = self._task_data(
        'salient_term',
        dtype={
            'locale': str,
            'candidate_salient_terms': Sequence[str],
        },
    )
    for example in df.to_dict('records'):
      yield (
          example['utt_id'],
          [
              types.Text(
                  text=candidate, context=types.TextContextParams(id=candidate)
              )
              for candidate in example['candidate_salient_terms']
          ],
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
    suffix,
    eval_lang,
    description,
):
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
              documentation_file='svq_retrieval.md',
              dataset_documentation_file='dataset_svq.md',
              type='SalientTermReranking',
              category='speech',
              main_score='NDCG',
              revision='1.0.0',
              dataset=types.Dataset(
                  name='SVQ',
                  path='https://huggingface.co/datasets/google/svq',
                  revision='1.0.0',
              ),
              scores=[
                  reranking_evaluator.ndcg(),
                  reranking_evaluator.map(),
                  reranking_evaluator.mrr(),
                  reranking_evaluator.wer(),
                  reranking_evaluator.cer(),
              ],
              eval_splits=['test'],
              eval_langs=[eval_lang],
              domains=['speech'],
              task_subtypes=['reranking'],
          ),
      },
  )
  return cls


# Generate all locale-specific classes and register them in the module.
# Default size.
for _locale, (_suffix, _eval_lang) in _SVQ_LOCALES.items():
  _cls = _make_task_class(  # pylint: disable=invalid-name
      base_cls=SVQSalientTermReranking,
      locale=_locale,
      suffix=_suffix,
      eval_lang=_eval_lang,
      description='Salient term reranking task.',
  )
  globals()[_cls.__name__] = _cls
