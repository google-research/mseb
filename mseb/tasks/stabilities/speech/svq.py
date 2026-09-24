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

"""SVQ stability tasks split by representation mode."""

import functools
from typing import Any, Iterable, Type

from mseb import types
from mseb.datasets import simple_voice_questions as svq
from mseb.tasks import stability


class SVQStabilityBase(stability.StabilityTask):
  """Shared logic for SVQ Stability across all locales and modes.

  This task profiles the invariance of an encoder by comparing clean audio
  embeddings from the Simple Voice Questions (SVQ) dataset to augmented
  variants. It provides a measure of how representational geometry
  shifts under noise.
  """

  locale: str | None = None
  mode: str = ""  # Set to "continuous" or "discrete" by the factory

  @functools.cached_property
  def svq_dataset(self) -> svq.SimpleVoiceQuestionsDataset:
    return svq.SimpleVoiceQuestionsDataset()

  def _task_data(self, task_data_key: str, dtype: dict[str, Any] | None = None):
    ds = self.svq_dataset.get_task_data(task_data_key, dtype=dtype)
    ds = ds[ds["stabilities/speech"]]
    return ds

  def base_sounds(self) -> Iterable[types.Sound]:
    """Yields clean recordings to serve as stability anchors.

    Filter is applied to ensure only the 'clean' environment recordings are
    used, providing a consistent baseline for synthetic perturbation.

    Yields:
      A stream of Sound objects from the 'clean' subset of the dataset.

    Raises:
      ValueError: If the `locale` attribute is not set by a subclass.
    """
    if not self.locale:
      raise ValueError("`locale` must be set by a concrete task subclass.")

    df = self._task_data(
        f"utts_{self.locale}_clean",
        dtype={"utt_id": str},
    )
    for example in df.to_dict("records"):
      yield self.svq_dataset.get_sound(example)


def create_svq_stability_variant(
    locale: str, lang_code: str, mode: str
) -> Type[SVQStabilityBase]:
  """Factory to create a specific Task class for a locale and mode.

  The factory creates dual entries for each locale. This ensures that a
  single model run can populate both Continuous and Discrete leaderboards
  using the most appropriate primary metric for each.

  Args:
    locale: The dataset locale string (e.g., 'en_us').
    lang_code: The standard BCP-47 language tag (e.g., 'en-US').
    mode: The representation type, either 'continuous' or 'discrete'.

  Returns:
    A unique class type inheriting from SVQStabilityBase with populated
    metadata.
  """
  mode_label = "Continuous" if mode == "continuous" else "Discrete"
  class_name = f"SVQ{locale.title().replace('_', '')}{mode_label}Stability"

  # The main_score defines which metric the leaderboard uses for ranking.
  main_metric = "Corpus_Mean_CED" if mode == "continuous" else "Corpus_Mean_UED"
  main_score_obj = types.Score(
      metric=main_metric,
      description=f"Global Micro-Average {mode} stability drift.",
      value=0.0,
      min=0.0,
      max=1.0,
  )
  return type(
      class_name,
      (SVQStabilityBase,),
      {
          "locale": locale,
          "mode": mode,
          "metadata": types.TaskMetadata(
              name=class_name,
              description=(
                  f"Stability profiling ({mode} focus) task on SVQ"
                  f"for {locale}. Measures representational drift under noise."
              ),
              reference="https://huggingface.co/datasets/google/svq",
              documentation_file="svq_stability.md",
              dataset_documentation_file="dataset_svq.md",
              type="Stability",
              category="speech",
              main_score=main_metric,
              revision="1.0.0",
              dataset=types.Dataset(
                  name="SVQ",
                  path="https://huggingface.co/datasets/google/svq",
                  revision=svq.VERSION,
              ),
              scores=[main_score_obj],
              eval_splits=["test"],
              eval_langs=[lang_code],
              domains=["speech"],
              task_subtypes=["stability", mode],
          ),
      },
  )


# Full set of SVQ Locales
_LOCALES = {
    "ar_eg": "ar-EG",
    "ar_x_gulf": "ar-x-gulf",
    "ar_x_levant": "ar-x-levant",
    "ar_x_maghrebi": "ar-x-maghrebi",
    "bn_bd": "bn-BD",
    "bn_in": "bn-IN",
    "en_au": "en-AU",
    "en_gb": "en-GB",
    "en_in": "en-IN",
    "en_ph": "en-PH",
    "en_us": "en-US",
    "fi_fi": "fi-FI",
    "gu_in": "gu-IN",
    "hi_in": "hi-IN",
    "id_id": "id-ID",
    "ja_jp": "ja-JP",
    "kn_in": "kn-IN",
    "ko_kr": "ko-KR",
    "ml_in": "ml-IN",
    "mr_in": "mr-IN",
    "ru_ru": "ru-RU",
    "sw": "Sw",
    "ta_in": "ta-IN",
    "te_in": "te-IN",
    "ur_in": "ur-IN",
    "ur_pk": "ur-PK",
}

# Dynamically register both Continuous and Discrete variants
for loc, lang in _LOCALES.items():
  # Register the Continuous variant (Main Score: CED)
  ContCls = create_svq_stability_variant(loc, lang, "continuous")
  globals()[ContCls.__name__] = ContCls

  # Register the Discrete variant (Main Score: UED)
  DiscCls = create_svq_stability_variant(loc, lang, "discrete")
  globals()[DiscCls.__name__] = DiscCls
