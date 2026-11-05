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

"""HuthLab brain encoding task implementation.

Concrete brain encoding task using the HuthLab encoding-model-scaling-laws
dataset. Evaluates speech embeddings by training ridge regression models to
predict fMRI BOLD responses from subjects listening to narrative stories.

Reference:
  Antonello, Turek, Vo, and Huth (2024). "Scaling in Speech and Language Models:
  A Path to Human-Level Performance?" arXiv:2401.10150.
"""

from collections.abc import Iterable, Mapping, Sequence
import dataclasses

from mseb import types
from mseb.datasets import huthlab_fmri
from mseb.evaluators import brain_encoding_evaluator
from mseb.tasks import brain_encoding
import numpy as np

_CITATION = """@article{antonello2024scaling,
  title={Scaling in Speech and Language Models: A Path to Human-Level Performance?},
  author={Richard J. Antonello and Tuomas Turek and Nghia Vo and Alexander G. Huth},
  journal={arXiv preprint arXiv:2401.10150},
  year={2024}
}"""


class _HuthLabBrainEncoding(brain_encoding.BrainEncodingTask):
  """Base class for brain encoding tasks using HuthLab fMRI data.

  This task loads narrative story stimuli and corresponding fMRI responses
  from the HuthLab encoding-model-scaling-laws dataset. For each sub-task
  (one per subject), it:

  1. Yields story audio as Sound objects for encoding.
  2. Aligns the resulting embeddings to fMRI TRs.
  3. Trains ridge regression to predict held-out fMRI responses.
  4. Reports voxel-wise prediction correlations.

  The sub-tasks correspond to individual subjects (e.g., 'S01', 'S02', 'S03').
  """

  train_stories: Sequence[str] | None = None
  test_stories: Sequence[str] | None = None

  def __init__(
      self,
      base_path: str | None = None,
      subjects: Sequence[str] = ('S03',),
      tr_duration: float = huthlab_fmri.DEFAULT_TR_DURATION,
      delays: Sequence[int] = huthlab_fmri.DEFAULT_DELAYS,
      alphas: Sequence[float] | None = None,
  ):
    """Initializes the HuthLabBrainEncoding task.

    Args:
      base_path: Root path to the HuthLab data. If None, uses the
        --huthlab_data_path flag or --dataset_basepath.
      subjects: Subjects to evaluate (each becomes a sub-task).
      tr_duration: Duration of one fMRI TR in seconds.
      delays: FIR delay values in TRs.
      alphas: Ridge regularization candidates.
    """
    super().__init__(
        tr_duration=tr_duration,
        delays=delays,
        alphas=alphas,
    )
    self._subjects = list(subjects)
    self._tr_duration = tr_duration
    self._datasets: dict[str, huthlab_fmri.HuthLabDataset] = {}

    for subject in self._subjects:
      self._datasets[subject] = huthlab_fmri.HuthLabDataset(
          base_path=base_path,
          subject=subject,
          tr_duration=tr_duration,
          train_stories=self.train_stories,
          test_stories=self.test_stories,
      )

  def _dataset(self, subject: str) -> huthlab_fmri.HuthLabDataset:
    return self._datasets[subject]

  @property
  def sub_tasks(self) -> list[str]:
    """Each subject is a sub-task."""
    return self._subjects

  def multimodal_inputs(self) -> Iterable[types.Sound]:
    """Yields story audio as Sound objects for all stimulus stories.

    Stories from both training and test sets are yielded (deduplicated).
    """
    # Use the first subject's dataset for story metadata (same across subjects).
    dataset = self._dataset(self._subjects[0])
    all_stories = list(dataset.get_train_stories()) + list(
        dataset.get_test_stories()
    )
    seen = set()
    for story in all_stories:
      if story in seen:
        continue
      seen.add(story)
      yield dataset.get_sound(story)

  def _build_examples(
      self,
      stories: Sequence[str],
      responses: Mapping[str, np.ndarray],
      trim_start: int,
      resp_offset: int = 0,
  ) -> Sequence[brain_encoding_evaluator.BrainEncodingExample]:
    """Builds BrainEncodingExample objects for a set of stories.

    Each story maps to a contiguous range of TRs in the concatenated fMRI
    response matrix.

    Args:
      stories: Story names.
      responses: Mapping of story name to fMRI response matrix.
      trim_start: Number of TRs to trim from the start of each story audio.
      resp_offset: Starting TR offset within the story's fMRI response matrix.

    Returns:
      Sequence of BrainEncodingExample objects.
    """
    examples = []
    for story in stories:
      n_trs = responses[story][resp_offset:].shape[0]
      examples.append(
          brain_encoding_evaluator.BrainEncodingExample(
              sound_id=story,
              tr_index_start=trim_start,
              tr_index_end=trim_start + n_trs,
          )
      )
    return examples

  def train_examples(
      self, sub_task: str
  ) -> Sequence[brain_encoding_evaluator.BrainEncodingExample]:
    """Returns training examples for a subject."""
    dataset = self._dataset(sub_task)
    return self._build_examples(
        dataset.get_train_stories(),
        dataset.load_responses(),
        trim_start=5,
        resp_offset=0,
    )

  def test_examples(
      self, sub_task: str
  ) -> Sequence[brain_encoding_evaluator.BrainEncodingExample]:
    """Returns test examples for a subject."""
    dataset = self._dataset(sub_task)
    return self._build_examples(
        dataset.get_test_stories(),
        dataset.load_responses(),
        trim_start=45,
        resp_offset=40,
    )

  def fmri_train(self, sub_task: str) -> np.ndarray:
    """Returns training fMRI responses for a subject."""
    return self._dataset(sub_task).load_train_responses()

  def fmri_test(self, sub_task: str) -> np.ndarray:
    """Returns test fMRI responses (averaged across sessions) for a subject."""
    return self._dataset(sub_task).load_test_responses()


class HuthLabBrainEncoding(_HuthLabBrainEncoding):
  """Brain encoding task using HuthLab fMRI data."""

  metadata = types.TaskMetadata(
      name='HuthLabBrainEncoding',
      description=(
          'Brain encoding model evaluation using the HuthLab'
          ' encoding-model-scaling-laws fMRI dataset. Predicts voxel-level'
          ' fMRI BOLD responses from speech embeddings via ridge regression.'
      ),
      reference=_CITATION,
      type='BrainEncoding',
      category='speech',
      main_score='MeanCorrelation',
      revision='1.0.0',
      dataset=types.Dataset(
          name='HuthLabScalingLaws',
          path='https://utexas.box.com/v/EncodingModelScalingLaws',
          revision='1.0.0',
      ),
      scores=[
          brain_encoding_evaluator.mean_correlation_score(),
          brain_encoding_evaluator.median_correlation_score(),
          brain_encoding_evaluator.fraction_significant_score(),
      ],
      eval_splits=['test'],
      eval_langs=['en'],
      domains=['speech', 'neuroscience', 'brain_encoding'],
      task_subtypes=['brain_encoding', 'ridge_regression'],
  )


class HuthLabBrainEncodingDebug(_HuthLabBrainEncoding):
  """Debug version of HuthLabBrainEncoding using a small subset of stories."""

  train_stories = huthlab_fmri.DEBUG_TRAIN_STORIES
  test_stories = huthlab_fmri.DEBUG_TEST_STORIES

  metadata = dataclasses.replace(
      HuthLabBrainEncoding.metadata,
      name='HuthLabBrainEncodingDebug',
  )
