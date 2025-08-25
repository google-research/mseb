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

"""Segmentation tasks."""

import abc
import os
from mseb import svq_data
from mseb import task
from mseb import types
from mseb.evaluators import segmentation_evaluator


class SegmentationTask(task.MSEBTask):
  """Segmentation task."""

  def __init__(self, base_path: str):
    self._base_path = base_path

  @abc.abstractmethod
  def targets(self, embeddings: types.SoundEmbeddingCache):
    """Get example labels for the segmentation task."""
    # TODO(tombagby): This is only taking embeddings right now because we
    # don't have reference labels in data yet and this is the easiest way
    # to fake it. Actual signature will not take embeddings.

  def compute_scores(
      self, embeddings: types.SoundEmbeddingCache
  ) -> dict[str, list[types.Score]]:
    evaluator = segmentation_evaluator.SegmentationEvaluator()
    scores = []
    for ex in self.targets(embeddings):
      # FAKE REFERENCES
      reference_waveform_embeddings = ex.embedding
      reference_embedding_timestamps = ex.timestamps
      scores.extend([
          evaluator.evaluate(
              ex.embedding,
              ex.timestamps,
              ex.context,
              reference_waveform_embeddings=reference_waveform_embeddings,
              reference_embedding_timestamps=reference_embedding_timestamps,
          )
      ])
    return {'segmentation': evaluator.combine_scores(scores)}


class SegmentationTaskSVQ(SegmentationTask):
  """Segmentation task on SVQ dataset."""

  metadata = types.TaskMetadata(
      name='SegmentationTaskSVQ',
      description='Segmentation task.',
      reference='TODO',
      type='Segmentation',
      category='speech',
      main_score='TimestampsAccuracy',
      revision='1.0.0',
      dataset=types.Dataset(
          path='https://huggingface.co/datasets/google/svq',
          revision='1.0.0',
      ),
      scores=[
          segmentation_evaluator.timestamps_accuracy_score(),
          segmentation_evaluator.embeddings_accuracy_score(),
          segmentation_evaluator.timestamps_and_embedding_accuracy_score(),
      ],
      eval_splits=['test'],
      eval_langs=['en-US'],
      domains=['speech'],
      task_subtypes=['segmentation'],
  )

  def sounds(self):
    for example in svq_data.generate_examples(
        os.path.join(self._base_path, 'utt_index.jsonl')
    ):
      yield types.Sound(
          example['waveform'],
          types.SoundContextParams(
              sample_rate=48000,
              length=len(example['waveform']),
              id=example['utt_id'],
          ),
      )

  def targets(self, embeddings: types.SoundEmbeddingCache):
    for ex in svq_data.generate_examples(
        os.path.join(self._base_path, 'utt_index.jsonl')
    ):
      # TODO(tombagby): Get actual reference labels out, faking for now.
      yield embeddings[ex['utt_id']]
