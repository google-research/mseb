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

import os
import pathlib
from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from mseb import dataset
from mseb import runner as runner_lib
from mseb.encoders import raw_encoder
from mseb.tasks.clusterings import svq

FLAGS = flags.FLAGS

# Ensure flags are parsed, for example when running with pytest
if not FLAGS.is_parsed():
  FLAGS([''])


def get_test_encoder():
  return raw_encoder.RawEncoder(
      transform_fn=raw_encoder.spectrogram_transform,
      pooling='mean',
      frame_length=(48000 // 1000 * 25),
      frame_step=(48000 // 1000 * 10),
  )


class SVQClusteringTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        flagsaver.flagsaver(
            (dataset._DATASET_BASEPATH, self.get_testdata_path())
        )
    )

  def get_testdata_path(self, *args):
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent.parent, 'testdata'
    )
    return os.path.join(testdata_path, *args)

  def test_clustering_task(self):
    encoder = get_test_encoder()
    runner = runner_lib.DirectRunner(encoder=encoder)
    task = svq.SVQClustering()
    task.setup()
    self.assertEqual(
        task.sub_tasks, ['speaker_gender', 'speaker_age', 'speaker_id']
    )
    embeddings = runner.run(task.multimodal_inputs())
    scores = task.compute_scores(embeddings)
    self.assertLen(scores, 3)
    self.assertIn('speaker_gender', scores)
    self.assertLen(scores['speaker_gender'], 1)
    self.assertEqual(scores['speaker_gender'][0].metric, 'VMeasure')
    self.assertIn('speaker_age', scores)
    self.assertIn('speaker_id', scores)


class DynamicClassGenerationTest(absltest.TestCase):
  """Tests for the factory-generated locale-specific task classes."""

  def test_all_locale_classes_exist(self):
    for locale, (suffix, _) in svq._SVQ_LOCALES.items():
      class_name = f'SVQ{suffix}Clustering'
      self.assertTrue(
          hasattr(svq, class_name),
          f'Missing class {class_name} for locale {locale}',
      )

  def test_locale_class_has_correct_locale(self):
    task_cls = getattr(svq, 'SVQEnUsClustering')
    self.assertEqual(task_cls.locale, 'en_us')

  def test_locale_class_has_correct_metadata_name(self):
    task_cls = getattr(svq, 'SVQArEgClustering')
    self.assertEqual(task_cls.metadata.name, 'SVQArEgClustering')

  def test_locale_class_has_correct_eval_langs(self):
    task_cls = getattr(svq, 'SVQFiFiClustering')
    self.assertEqual(task_cls.metadata.eval_langs, ['fi-FI'])

  def test_locale_class_inherits_from_base(self):
    task_cls = getattr(svq, 'SVQKoKrClustering')
    self.assertTrue(issubclass(task_cls, svq.SVQClustering))

  def test_metadata_type_is_clustering(self):
    task_cls = getattr(svq, 'SVQHiInClustering')
    self.assertEqual(task_cls.metadata.type, 'Clustering')

  def test_metadata_main_score_is_vmeasure(self):
    task_cls = getattr(svq, 'SVQJaJpClustering')
    self.assertEqual(task_cls.metadata.main_score, 'VMeasure')

  def test_total_class_count(self):
    """26 locales = 26 classes."""
    num_locales = len(svq._SVQ_LOCALES)
    generated = [
        name
        for name in dir(svq)
        if name.startswith('SVQ')
        and name not in ('SVQClustering', 'SVQClusteringAll')
        and isinstance(getattr(svq, name), type)
        and issubclass(getattr(svq, name), svq.SVQClustering)
    ]
    self.assertLen(generated, num_locales)


class SVQClusteringAllTest(absltest.TestCase):
  """Tests for the SVQClusteringAll class."""

  def test_locale_is_none(self):
    self.assertIsNone(svq.SVQClusteringAll.locale)

  def test_inherits_from_base(self):
    self.assertTrue(issubclass(svq.SVQClusteringAll, svq.SVQClustering))


class BaseClassTest(absltest.TestCase):
  """Tests for SVQClustering base class attributes."""

  def test_base_locale_is_none(self):
    self.assertIsNone(svq.SVQClustering.locale)

  def test_sub_tasks(self):
    task = svq.SVQClustering()
    self.assertEqual(
        task.sub_tasks, ['speaker_gender', 'speaker_age', 'speaker_id']
    )


if __name__ == '__main__':
  absltest.main()
