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

import collections
import json
import os
import pathlib
import shutil
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
    testdata_path = os.path.join(
        pathlib.Path(os.path.abspath(__file__)).parent.parent.parent, 'testdata'
    )
    temp_dir = self.create_tempdir().full_path
    shutil.copytree(testdata_path, temp_dir, dirs_exist_ok=True)
    os.chmod(temp_dir, 0o755)
    utt_index_path = os.path.join(temp_dir, 'utt_index.jsonl')
    with open(utt_index_path, 'r') as f:
      records = [json.loads(line) for line in f if line.strip()]

    by_loc_env = collections.defaultdict(list)
    for record in records:
      by_loc_env[(record['locale'], record['environment'])].append(record)

    for (loc, env), recs in by_loc_env.items():
      path = os.path.join(temp_dir, f'utts_{loc}_{env}.jsonl')
      with open(path, 'w') as f:
        for r in recs:
          f.write(json.dumps(r) + '\n')

    self.enter_context(
        flagsaver.flagsaver((dataset._DATASET_BASEPATH, temp_dir))
    )

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
      class_name = f'SVQClustering{suffix}'
      self.assertTrue(
          hasattr(svq, class_name),
          f'Missing class {class_name} for locale {locale}',
      )

  def test_locale_class_has_correct_locale(self):
    task_cls = getattr(svq, 'SVQClusteringEnUs')
    self.assertEqual(task_cls.locale, 'en_us')

  def test_locale_class_has_correct_metadata_name(self):
    task_cls = getattr(svq, 'SVQClusteringArEg')
    self.assertEqual(task_cls.metadata.name, 'SVQClusteringArEg')

  def test_locale_class_has_correct_eval_langs(self):
    task_cls = getattr(svq, 'SVQClusteringFiFi')
    self.assertEqual(task_cls.metadata.eval_langs, ['fi-FI'])

  def test_locale_class_inherits_from_base(self):
    task_cls = getattr(svq, 'SVQClusteringKoKr')
    self.assertTrue(issubclass(task_cls, svq.SVQClustering))

  def test_locale_class_default_size_is_none(self):
    task_cls = getattr(svq, 'SVQClusteringSw')
    self.assertIsNone(task_cls.size)

  def test_compact_classes_exist(self):
    for locale, (suffix, _) in svq._SVQ_LOCALES.items():
      class_name = f'SVQClustering{suffix}Compact'
      self.assertTrue(
          hasattr(svq, class_name),
          f'Missing compact class {class_name} for locale {locale}',
      )

  def test_compact_class_has_correct_size(self):
    task_cls = getattr(svq, 'SVQClusteringEnUsCompact')
    self.assertEqual(task_cls.size, 'compact')

  def test_compact_class_has_correct_locale(self):
    task_cls = getattr(svq, 'SVQClusteringRuRuCompact')
    self.assertEqual(task_cls.locale, 'ru_ru')

  def test_compact_class_inherits_from_base(self):
    task_cls = getattr(svq, 'SVQClusteringTeInCompact')
    self.assertTrue(issubclass(task_cls, svq.SVQClustering))

  def test_debug_classes_exist(self):
    for suffix in ('EnUs', 'FiFi'):
      class_name = f'SVQClustering{suffix}Debug'
      self.assertTrue(
          hasattr(svq, class_name),
          f'Missing debug class {class_name}',
      )

  def test_debug_class_has_correct_size(self):
    task_cls = getattr(svq, 'SVQClusteringEnUsDebug')
    self.assertEqual(task_cls.size, 'debug')

  def test_debug_only_en_us_and_fi_fi(self):
    """Debug classes should only exist for en-US and fi-FI."""
    for locale, (suffix, _) in svq._SVQ_LOCALES.items():
      class_name = f'SVQClustering{suffix}Debug'
      if locale in ('en_us', 'fi_fi'):
        self.assertTrue(hasattr(svq, class_name))
      else:
        self.assertFalse(
            hasattr(svq, class_name),
            f'Unexpected debug class {class_name}',
        )

  def test_metadata_type_is_clustering(self):
    task_cls = getattr(svq, 'SVQClusteringHiIn')
    self.assertEqual(task_cls.metadata.type, 'Clustering')

  def test_metadata_main_score_is_vmeasure(self):
    task_cls = getattr(svq, 'SVQClusteringJaJp')
    self.assertEqual(task_cls.metadata.main_score, 'VMeasure')

  def test_total_class_count(self):
    """26 locales * 2 (default + compact) + 2 debug = 54 classes."""
    num_locales = len(svq._SVQ_LOCALES) * 2 + 2
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

  def test_base_size_is_none(self):
    self.assertIsNone(svq.SVQClustering.size)

  def test_sub_tasks(self):
    task = svq.SVQClustering()
    self.assertEqual(
        task.sub_tasks, ['speaker_gender', 'speaker_age', 'speaker_id']
    )


if __name__ == '__main__':
  absltest.main()
