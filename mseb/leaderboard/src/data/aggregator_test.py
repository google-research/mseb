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

from absl.testing import absltest
from src import config
from src.data import aggregator
from src.data import models


def _classification_record(score: float) -> models.EvaluationRecord:
  return models.EvaluationRecord(
      model_entry=models.ModelEntry(entry_id="model", top_model="model"),
      task_name="classification",
      dataset_name="dataset",
      main_score_name="Accuracy",
      main_score_value=score,
  )


class AggregatorTest(absltest.TestCase):

  def test_overall_average_excludes_inactive_tasks(self):
    df = aggregator.build_overall_dataframe(
        records=[_classification_record(0.6)]
    )
    self.assertAlmostEqual(
        df.iloc[0]["Average ⬆️"], 60.0 / len(config.ACTIVE_TASK_KEYS), places=2
    )

  def test_dynamic_average_excludes_inactive_tasks(self):
    df = aggregator.build_overall_dataframe(
        records=[_classification_record(0.6)]
    )
    df = aggregator.compute_dynamic_overall_average(
        df, ["Classification", "Segmentation"]
    )
    self.assertAlmostEqual(df.iloc[0]["Average ⬆️"], 60.0)


if __name__ == "__main__":
  absltest.main()
