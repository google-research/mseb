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

import unittest

from mseb import leaderboard
from mseb import types


class LeaderboardTest(unittest.TestCase):

  def test_leaderboard_task_evaluation_result_json_conversion(self):
    """Test that to_json and from_json work correctly."""
    result = leaderboard.LeaderboardResult(
        name='test_encoder',
        sub_task_name='test_sub_task',
        task_metadata=types.TaskMetadata(
            name='test_task',
            description='test description',
            reference='test reference',
            type='test type',
            category='test category',
            main_score='accuracy',
            revision='test revision',
            dataset=types.Dataset(path='test path', revision='test revision'),
            scores=[
                types.Score(
                    metric='accuracy',
                    description='test description',
                    value=0.9,
                    min=0,
                    max=1,
                ),
                types.Score(
                    metric='f1',
                    description='test description',
                    value=0.8,
                    min=0,
                    max=1,
                ),
            ],
            eval_splits=['test'],
            eval_langs=['en'],
        ),
        scores=[
            types.Score(
                metric='accuracy',
                description='test description',
                value=0.9,
                min=0,
                max=1,
            ),
            types.Score(
                metric='f1',
                description='test description',
                value=0.8,
                min=0,
                max=1,
            ),
        ],
    )

    json_str = result.to_json()
    new_result = leaderboard.LeaderboardResult.from_json(json_str)
    self.assertEqual(result, new_result)


if __name__ == '__main__':
  unittest.main()
