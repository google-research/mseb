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

"""Test utilities for the Doppelganger dataset."""

import os

import numpy as np
import pandas as pd
from scipy.io import wavfile


def create_mock_dataset(base_dir: str) -> None:
  """Creates a mock Doppelganger dataset in the given directory."""
  metadata_dir = os.path.join(base_dir, 'mseb')
  os.makedirs(metadata_dir)
  audio_dir = os.path.join(base_dir, 'audio')
  os.makedirs(audio_dir)

  records = []
  for pair_id, event_id in (('1', 'A'), ('2', 'A'), ('3', 'B')):
    real_path = f'audio/real_{pair_id}.wav'
    synthetic_path = f'audio/synthetic_{pair_id}.wav'
    wavfile.write(
        os.path.join(base_dir, real_path),
        16000,
        np.zeros(8000, dtype=np.float32),
    )
    wavfile.write(
        os.path.join(base_dir, synthetic_path),
        16000,
        np.ones(4000, dtype=np.float32),
    )
    records.append({
        'pair_id': pair_id,
        'event_id': event_id,
        'event_name': event_id,
        'morphology': 'test',
        'source_clip_id': pair_id,
        'real_repo_id': 'unused',
        'real_revision': 'unused',
        'real_audio_path': real_path,
        'synthetic_repo_id': 'unused',
        'synthetic_revision': 'unused',
        'synthetic_audio_path': synthetic_path,
    })
  pd.DataFrame(records).to_csv(
      os.path.join(metadata_dir, 'test.csv'), index=False
  )
