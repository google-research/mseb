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

"""Tests for Hugging Face datasets streaming."""

import os
import sys
import tempfile

current_dir = os.path.dirname(os.path.abspath(__file__))
google3_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
)
tmp_test_lib_dir = os.path.join(google3_dir, "tmp_test_lib")
sys.path.append(tmp_test_lib_dir)

# pylint: disable=g-import-not-at-top
from mseb.datasets import fsd50k
from mseb.datasets import simple_voice_questions
from mseb.datasets import speech_massive
# pylint: enable=g-import-not-at-top


def test_dataset(dataset_class, name, **kwargs):
  """Tests a dataset for initialization and sound loading."""
  print(f"\n--- Testing {name} ---")
  with tempfile.TemporaryDirectory() as temp_dir:
    print(f"Using temp dir: {temp_dir}")
    try:
      ds = dataset_class(base_path=temp_dir, **kwargs)
      print(f"  Initialized {name}")
      print(f"  Length: {len(ds)}")
      if hasattr(ds, "get_task_data"):
        df = ds.get_task_data()
        print(f"  Got task data, shape: {df.shape}")
        if not df.empty:
          record = df.iloc[0].to_dict()
          sound = ds.get_sound(record)
          print(f"  Got sound: {sound.context.id}")
          print(f"  Waveform shape: {sound.waveform.shape}")
        else:
          print("  Dataset is empty")
      else:
        print("  Dataset doesn't support get_task_data")
    except Exception as e:  # pylint: disable=broad-except
      print(f"  Error testing {name}: {e}")


if __name__ == "__main__":
  test_dataset(
      speech_massive.SpeechMassiveDataset,
      "SpeechMassive",
      filename="de-DE/test-*.parquet",
      streaming=True,
  )

  test_dataset(fsd50k.FSD50KDataset, "FSD50K", split="test", streaming=True)

  test_dataset(
      simple_voice_questions.SimpleVoiceQuestionsDataset, "SVQ", streaming=True
  )
