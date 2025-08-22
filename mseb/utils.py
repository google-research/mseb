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

"""Utilities for MSEB library."""

import os
import subprocess
from typing import Optional

import librosa
import numpy as np
import soundfile


def download_from_hf(repo_id: str, target_dir: str, repo_type: str = "dataset"):
  """Clones a repository from Hugging Face if not already present."""
  if os.path.exists(os.path.join(target_dir, ".git")):
    print(f"Repo '{repo_id}' already found at {target_dir}. Skipping.")
    return

  print(f"Cloning '{repo_id}' to {target_dir}...")
  # For a real library, replacing this with the `huggingface_hub`
  # library would be more robust.
  clone_url = f"https://huggingface.co/{repo_type}s/{repo_id}"
  subprocess.run(
      ["git", "clone", clone_url, target_dir],
      check=True
  )


def read_audio(file_path: str,
               target_sr: Optional[int] = None
               ) -> tuple[np.ndarray, int]:
  """Reads an audio file."""
  waveform, orig_sr = soundfile.read(
      file_path,
      dtype="float32"
  )

  # Convert to mono
  if waveform.ndim > 1:
    waveform = np.mean(waveform, axis=1)

  if target_sr and target_sr != orig_sr:
    waveform = librosa.resample(
        waveform,
        orig_sr=orig_sr,
        target_sr=target_sr
    )
    return waveform, target_sr

  return waveform, orig_sr
