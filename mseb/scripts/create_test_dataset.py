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

r"""Creates a small, self-contained mini-dataset from the full SVQ dataset.

This is a one-time setup step required to run the end-to-end tests without
mocking the data source. It allows for fast, realistic integration tests by
subsetting the real data.

Prerequisite:
  You must first download the full SVQ dataset from Hugging Face:
  $ git lfs install
  $ git clone https://huggingface.co/datasets/google/svq

How to run:
  Navigate to the directory containing this script and run:

  $ python create_test_dataset.py --full_svq_path=/path/to/your/svq
"""

import os

from absl import app
from absl import flags
from absl import logging
import array_record
from mseb import utils
from mseb.datasets import simple_voice_questions
import tqdm


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'full_svq_path',
    None,
    'Path to the full SVQ dataset.',
    required=True
)

flags.DEFINE_string(
    'output_path',
    './testdata/svq_mini',
    'Path to write the mini dataset.'
)

flags.DEFINE_integer(
    'num_samples',
    10,
    'Number of samples to create.'
)

flags.DEFINE_string(
    'locale',
    'en_us',
    'Locale to sample from.'
)


def main(_):
  logging.info('Creating mini dataset at: %s', {FLAGS.output_path})
  audio_output_dir = os.path.join(FLAGS.output_path, 'audio')
  os.makedirs(audio_output_dir, exist_ok=True)
  full_dataset = simple_voice_questions.SimpleVoiceQuestionsDataset(
      base_path=FLAGS.full_svq_path
  )
  df_full_index = full_dataset.get_task_data('utt_index')
  df_mini = (
      df_full_index[df_full_index['locale'] == FLAGS.locale]
      .head(FLAGS.num_samples)
      .copy()
  )
  if len(df_mini) < FLAGS.num_samples:
    logging.info(
        'Warning: Found only %d samples for locale '
        '%s, which is less than the requested %d.',
        len(df_mini), FLAGS.locale, FLAGS.num_samples
    )
  mini_record_path = os.path.join(
      audio_output_dir,
      'mini_utts.array_record'
  )
  writer = array_record.ArrayRecordWriter(mini_record_path)
  try:
    new_indices = []
    logging.info('Processing %d samples...', len(df_mini))
    for i, row in enumerate(
        tqdm.tqdm(df_mini.itertuples(), total=len(df_mini))
    ):
      sound = full_dataset.get_sound({'utt_id': row.utt_id})
      wav_bytes = utils.sound_to_wav_bytes(sound)
      writer.write(wav_bytes)
      new_indices.append(f'audio/mini_utts:{i}')
  finally:
    writer.close()

  df_mini['index'] = new_indices
  df_mini.to_json(
      os.path.join(FLAGS.output_path, 'utt_index.jsonl'),
      orient='records',
      lines=True
  )

  logging.info(
      '\nSuccessfully created mini dataset with %d samples.',
      len(df_mini)
  )


if __name__ == '__main__':
  app.run(main)
