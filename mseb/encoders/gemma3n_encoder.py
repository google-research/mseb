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

"""Gemma 3n encoder."""

import logging
import os
import re
import tempfile
from typing import Optional, Tuple

from mseb.encoders import hf_llm_encoder
import numpy as np
import transformers


class Gemma3nEncoder(hf_llm_encoder.HFLLMEncoder):
  """Encoder using a Gemma 3n model."""

  def _setup(self):
    """Loads the HF Transformers model."""
    if self._model is None:
      logging.info('Loading model at: %s', self._model_path)
      self._processor = transformers.AutoProcessor.from_pretrained(
          self._model_path
      )
      # Gemma 3n is registered under AutoModelForConditionalGeneration
      self._model = transformers.AutoModelForImageTextToText.from_pretrained(
          self._model_path,
          device_map=self._device_map,
          torch_dtype=self._torch_dtype,
      )

    self.prompt_encode_fn = lambda prompts: np.array([
        Gemma3nEncoder.get_response(
            prompt,
            processor=self._processor,
            model=self._model,
            max_new_tokens=self._max_new_tokens,
        )
        for prompt in prompts
    ])

  @staticmethod
  def get_response(
      request_prompt: Tuple[str, Optional[bytes]],
      *,
      processor: transformers.AutoProcessor,
      model: transformers.AutoModelForImageTextToText,
      max_new_tokens: int,
  ) -> str:
    """Returns the prediction for the given question, title and context."""
    assert processor is not None, 'Processor is not initialized.'
    assert model is not None, 'Model is not initialized.'
    messages = list([{
        'role': 'user',
        'content': list(),
    }])
    messages[0]['content'].append({'type': 'text', 'text': request_prompt[0]})
    tmp_wavefile = None
    if request_prompt[1] is not None:
      tmp_wavefile = tempfile.NamedTemporaryFile(
          mode='wb', delete=False, delete_on_close=False
      )

      try:
        tmp_wavefile.write(request_prompt[1])
      finally:
        tmp_wavefile.close()

      # Gemma 3n expects 'audio' key instead of 'path'
      messages[0]['content'].append(
          {'type': 'audio', 'audio': tmp_wavefile.name}
      )

    input_ids = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors='pt',
    )
    input_len = input_ids['input_ids'].shape[-1]
    input_ids = input_ids.to(model.device, dtype=model.dtype)

    if tmp_wavefile is not None:
      os.unlink(tmp_wavefile.name)

    # Generate output from the model
    outputs = model.generate(**input_ids, max_new_tokens=max_new_tokens)

    # decode and print the output as text
    text = processor.batch_decode(
        outputs[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    response = text[0]

    # Remove markdown and json formatting.
    response = re.sub(r'```json\s*|\s*```', '', response)
    response = re.sub(r'JSON\s*', '', response)

    return response
