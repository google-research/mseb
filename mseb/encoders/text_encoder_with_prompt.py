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

"""Multi-modal encoders with prompt."""

import dataclasses
import inspect
import json
import logging
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, final

from absl import flags
import jaxtyping
from mseb import encoder
from mseb import tasks
from mseb import types
from mseb import utils
from mseb.encoders import prompt as prompt_lib
from mseb.encoders import prompt_registry
from mseb.tasks import classification
import numpy as np

_PROMPT_OVERRIDE = flags.DEFINE_string(
    'prompt_override', None, 'Override the prompt template.'
)


class TextEncoderWithPrompt(encoder.MultiModalEncoder):
  """Defines the interface for encoding text and sound into embeddings using a prompt.

  This abstract class provides a standardized structure for encoders
  within the MSEB benchmark. It's designed for lazy loading of models, making it
  suitable for large-scale, distributed processing environments.

  Subclasses are responsible for implementing the model loading logic (`setup`).
  """

  # String to use for the `text`` field in the prompt template when encoding
  # audio.
  TEXT_FOR_AUDIO = '(in following audio file)'

  def __init__(
      self,
      normalizer: Callable[[str], str] | None = None,
      prompt: prompt_lib.Prompt = prompt_lib.DefaultPrompt(),
      task_prompts: Mapping[str, Any] | None = None,
  ):
    """Initializes the encoder with configuration.

    All subclasses of this class are expected to load the model in `_setup` and
    set `prompt_encode_fn` that takes a sequence of multimodal prompts and
    returns a sequence of embeddings (2d np.ndarray or sequence of 1d
    np.ndarray).

    Args:
      normalizer: A function that normalizes the text before encoding. This is
        useful for removing special characters or formatting the text for better
        encoding results.
      prompt: Prompt object definiing the prompt template and reponse format to
        be used for encoding.
      task_prompts: A mapping from task name to Prompt object.
    """
    super().__init__()
    self.normalizer = normalizer
    self.prompt = prompt
    self.task_prompts = task_prompts
    self._current_task = None
    self._last_used_prompt = None
    self._resolved_prompts: dict[tuple[int, str | None], prompt_lib.Prompt] = {}

    self.prompt_encode_fn: (
        Callable[
            [Sequence[Tuple[str, Optional[bytes]]]],
            jaxtyping.Float[jaxtyping.Array, 'N D']
            | jaxtyping.Shaped[np.ndarray, 'N']
            | Sequence[
                jaxtyping.Float[jaxtyping.Array, 'D']
                | jaxtyping.Shaped[np.ndarray, '']
            ],
        ]
        | None
    ) = None

  def set_task(self, task):
    """Sets the current task for the encoder."""
    self._current_task = task
    self._resolved_prompts.clear()

  def get_last_used_prompt(self) -> prompt_lib.Prompt | None:
    """Returns the last used prompt."""
    return self._last_used_prompt

  @final
  def _check_input_types(self, batch: Sequence[types.MultiModalObject]) -> None:
    if not all(
        any(isinstance(x, type) for type in [types.Text, types.Sound])
        for x in batch
    ):
      raise ValueError(
          'TextEncoderWithPrompt only supports a batch of Text or Sound inputs.'
      )

  def _resolve_prompt(self, task_name: str | None = None) -> prompt_lib.Prompt:
    """Resolves, loads, and caches the prompt for the given or current task."""
    prompt: Any = self.prompt
    task_cls = None
    task_obj = None
    metadata = None
    if task_name:
      try:
        task_cls = tasks.get_task_by_name(task_name)
        metadata = task_cls.metadata
        if (
            self._current_task is not None
            and getattr(
                getattr(self._current_task, 'metadata', None), 'name', None
            )
            == task_name
        ):
          task_obj = self._current_task
      except Exception as e:  # pylint: disable=broad-except
        logging.warning('Failed to get task by name %s: %s', task_name, e)
    elif self._current_task is not None:
      task_obj = self._current_task
      task_cls = type(self._current_task)
      metadata = getattr(self._current_task, 'metadata', None)

    if self.task_prompts:
      candidate_keys: list[str] = []
      if metadata is not None:
        if isinstance(getattr(metadata, 'name', None), str):
          candidate_keys.append(metadata.name)
        if isinstance(getattr(metadata, 'type', None), str):
          candidate_keys.append(metadata.type)
      if task_cls is not None:
        candidate_keys.extend(cls.__name__ for cls in task_cls.__mro__)
      if metadata is not None:
        subtypes = getattr(metadata, 'task_subtypes', None)
        if isinstance(subtypes, Sequence) and not isinstance(subtypes, str):
          candidate_keys.extend(s for s in subtypes if isinstance(s, str))
        if isinstance(getattr(metadata, 'category', None), str):
          candidate_keys.append(metadata.category)
      for key in candidate_keys:
        if key in self.task_prompts:
          prompt = self.task_prompts[key]
          break

    if hasattr(prompt, 'load'):
      meta_name = getattr(metadata, 'name', None)
      task_cache_key = (
          meta_name
          if isinstance(meta_name, str)
          else (task_cls.__name__ if task_cls is not None else None)
      )
      cache_key = (id(prompt), task_cache_key)
      if cache_key not in self._resolved_prompts:
        if isinstance(prompt, prompt_registry.PromptMetadata):
          params = dict(prompt.params())
          if (
              'class_labels' not in params
              and 'class_labels' in inspect.signature(prompt.prompt).parameters
          ):
            cls_task = None
            if isinstance(task_obj, classification.ClassificationTask):
              cls_task = task_obj
            elif isinstance(task_cls, type) and issubclass(
                task_cls, classification.ClassificationTask
            ):
              cls_task = task_cls()
            if cls_task is not None:
              params['class_labels'] = list(cls_task.class_labels())
          self._resolved_prompts[cache_key] = prompt.prompt(**params)
        else:
          self._resolved_prompts[cache_key] = prompt.load()
      prompt = self._resolved_prompts[cache_key]
    return prompt

  def _get_normalized_text_prompt(
      self,
      text: str,
      title: str | None = None,
      context: str | None = None,
      task_name: str | None = None,
      context_params: (
          types.TextContextParams | types.SoundContextParams | None
      ) = None,
      prompt: prompt_lib.Prompt | None = None,
  ) -> str:
    """Returns the prompt to be used for encoding."""
    if prompt is None:
      prompt = self._resolve_prompt(task_name)
    self._last_used_prompt = prompt
    prompt_template = _PROMPT_OVERRIDE.value or prompt.GetPromptTemplate()
    if self.normalizer is not None:
      text = self.normalizer(text)

    if prompt_template is None:
      return text

    if title is not None:
      title = title if self.normalizer is None else self.normalizer(title)
    else:
      title = 'None'
    if context is not None:
      context = context if self.normalizer is None else self.normalizer(context)
    else:
      context = 'None'
    text_prompt = prompt_template.format(
        text=text,
        title=title,
        context=context,
        task_name=task_name,
        context_params=context_params,
    )
    return text_prompt

  @final
  def _encode(
      self, batch: Sequence[types.MultiModalObject]
  ) -> Sequence[types.TextEmbedding | types.SoundEmbedding]:
    """Encodes a batch of text or audio sources."""
    prompt_batch = []
    resolved_prompts = []
    for example in batch:
      ctx_task_name = getattr(example.context, 'task_name', None)
      prompt = self._resolve_prompt(ctx_task_name)
      resolved_prompts.append(prompt)

      if isinstance(example, types.TextWithTitleAndContext):
        prompt_text = self._get_normalized_text_prompt(
            example.text,
            title=example.title_text,
            context=example.context_text,
            task_name=ctx_task_name,
            context_params=example.context,
            prompt=prompt,
        )
        prompt_audio = None
      elif isinstance(example, types.Text):
        prompt_text = self._get_normalized_text_prompt(
            example.text,
            task_name=ctx_task_name,
            context_params=example.context,
            prompt=prompt,
        )
        prompt_audio = None
      elif isinstance(example, types.SoundWithTitleAndContext):
        prompt_text = self._get_normalized_text_prompt(
            self.TEXT_FOR_AUDIO,
            title=example.title_text,
            context=example.context_text,
            task_name=ctx_task_name,
            context_params=example.context,
            prompt=prompt,
        )
        prompt_audio = utils.sound_to_wav_bytes(example)
      elif isinstance(example, types.Sound):
        prompt_text = self._get_normalized_text_prompt(
            self.TEXT_FOR_AUDIO,
            task_name=ctx_task_name,
            context_params=example.context,
            prompt=prompt,
        )
        prompt_audio = utils.sound_to_wav_bytes(example)
      else:
        raise ValueError('Unexpected input type.')
      prompt_batch.append((prompt_text, prompt_audio))

    assert self.prompt_encode_fn is not None
    response_batch = self.prompt_encode_fn(prompt_batch)

    embeddings_batch = []
    for response, prompt in zip(response_batch, resolved_prompts):
      embeddings_batch.append(prompt.ProcessResponse(response))

    outputs = []
    for embeddings, example, response, example_prompt in zip(
        embeddings_batch, batch, response_batch, prompt_batch
    ):
      if isinstance(response, str):
        debug_text = json.dumps({'prompt_text': example_prompt[0],
                                 'model_response': response})
      else:
        debug_text = None
      assert isinstance(example, types.Text) or isinstance(example, types.Sound)
      if isinstance(example, types.Text):
        outputs.append(
            types.TextEmbedding(
                embedding=np.expand_dims(embeddings, axis=0),
                spans=np.array([[0, len(example.text)]]),  # pyrefly: ignore[bad-argument-type]
                context=dataclasses.replace(
                    example.context, text=example.text, debug_text=debug_text
                ),
            )
        )
      if isinstance(example, types.Sound):
        outputs.append(
            types.SoundEmbedding(
                embedding=np.expand_dims(embeddings, axis=0),
                timestamps=np.array(  # pyrefly: ignore[bad-argument-type]
                    [[0, len(example.waveform) / example.context.sample_rate]]
                ),
                context=dataclasses.replace(
                    example.context,
                    debug_text=debug_text,
                ),
            )
        )
    return outputs


# For testing only.
class MockTextEncoder(TextEncoderWithPrompt):

  def _setup(self):
    self.prompt_encode_fn = lambda prompts: np.zeros((len(prompts), 3))
