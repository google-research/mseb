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

"""Prompts for prompt-based encoders."""

import abc
import json
import logging
from typing import Any, Mapping, Optional, Sequence

from mseb.evaluators import reasoning_evaluator


class Prompt(abc.ABC):
  """Base class for defining prompts for prompt-based encoders."""

  @abc.abstractmethod
  def GetPromptTemplate(self) -> Optional[str]:
    """Returns the prompt template."""

  @abc.abstractmethod
  def ProcessResponse(self, response: Any) -> Any:
    """Processes the response."""


class DefaultPrompt(Prompt):
  """A prompt that leaves the response unchanged."""

  def __init__(self, prompt_template: Optional[str] = None):
    self.prompt_template = prompt_template

  def GetPromptTemplate(self) -> Optional[str]:
    return self.prompt_template

  def ProcessResponse(self, response: Any) -> Any:
    return response


def ProcessJsonResponse(
    response: str,
    keys: Sequence[str],
    key_to_value_map: Optional[Mapping[str, str]],
    invalid_response_value: str,
) -> str:
  """Processes the json response."""
  try:
    result = json.loads(response)
    for key in keys:
      if key in result:
        return result[key]
    if key_to_value_map is not None:
      for key, value in key_to_value_map.items():
        if key in result:
          return value
  except json.JSONDecodeError:
    logging.warning('Failed to parse json: %s', response)
  return invalid_response_value


class ReasoningPrompt(Prompt):
  """A prompt for the reasoning task."""

  NO_ANSWER_STR = reasoning_evaluator.NO_ANSWER_STR
  INVALID_ANSWER_STR = reasoning_evaluator.INVALID_ANSWER_STR
  PROMPT_TEMPLATE = """
**Task: Answerability Determination and Exact Answer Extraction**

**Goal:** Determine if a question can be answered using the provided context and title, and if so, extract the **exact** answer verbatim from the text.

**Input:** You will receive a question, title and context each in a new line.
 * "question": The question being asked (string).
 * "title": The title of the document the context is from (string).
 * "context": A text passage which can either be a wiki page or a wiki paragraph that may or may not contain the answer.

**Output:** You will produce a single JSON object as a plain text string (no markup). The structure depends on answerability:

If the question IS answerable:
 * "rationale": (string) A concise explanation of why the provided answer is correct.  Be specific, referencing sentences or phrases.
 * "answer": (string) The answer to the question, copied  **exactly** from the title or context. Do not paraphrase or summarize. Prefer concise answers (shortest possible while complete).
If the question IS NOT answerable:
 * "{no_answer}": (string) A clear and concise explanation of why the question cannot be answered. Specify what information is missing.

**Important Considerations:**
* **Code change:**  The question may be in a language differ from context and title.  In that case, answer the question with the same language as context and title.
* **Exact Matches:**  Prioritize using the exact words within the provided text. Do not rephrase or summarize. Do not translate to English.
* **Specificity:** Be as specific as possible in your rationale and no_answer explanations.
* **Title and Context:** Consider both.
* **Direct Answers:** Only use the text from title or context. Do not infer or conclude.
* **Ambiguity:** If a question could have multiple different answers based on the context, the answer should be considered "{no_answer}" as the context is not specific enough. If multiple answers are equally supported and equally correct, select '{no_answer}'.
* **Plain Text JSON Output:** The output must be a valid JSON string, but it must be a plain text string – no markup of any kind.

{{{{"question": {{text}}, "title": {{title}}, "context": {{context}}}}}}
"""

  def __init__(self, prompt_template: str = PROMPT_TEMPLATE):
    self.prompt_template = prompt_template.format(no_answer=self.NO_ANSWER_STR)

  def GetPromptTemplate(self) -> str:
    return self.prompt_template

  def ProcessResponse(self, response: Any) -> Any:
    assert isinstance(response, str)
    return ProcessJsonResponse(
        response,
        keys=['answer'],
        key_to_value_map=dict({self.NO_ANSWER_STR: self.NO_ANSWER_STR}),
        invalid_response_value=self.INVALID_ANSWER_STR,
    )
