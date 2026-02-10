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

"""Prompts for prompt-based encoders."""

import abc
import json
import logging
from typing import Any, Iterable, Mapping, Optional, Sequence

from mseb import types
from mseb.evaluators import classification_evaluator
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


def ValidateJson(
    response: str,
    expected_types: Mapping[str, Any],
) -> bool:
  """Validates the json response."""
  try:
    result = json.loads(response)
    for key, expected_type in expected_types.items():
      if key not in result:
        return False
      if not isinstance(result[key], type(expected_type)):
        return False
    return True
  except json.JSONDecodeError:
    logging.warning('Failed to parse json: %s', response)
  return False


class ReasoningPrompt(Prompt):
  """A prompt for the reasoning task."""

  NO_ANSWER_STR = reasoning_evaluator.NO_ANSWER_STR
  INVALID_ANSWER_STR = reasoning_evaluator.INVALID_ANSWER_STR
  NO_RESPONSE_STR = reasoning_evaluator.NO_RESPONSE_STR
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
    if response == self.NO_RESPONSE_STR:
      return self.NO_RESPONSE_STR
    return ProcessJsonResponse(
        response,
        keys=['answer'],
        key_to_value_map=dict({self.NO_ANSWER_STR: self.NO_ANSWER_STR}),
        invalid_response_value=self.INVALID_ANSWER_STR,
    )


class ClassificationPrompt(Prompt):
  """A prompt for the classification task."""

  PROMPT_TEMPLATE = """
**Task: Intent Classification**

**Goal:** Classify the provided text into one of the following intent classes:
  {class_labels}

**Input:** You will receive a query.
 * "query": The query being issued (string).

**Output:** You will produce a single JSON object as a plain text string (no markup). The structure depends on answerability:
 * "answer": (string) The intent class of the query.

**Important Considerations:**
* **Exact Matches:** The output should match exactly one of the intent class names. Do not rephrase or summarize.
* **No Other Output:** The output should only contain the intent class name.
* **Plain Text JSON Output:** The output must be a valid JSON string, but it must be a plain text string – no markup of any kind.

{{{{"query": {{text}}}}}}
"""
  INVALID_ANSWER_STR = classification_evaluator.INVALID_ANSWER_STR
  NO_RESPONSE_STR = classification_evaluator.NO_RESPONSE_STR

  def __init__(
      self, class_labels: Sequence[str], prompt_template: str = PROMPT_TEMPLATE
  ):
    self.class_labels = set(class_labels)
    self.prompt_template = prompt_template.format(
        class_labels=json.dumps(class_labels)
    )

  def GetPromptTemplate(self) -> str:
    return self.prompt_template

  def ProcessResponse(self, response: Any) -> Any:
    assert isinstance(response, str)
    if response == self.NO_RESPONSE_STR:
      return self.NO_RESPONSE_STR
    result = ProcessJsonResponse(
        response,
        keys=['answer'],
        key_to_value_map=None,
        invalid_response_value=self.INVALID_ANSWER_STR,
    )
    if result not in self.class_labels:
      return self.INVALID_ANSWER_STR
    return result


class SpeakerClassificationPrompt(ClassificationPrompt):
  """A prompt for the speaker classification tasks."""

  PROMPT_TEMPLATE = """
*Task: Speaker Classification**

**Goal:** Classify the speaker inthe provided audio clip into one of the following classes:
  {class_labels}

**Input:** You will receive an audio clip containing the recording of a person speaking {{text}}.

**Output:** You will produce a single JSON object as a plain text string (no markup) having the following structure:
 * "answer": (string) The class label of the speaker in the audio clip.

**Important Considerations:**
* **Exact Matches:** The ouput should match exactly one of the class names. Do not rephrase or summarize.
* **No Other Output:** The output should only contain the class name.
* **Plain Text JSON Output:** The output must be a valid JSON string, but it must be a plain text string – no markup of any kind.

{{{{"query": {{text}}}}}}
"""

  def __init__(
      self, class_labels: Sequence[str], prompt_template: str = PROMPT_TEMPLATE
  ):
    super().__init__(class_labels, prompt_template)


class SoundClassificationPrompt(Prompt):
  """A prompt for the sound classification tasks."""
  PROMPT_TEMPLATE = """
**Task: {Sound_Name} Classification**

**Goal:** Classify the provided audio clip into the following {sound_name} classes:
  {class_labels}

**Input:** You will receive an audio clip containing the recording of a {sound_name} {{text}}.

**Output:** For each {sound_name} class that correctly describes the audio clip, you will produce a single JSON object as a plain text string (no markup).
Up to {max_num_labels} JSON objects can be output.
Each JSON object should be produced on a new line.
Each JSON object should have the following structure:
 * "answer": (string) The {sound_name} class of the audio clip.

**Important Considerations:**
* Class names are descriptive.
* **Exact Matches:** Each output should match exactly one of the {sound_name} class names. Do not rephrase or summarize.
* **No Other Output:** Each output should only contain the {sound_name} class name.
* **Plain Text JSON Output:** Each output must be a valid JSON string, but it must be a plain text string – no markup of any kind.
"""
  INVALID_ANSWER_STR = reasoning_evaluator.INVALID_ANSWER_STR
  NO_RESPONSE_STR = reasoning_evaluator.NO_RESPONSE_STR

  def __init__(
      self, class_labels: Sequence[str],
      sound_name: str = 'Sound',
      prompt_template: str = PROMPT_TEMPLATE,
      max_num_labels: int = 5,
  ):
    self.class_labels = set(class_labels)
    self.prompt_template = prompt_template.format(
        Sound_Name=sound_name,
        sound_name=sound_name.lower(),
        class_labels=json.dumps(class_labels),
        max_num_labels=max_num_labels,
    )

  def GetPromptTemplate(self) -> str:
    return self.prompt_template

  def ProcessResponse(self, response: Any) -> Any:
    assert isinstance(response, str)
    if response == self.NO_RESPONSE_STR:
      return self.NO_RESPONSE_STR
    lines = response.split('\n')
    results = []
    for line in lines:
      result = ProcessJsonResponse(
          line,
          keys=['answer'],
          key_to_value_map=None,
          invalid_response_value=self.INVALID_ANSWER_STR,
      )
      if result in self.class_labels:
        results.append(result)
    if not results:
      return self.INVALID_ANSWER_STR
    return '\n'.join(results)


class RetrievalPrompt(Prompt):
  """A prompt for the retrieval task."""

  NO_RESPONSE_STR = reasoning_evaluator.NO_RESPONSE_STR
  INVALID_ANSWER_STR = reasoning_evaluator.INVALID_ANSWER_STR
  PROMPT_TEMPLATE = """
**Task: DocumentRetrieval**

**Goal:** Given a query, find the most relevant document from the provided documents.

*Input:** You will receive a query and a list of documents.
 * "query": The query being issued (string).
 * "documents": The list of documents. Each document is represented as a JSON object with the following fields:
  * "id": (string) The unique identifier of the document.
  * "text": (string) The text of the document.

*Output:** You will produce a list of document ids ordered from most to least relevant.

**Important Considerations:**
* Relevance should be determined based on the text of the document and the query.
* All documents should be considered: the ranklist produced should contain all document ids.
* **Exact Matches:** The output should contain document ids that match exactly ones provided.
* **No Other Output:** The output should only contain the ranked list of document ids.
* **Plain Text JSON Output:** The output must be a valid JSON string, but it must be a plain text string – no markup of any kind.

{{"query": {text}, "documents": {context}}}
"""

  def __init__(self, prompt_template: str = PROMPT_TEMPLATE):
    self.prompt_template = prompt_template

  def GetPromptTemplate(self) -> str:
    return self.prompt_template

  def ProcessResponse(self, response: Any) -> str:
    assert isinstance(response, str)
    if response != self.NO_RESPONSE_STR:
      try:
        result = json.loads(response)
        if not isinstance(result, list):
          raise ValueError('Result is not a list of strings: %s' % result)
        for item in result:
          if not isinstance(item, str):
            raise ValueError('Item is not a string: %s' % item)
        result = types.ValidListPrediction(
            [{'id': item} for item in result]
        )
      except (json.JSONDecodeError, ValueError):
        logging.warning('Invalid response format/type: %s', response)
        return types.InvalidAnswerListPrediction().to_json()
    else:
      return types.NoResponseListPrediction().to_json()
    return result.to_json()


class SegmentationPrompt(Prompt):
  """A prompt for the segmentation task."""

  PROMPT_TEMPLATE = """
**Task: Salient Term Segmentation**

**Goal:** Given a query, return the top-{top_k} most salient terms from the query.

**Input:** You will receive an audio query.

**Output:** For each of the top-{top_k} most salient terms in the query, you will produce a single JSON object as a plain text string (no markup).
Up to {top_k} JSON objects can be output, ordered from most to least salient.
Each JSON object should be produced on a new line.
Each JSON object should have the following structure:
 * "term": (string) The salient term from the query.
 * "start_time": (float) start time of the term in seconds.
 * "end_time": (float) end time of the term in seconds.

**Important Considerations:**
* A term must consist of just one word.
* A term is a salient term if it is a topic or a concept that is relevant to the query.
* A term is salient if it would appear in a relatively small number of wikipedia articles.
* **Plain Text JSON Output:** The output must be a valid JSON string, but it must be a plain text string – no markup of any kind.
"""
  INVALID_ANSWER_STR = reasoning_evaluator.INVALID_ANSWER_STR
  NO_RESPONSE_STR = reasoning_evaluator.NO_RESPONSE_STR

  def __init__(self, prompt_template: str = PROMPT_TEMPLATE, top_k: int = 3):
    self.prompt_template = prompt_template.format(top_k=top_k)

  def GetPromptTemplate(self) -> str:
    return self.prompt_template

  def ProcessResponse(self, response: Any) -> Any:
    assert isinstance(response, str)
    if response == self.NO_RESPONSE_STR:
      return self.NO_RESPONSE_STR
    lines = response.split('\n')
    results = []
    for line in lines:
      if ValidateJson(
          line,
          {'term': str(), 'start_time': float(), 'end_time': float()},
      ):
        result = json.loads(line)
        results.append(result)
    if not results:
      return self.INVALID_ANSWER_STR
    return json.dumps(results)


class SegmentationFromAlignmentPrompt(Prompt):
  """A prompt for the segmentation task."""

  PROMPT_TEMPLATE = """
**Task: Salient Term Segmentation**

**Goal:** Given a query, return the top-{top_k} most salient terms from the query.

**Input:** You will receive the time-aligned transcript of an audio query which has already been transcribed. In the following format:
  * "alignment": The time-aligned transcript of the query. This is a JSON-encoded list of objects, where each object has the following fields:
    * "text": (string) The word at this position in the transcript.
    * "start_time": (float) start time of that word in seconds in the original audio for the query.
    * "end_time": (float) end time of that word in seconds in the original audio for the query.

**Output:** For each of the top-{top_k} most salient terms in the query, you will produce a single JSON object as a plain text string (no markup).
Up to {top_k} JSON objects can be output, ordered from most to least salient.
Each JSON object should be produced on a new line.
Each JSON object should have the following structure:
 * "term": (string) The salient term from the query.
 * "start_time": (float) start time of the term in seconds.
 * "end_time": (float) end time of the term in seconds.

**Important Considerations:**
* A term must consist of just one word.
* A term is a salient term if it is a topic or a concept that is relevant to the query.
* A term is salient if it would appear in a relatively small number of wikipedia articles.
* **Plain Text JSON Output:** The output must be a valid JSON string, but it must be a plain text string – no markup of any kind.

{{{{"alignment": {{text}}}}}}
"""
  INVALID_ANSWER_STR = reasoning_evaluator.INVALID_ANSWER_STR
  NO_RESPONSE_STR = reasoning_evaluator.NO_RESPONSE_STR

  def __init__(self, prompt_template: str = PROMPT_TEMPLATE, top_k: int = 3):
    self.prompt_template = prompt_template.format(top_k=top_k)

  def GetPromptTemplate(self) -> str:
    return self.prompt_template

  def ProcessResponse(self, response: Any) -> Any:
    assert isinstance(response, str)
    if response == self.NO_RESPONSE_STR:
      return self.NO_RESPONSE_STR
    lines = response.split('\n')
    results = []
    for line in lines:
      if ValidateJson(
          line,
          {'term': str(), 'start_time': float(), 'end_time': float()},
      ):
        result = json.loads(line)
        results.append(result)
    if not results:
      return self.INVALID_ANSWER_STR
    return json.dumps(results)


class TranscriptionPrompt(Prompt):
  """A prompt for the transcription task."""

  NO_RESPONSE_STR = types.LLM_NO_RESPONSE_STR
  PROMPT_TEMPLATE = """
**Task: Speech Transcription**

**Goal:** Provide a high-fidelity transcription of the provided audio clip.

**Input:** You will receive an audio clip containing the recording of spoken text and optionally a contextual bias.

**Output:** You will produce a plain text string (no markup) containing the transcription of the audio clip .

**Important Considerations:**
* **No Other Output:** The output should only contain the transcription.
* **Plain Text Output:** The output must be a plain text string – no markup of any kind.

{{"contextual bias": {context}}}
"""

  def __init__(self, prompt_template: str = PROMPT_TEMPLATE):
    self.prompt_template = prompt_template

  def GetPromptTemplate(self) -> str:
    return self.prompt_template

  def ProcessResponse(self, response: Any) -> str:
    return response


class RerankingPrompt(Prompt):
  """A prompt for the reranking task."""

  NO_RESPONSE_STR = types.LLM_NO_RESPONSE_STR
  INVALID_ANSWER_STR = types.LLM_INVALID_ANSWER_STR
  PROMPT_TEMPLATE = """
**Task: Candidate Reranking**

**Goal:** Find the most accurate transcription of the query from the provided candidates.

*Input:** You will receive a query and a list of candidate transcriptions.
 * "query": The query being issued (string).
 * "candidates": The list of candidates. Each candidate is represented as a JSON object with the following fields:
  * "id": (string) The unique identifier of the candidate.
  * "text": (string) The text of the candidate.

*Output:** You will produce a list of candidate ids ordered from most to least accurate.

**Important Considerations:**
* Accuracteness should be determined based on the word error rate between the spoken query and the candidate transcription.
* All candidates should be considered: the ranklist produced should contain all candidate ids.
* **Exact Matches:** The output should contain candidate ids that match exactly ones provided.
* **No Other Output:** The output should only contain the ranked list of candidate ids.
* **Plain Text JSON Output:** The output must be a valid JSON string, but it must be a plain text string – no markup of any kind.
* **Example output:** "[0, 3, 1, 2]".

{{"query": {text}, "candidates": {context}}}
"""

  def __init__(self, prompt_template: str = PROMPT_TEMPLATE):
    self.prompt_template = prompt_template

  def GetPromptTemplate(self) -> str:
    return self.prompt_template

  def ProcessResponse(self, response: Any) -> str:
    assert isinstance(response, str)
    if response != self.NO_RESPONSE_STR:
      try:
        result = json.loads(response)
        if not isinstance(result, Iterable):
          raise ValueError('Result is not a list: %s' % result)
        if not result:
          raise ValueError('Result is empty')
        for item in result:
          try:
            _ = int(item)
          except (ValueError, TypeError) as exc:
            raise ValueError(
                'Item can not be converted to an integer: %s' % item
            ) from exc
        result = types.ValidListPrediction([{'id': item} for item in result])
      except (json.JSONDecodeError, ValueError) as exc:
        logging.warning('Invalid response format/type: %s', response)
        logging.warning('Error: %s', exc)
        return types.InvalidAnswerListPrediction().to_json()
    else:
      return types.NoResponseListPrediction().to_json()
    return result.to_json()
