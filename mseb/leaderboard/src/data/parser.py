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

"""MSEB Evaluation Results JSONL Parser and Model Discovery Engine.

This module provides functions to crawl the filesystem, discover evaluated model
configurations, parse JSONL evaluation files, deserialize records into typed
dataclasses, handle noise condition slices and multi-target clustering
consolidation,
and load the entire evaluation corpus into memory.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from src.config import CLUSTERING_TARGETS
from src.config import resolve_results_dir
from src.config import RESOURCE_METRIC_KEYS
from src.config import TASK_PRIMARY_METRICS
from src.data.models import EvaluationRecord
from src.data.models import ModelEntry
from src.data.models import TaskMetadata

logger = logging.getLogger(__name__)

# Canonical mapping from raw task subtype / type strings to standardized task
# keys.
TASK_TYPE_TO_CANONICAL: Dict[str, str] = {
    # Lowercase names
    "classification": "classification",
    "clustering": "clustering",
    "reasoning": "reasoning",
    "reranking": "reranking",
    "retrieval": "retrieval",
    "segmentation": "segmentation",
    "stability": "stability",
    "transcription": "transcription",
    "reconstruction": "reconstruction",
    "speech_transcription": "transcription",
    "speechtranscription": "transcription",
    "asr": "transcription",
    "span_reasoning": "reasoning",
    "spaninlangreasoning": "reasoning",
    "spancrosslangreasoning": "reasoning",
    "query_reranking": "reranking",
    "queryreranking": "reranking",
    "document_retrieval": "retrieval",
    "documentinlangretrieval": "retrieval",
    "documentcrosslangretrieval": "retrieval",
    "passage_retrieval": "retrieval",
    "passageinlangretrieval": "retrieval",
    "passagecrosslangretrieval": "retrieval",
    "speech_retrieval": "retrieval",
    "speechretrieval": "retrieval",
    "image_retrieval": "retrieval",
    "imageretrieval": "retrieval",
    "intent_classification": "classification",
    "intentclassification": "classification",
    "speaker_gender_classification": "classification",
    "speakergenderclassification": "classification",
    "salient_term_segmentation": "segmentation",
    "salienttermsegmentation": "segmentation",
    "continuous_stability": "stability",
    "continuousstability": "stability",
    "discrete_stability": "stability",
    "discretestability": "stability",
    "audio_reconstruction": "reconstruction",
    "audioreconstruction": "reconstruction",
    # PascalCase task types from task_metadata.type
    "Classification": "classification",
    "IntentClassification": "classification",
    "SpeakerGenderClassification": "classification",
    "Clustering": "clustering",
    "Reasoning": "reasoning",
    "SpanInLangReasoning": "reasoning",
    "SpanCrossLangReasoning": "reasoning",
    "Reranking": "reranking",
    "QueryReranking": "reranking",
    "Retrieval": "retrieval",
    "DocumentInLangRetrieval": "retrieval",
    "DocumentCrossLangRetrieval": "retrieval",
    "PassageInLangRetrieval": "retrieval",
    "PassageCrossLangRetrieval": "retrieval",
    "ImageRetrieval": "retrieval",
    "SpeechRetrieval": "retrieval",
    "Segmentation": "segmentation",
    "SalientTermSegmentation": "segmentation",
    "Stability": "stability",
    "ContinuousStability": "stability",
    "DiscreteStability": "stability",
    "Transcription": "transcription",
    "SpeechTranscription": "transcription",
    "Reconstruction": "reconstruction",
    "AudioReconstruction": "reconstruction",
}

TASK_DEFAULT_METRICS: Dict[str, str] = TASK_PRIMARY_METRICS


def resolve_task_name(raw_type: Any, dataset_name: Any = "") -> str:
  """Normalizes raw task type string or dataset name into one of 9 canonical MSEB tasks."""
  if isinstance(raw_type, (list, tuple)) and raw_type:
    raw_type = raw_type[0]
  raw_str = str(raw_type or "")
  raw_lower = raw_str.lower().strip()
  if raw_lower in TASK_TYPE_TO_CANONICAL:
    return TASK_TYPE_TO_CANONICAL[raw_lower]

  for key, val in TASK_TYPE_TO_CANONICAL.items():
    if key in raw_lower:
      return val

  # Fallback to dataset name
  d_str = str(dataset_name or "")
  d_lower = d_str.lower().strip()
  for key, val in TASK_TYPE_TO_CANONICAL.items():
    if key in d_lower:
      return val

  return "classification"


def _resolve_model_metadata(
    dir_path: str, results_dir: str, cache: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
  """Traverses upward from dir_path to results_dir to resolve and cache model.json metadata."""
  curr = os.path.abspath(dir_path)
  base_boundary = os.path.abspath(results_dir)
  traversed: List[str] = []

  while True:
    if curr in cache:
      meta = cache[curr]
      for p in traversed:
        cache[p] = meta
      return meta

    model_json_path = os.path.join(curr, "model.json")
    if os.path.isfile(model_json_path):
      try:
        with open(
            model_json_path, "r", encoding="utf-8", errors="replace"
        ) as f:
          data = json.load(f)
          if isinstance(data, dict):
            cache[curr] = data
            for p in traversed:
              cache[p] = data
            return data
      except (json.JSONDecodeError, OSError) as e:
        logger.warning(
            "Failed to read or parse metadata file %s: %s", model_json_path, e
        )
        cache[curr] = {}
        for p in traversed:
          cache[p] = {}
        return {}

    traversed.append(curr)
    if curr == base_boundary or os.path.dirname(curr) == curr:
      break
    curr = os.path.dirname(curr)

  for p in traversed:
    cache[p] = {}
  return {}


def discover_models(results_dir: Optional[str] = None) -> List[ModelEntry]:
  """Recursively crawls results_dir to discover all model evaluation configurations.

  Identifies top-level model folders and nested subconfiguration folders
  (e.g., asr=truth, asr=whisper) containing at least one .jsonl evaluation file.
  Inherits model metadata (name, url) from parent model.json when not present
  locally.

  Args:
      results_dir: Path to root evaluation results directory.

  Returns:
      Sorted list of ModelEntry objects for each valid evaluation configuration.
  """
  resolved_path = resolve_results_dir(results_dir)
  if (
      not resolved_path
      or not os.path.exists(resolved_path)
      or not os.path.isdir(resolved_path)
  ):
    logger.warning(
        "Results directory does not exist or is not a directory: %s",
        resolved_path,
    )
    return []

  abs_results_dir = os.path.abspath(resolved_path)
  metadata_cache: Dict[str, Dict[str, Any]] = {}
  entries: List[ModelEntry] = []

  try:
    for root, dirs, files in os.walk(abs_results_dir, followlinks=True):
      # Prune hidden directories in place
      dirs[:] = [d for d in dirs if not d.startswith(".")]

      # Filter for non-hidden .jsonl evaluation files
      jsonl_files = sorted(
          [f for f in files if f.endswith(".jsonl") and not f.startswith(".")]
      )
      if not jsonl_files:
        continue

      rel_path = os.path.relpath(root, abs_results_dir).replace(os.sep, "/")
      if rel_path == "." or not rel_path:
        # Loose jsonl files directly in results_dir root without model namespace
        continue

      parts = [p for p in rel_path.split("/") if p and p != "."]
      if not parts:
        continue

      top_model = parts[0]
      sub_config = "/".join(parts[1:]) if len(parts) > 1 else None

      meta = _resolve_model_metadata(root, abs_results_dir, metadata_cache)
      if not isinstance(meta, dict):
        meta = {}
      base_name = str(meta.get("name") or top_model)
      url = str(meta.get("url") or "")

      display_name = f"{base_name} ({sub_config})" if sub_config else base_name

      entries.append(
          ModelEntry(
              entry_id=rel_path,
              top_model=top_model,
              sub_config=sub_config,
              display_name=display_name,
              url=url,
              dir_path=root,
              jsonl_files=list(jsonl_files),
              tags=[sub_config] if sub_config else [],
              metadata=meta,
          )
      )
  except OSError as e:
    logger.error("Filesystem traversal error in %s: %s", abs_results_dir, e)

  # Deterministic sorting: top_model ascending, then sub_config
  entries.sort(key=lambda e: (e.top_model.lower(), e.sub_config or ""))
  return entries


def _extract_metric_score(
    scores_list: Any, target_metric: str
) -> Tuple[str, Optional[float]]:
  """Extracts the numeric score for target_metric from a scores list."""
  if not isinstance(scores_list, list):
    return target_metric, None

  s_map: Dict[str, float] = {}
  for s in scores_list:
    if isinstance(s, dict) and "metric" in s and s.get("value") is not None:
      m_key = str(s["metric"])
      val = s["value"]
      if isinstance(val, str) and val.lower() in (
          "infinity",
          "-infinity",
          "inf",
          "-inf",
      ):
        continue
      try:
        s_map[m_key] = float(val)
      except (ValueError, TypeError):
        continue

  # 1. Exact match
  if target_metric in s_map:
    return target_metric, s_map[target_metric]

  # 2. Case-insensitive match
  for m_key, m_val in s_map.items():
    if m_key.lower() == target_metric.lower():
      return m_key, m_val

  # 3. Fallback to first non-resource metric
  for m_key, m_val in s_map.items():
    if m_key.lower() not in RESOURCE_METRIC_KEYS:
      return m_key, m_val

  # 4. Any numeric metric
  if s_map:
    first_k = next(iter(s_map.keys()))
    return first_k, s_map[first_k]

  return target_metric, None


def _parse_scores_dict(scores_list: Any) -> Dict[str, float]:
  """Parses scores list into a clean dictionary of metric_name -> float."""
  all_s: Dict[str, float] = {}
  if not isinstance(scores_list, list):
    return all_s
  for s in scores_list:
    if isinstance(s, dict) and "metric" in s and s.get("value") is not None:
      m_key = str(s["metric"])
      val = s["value"]
      if isinstance(val, str) and val.lower() in (
          "infinity",
          "-infinity",
          "inf",
          "-inf",
      ):
        continue
      try:
        all_s[m_key] = float(val)
      except (ValueError, TypeError):
        pass
  return all_s


def parse_jsonl_file(
    file_path: str, model_entry: Optional[ModelEntry] = None
) -> List[EvaluationRecord]:
  """Parses a single JSONL evaluation results file into a list of EvaluationRecords.

  Handles:
  - Graceful skipping of empty lines, corrupt JSON lines, or binary corrupted
  files.
  - Multi-target clustering datasets (averaging VMeasure across 3 label
  targets).
  - Multi-line noise suites (clean, media_noise, traffic_noise,
  background_speech).
  - Repeated run deduplication (retains the latest run per sub_task_name).
  - String-encoded numbers, null values, and Infinity / NaN.

  Args:
      file_path: Absolute or relative path to the .jsonl file.
      model_entry: Associated ModelEntry instance (or generated placeholder if
        None).

  Returns:
      List of EvaluationRecord objects.
  """
  if model_entry is None:
    model_entry = ModelEntry(
        entry_id="unknown",
        top_model="unknown",
        sub_config=None,
        display_name="Unknown Model",
        url="",
        dir_path=os.path.dirname(file_path) if file_path else "",
        jsonl_files=[os.path.basename(file_path)] if file_path else [],
    )

  if not file_path or not os.path.exists(file_path):
    return []

  try:
    if os.path.getsize(file_path) == 0:
      return []
  except OSError:
    return []

  # Read and parse lines with errors="replace"
  raw_lines: List[Dict[str, Any]] = []
  try:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
      for line in f:
        line_str = line.strip()
        if not line_str:
          continue
        try:
          data = json.loads(line_str)
          if isinstance(data, dict):
            raw_lines.append(data)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
          logger.warning(
              "Skipping corrupt JSON line in %s: %s", file_path, str(e)
          )
          continue
  except (UnicodeDecodeError, OSError) as e:
    logger.warning("Failed to read JSONL file %s: %s", file_path, e)
    return []

  if not raw_lines:
    return []

  # Group lines by dataset name
  # In almost all cases, all lines in a file belong to the same dataset.
  datasets_map: Dict[str, Dict[str, Dict[str, Any]]] = {}
  for data in raw_lines:
    if not isinstance(data, dict):
      continue
    tm = data.get("task_metadata")
    if not isinstance(tm, dict):
      tm = {}
    raw_dname = tm.get("name")
    dname = (
        str(raw_dname)
        if raw_dname
        else os.path.splitext(os.path.basename(file_path))[0]
    )
    raw_st = data.get("sub_task_name")
    st = str(raw_st) if raw_st is not None else "default"
    if not st:
      st = "default"
    if dname not in datasets_map:
      datasets_map[dname] = {}
    # Deduplication: sequential lines overwrite earlier ones for same
    # sub_task_name
    datasets_map[dname][st] = data

  results: List[EvaluationRecord] = []

  for dname, subtask_dict in datasets_map.items():
    subtask_keys = set(subtask_dict.keys())

    # Case 1: 3-line SVQ Speaker Clustering (speaker_gender, speaker_age,
    # speaker_id)
    if len(subtask_dict) == 3 and subtask_keys == CLUSTERING_TARGETS:
      first_raw = list(subtask_dict.values())[0]
      raw_tm = first_raw.get("task_metadata")
      tm_dict = raw_tm if isinstance(raw_tm, dict) else {}
      task_meta = TaskMetadata.from_dict(tm_dict) if tm_dict else None
      task_name = "clustering"
      main_metric = str(tm_dict.get("main_score") or "VMeasure")

      target_scores: Dict[str, float] = {}
      target_vals: List[float] = []
      combined_all_scores: Dict[str, List[float]] = {}

      for st, raw in subtask_dict.items():
        scores_list = raw.get("scores")
        if not isinstance(scores_list, list):
          scores_list = []
        _, v = _extract_metric_score(scores_list, main_metric)
        if v is not None:
          target_scores[st] = v
          target_vals.append(v)

        for s in scores_list:
          if (
              isinstance(s, dict)
              and "metric" in s
              and s.get("value") is not None
          ):
            val = s["value"]
            if not (
                isinstance(val, str)
                and val.lower() in ("infinity", "-infinity", "inf", "-inf")
            ):
              try:
                combined_all_scores.setdefault(str(s["metric"]), []).append(
                    float(val)
                )
              except (ValueError, TypeError):
                pass

      mean_score = sum(target_vals) / len(target_vals) if target_vals else 0.0
      all_s = {
          m: sum(vl) / len(vl) for m, vl in combined_all_scores.items() if vl
      }
      all_s[main_metric] = mean_score

      raw_splits = tm_dict.get("eval_splits")
      eval_splits = (
          [str(x) for x in raw_splits if x is not None]
          if isinstance(raw_splits, list)
          else []
      )
      raw_langs = tm_dict.get("eval_langs")
      eval_langs = (
          [str(x) for x in raw_langs if x is not None]
          if isinstance(raw_langs, list)
          else []
      )
      raw_tags = first_raw.get("tags")
      tags = (
          [str(x) for x in raw_tags if x is not None]
          if isinstance(raw_tags, list)
          else []
      )

      results.append(
          EvaluationRecord(
              model_entry=model_entry,
              task_name=task_name,
              dataset_name=dname,
              sub_task_name="speaker_clustering",
              is_svq=dname.startswith("SVQ"),
              main_score_name=main_metric,
              main_score_value=mean_score,
              all_scores=all_s,
              condition_scores=target_scores,
              is_noise_slice=False,
              noise_condition=None,
              raw_record_name=str(first_raw.get("name"))
              if first_raw.get("name") is not None
              else None,
              tags=tags,
              prompt=str(first_raw.get("prompt"))
              if first_raw.get("prompt") is not None
              else None,
              task_metadata=task_meta,
              eval_splits=eval_splits,
              eval_langs=eval_langs,
              source_file=file_path,
          )
      )
      continue

    # Case 2: Noise Suite (contains colon noise conditions like :clean,
    # :media_noise, etc.)
    has_colon_noise = any(":" in str(st) for st in subtask_keys)
    if has_colon_noise and len(subtask_dict) > 1:
      primary_raw: Optional[Dict[str, Any]] = None
      condition_scores: Dict[str, float] = {}

      for st, raw in subtask_dict.items():
        raw_tm = raw.get("task_metadata")
        tm_dict = raw_tm if isinstance(raw_tm, dict) else {}
        stypes = tm_dict.get("task_subtypes")
        raw_type = tm_dict.get("type", "")
        task_type_arg = (
            stypes[0] if isinstance(stypes, list) and stypes else raw_type
        )
        task_name = resolve_task_name(task_type_arg, dname)
        main_metric = str(
            tm_dict.get("main_score")
            or TASK_DEFAULT_METRICS.get(task_name, "Accuracy")
        )

        scores_list = raw.get("scores")
        if not isinstance(scores_list, list):
          scores_list = []
        _, val = _extract_metric_score(scores_list, main_metric)

        if ":" in str(st):
          cond = str(st).split(":", 1)[1]
          if val is not None:
            condition_scores[cond] = val
        else:
          primary_raw = raw

      if primary_raw is None:
        primary_raw = list(subtask_dict.values())[0]

      raw_tm = primary_raw.get("task_metadata")
      tm_dict = raw_tm if isinstance(raw_tm, dict) else {}
      task_meta = TaskMetadata.from_dict(tm_dict) if tm_dict else None
      stypes = tm_dict.get("task_subtypes")
      raw_type = tm_dict.get("type", "")
      task_type_arg = (
          stypes[0] if isinstance(stypes, list) and stypes else raw_type
      )
      task_name = resolve_task_name(task_type_arg, dname)
      main_metric = str(
          tm_dict.get("main_score")
          or TASK_DEFAULT_METRICS.get(task_name, "Accuracy")
      )

      scores_list = primary_raw.get("scores")
      all_s = _parse_scores_dict(scores_list)
      matched_metric, main_val = _extract_metric_score(scores_list, main_metric)

      raw_splits = tm_dict.get("eval_splits")
      eval_splits = (
          [str(x) for x in raw_splits if x is not None]
          if isinstance(raw_splits, list)
          else []
      )
      raw_langs = tm_dict.get("eval_langs")
      eval_langs = (
          [str(x) for x in raw_langs if x is not None]
          if isinstance(raw_langs, list)
          else []
      )
      raw_tags = primary_raw.get("tags")
      tags = (
          [str(x) for x in raw_tags if x is not None]
          if isinstance(raw_tags, list)
          else []
      )
      raw_st = primary_raw.get("sub_task_name")
      sub_task_name = str(raw_st) if raw_st is not None else "default"

      results.append(
          EvaluationRecord(
              model_entry=model_entry,
              task_name=task_name,
              dataset_name=dname,
              sub_task_name=sub_task_name,
              is_svq=dname.startswith("SVQ"),
              main_score_name=matched_metric,
              main_score_value=float(main_val) if main_val is not None else 0.0,
              all_scores=all_s,
              condition_scores=condition_scores,
              is_noise_slice=False,
              noise_condition=None,
              raw_record_name=str(primary_raw.get("name"))
              if primary_raw.get("name") is not None
              else None,
              tags=tags,
              prompt=str(primary_raw.get("prompt"))
              if primary_raw.get("prompt") is not None
              else None,
              task_metadata=task_meta,
              eval_splits=eval_splits,
              eval_langs=eval_langs,
              source_file=file_path,
          )
      )
      continue

    # Case 3: Standard single records or separate distinct subtasks
    for st, raw in subtask_dict.items():
      raw_tm = raw.get("task_metadata")
      tm_dict = raw_tm if isinstance(raw_tm, dict) else {}
      task_meta = TaskMetadata.from_dict(tm_dict) if tm_dict else None
      stypes = tm_dict.get("task_subtypes")
      raw_type = tm_dict.get("type", "")
      task_type_arg = (
          stypes[0] if isinstance(stypes, list) and stypes else raw_type
      )
      task_name = resolve_task_name(task_type_arg, dname)
      main_metric = str(
          tm_dict.get("main_score")
          or TASK_DEFAULT_METRICS.get(task_name, "Accuracy")
      )

      scores_list = raw.get("scores")
      all_s = _parse_scores_dict(scores_list)
      matched_metric, main_val = _extract_metric_score(scores_list, main_metric)

      is_noise = ":" in str(st)
      noise_cond = str(st).split(":", 1)[1] if is_noise else None

      raw_splits = tm_dict.get("eval_splits")
      eval_splits = (
          [str(x) for x in raw_splits if x is not None]
          if isinstance(raw_splits, list)
          else []
      )
      raw_langs = tm_dict.get("eval_langs")
      eval_langs = (
          [str(x) for x in raw_langs if x is not None]
          if isinstance(raw_langs, list)
          else []
      )
      raw_tags = raw.get("tags")
      tags = (
          [str(x) for x in raw_tags if x is not None]
          if isinstance(raw_tags, list)
          else []
      )

      results.append(
          EvaluationRecord(
              model_entry=model_entry,
              task_name=task_name,
              dataset_name=dname,
              sub_task_name=str(st),
              is_svq=dname.startswith("SVQ"),
              main_score_name=matched_metric,
              main_score_value=float(main_val) if main_val is not None else 0.0,
              all_scores=all_s,
              condition_scores={},
              is_noise_slice=is_noise,
              noise_condition=noise_cond,
              raw_record_name=str(raw.get("name"))
              if raw.get("name") is not None
              else None,
              tags=tags,
              prompt=str(raw.get("prompt"))
              if raw.get("prompt") is not None
              else None,
              task_metadata=task_meta,
              eval_splits=eval_splits,
              eval_langs=eval_langs,
              source_file=file_path,
          )
      )

  return results


def load_all_evaluation_data(
    results_dir: Optional[str] = None,
) -> List[EvaluationRecord]:
  """Discovers all models and parses all JSONL evaluation files across results_dir.

  Args:
      results_dir: Path to third_party/py/mseb/results/google/ directory.

  Returns:
      List of all parsed EvaluationRecord instances.
  """
  models = discover_models(results_dir)
  all_records: List[EvaluationRecord] = []

  for model in models:
    for filename in model.jsonl_files:
      file_path = os.path.join(model.dir_path, filename)
      records = parse_jsonl_file(file_path, model)
      all_records.extend(records)

  logger.info(
      "Loaded %d evaluation records across %d model configurations.",
      len(all_records),
      len(models),
  )
  return all_records
