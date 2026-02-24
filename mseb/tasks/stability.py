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

"""Stability super task."""

import abc
import dataclasses
from typing import Dict, Iterable, List, Mapping, Optional

from mseb import metrics
from mseb import task as task_lib
from mseb import types
from mseb import utils
import numpy as np


def get_leaf_dist(
    r: types.MultiModalEmbedding,
    h: types.MultiModalEmbedding
) -> Mapping[str, Mapping[str, float]]:
  """Computes a suite of distance metrics between two leaf nodes.

  This function identifies the mathematical nature of the representation
  (Continuous vs. Discrete) based on its shape and calculates all applicable
  distance metrics. For continuous vector sequences, it returns CED, DTW,
  and L2. For discrete sequences or text, it returns UED.

  Args:
    r: The reference (clean) embedding node.
    h: The hypothesis (perturbed) embedding node.

  Returns:
    A mapping where keys are metric suffixes (e.g., 'CED', 'DTW') and
    values are dictionaries containing 'raw_distance' and 'reference_length'.
  """
  results = {}

  # 1. Handle Embeddings (Sound or Text modalities)
  if (isinstance(r, (types.SoundEmbedding, types.TextEmbedding)) and
      isinstance(h, (types.SoundEmbedding, types.TextEmbedding))):

    shape = r.embedding.shape

    # Continuous Suite (Rank 2: [Frames, Dimensions])
    if len(shape) == 2:
      # Main metric: Sequence-aware continuous edit distance
      results["CED"] = metrics.compute_continuous_edit_distance(
          r.embedding, h.embedding)
      # Temporal robustness: Non-linear alignment cost
      results["DTW"] = metrics.compute_dynamic_time_warping_distance(
          r.embedding, h.embedding)
      # Rigid alignment: Standard Euclidean distance (if shapes match)
      if r.embedding.shape == h.embedding.shape:
        results["L2"] = metrics.compute_lp_norm(r.embedding, h.embedding, p=2)

    # Discrete Suite (Rank 1: [Sequence_Length])
    elif len(shape) == 1:
      results["UED"] = metrics.compute_unit_edit_distance(
          r.embedding, h.embedding)

  # 2. Handle Transcription/Text Predictions (Always Discrete)
  elif (isinstance(r, types.TextPrediction) and
        isinstance(h, types.TextPrediction)):
    results["UED"] = metrics.compute_unit_edit_distance(
        r.prediction.split(), h.prediction.split()
    )

  return results


def _calculate_all_distances(
    ref: types.MultiModalEmbedding,
    hyp: types.MultiModalEmbedding
) -> Mapping[str, Mapping[str, float]]:
  """Traverses multi-modal representations to aggregate all applicable metrics.

  If the inputs are collections, this function iterates through matching keys
  to ensure every representation (e.g., different model layers) is evaluated.

  Args:
    ref: The reference (clean) representation.
    hyp: The hypothesis (perturbed) representation.

  Returns:
    A mapping of unique metric names (e.g., 'layer1_CED', 'UED') to
    their respective distance statistics.
  """
  all_metrics = {}

  if (isinstance(ref, types.SoundEmbeddingCollection) and
      isinstance(hyp, types.SoundEmbeddingCollection)):
    for key in ref.embeddings:
      if key in hyp.embeddings:
        # Extract metrics for this specific leaf
        leaf_results = get_leaf_dist(ref.embeddings[key], hyp.embeddings[key])
        for m_suffix, stats in leaf_results.items():
          # Namespace the metric by the collection key
          all_metrics[f"{key}_{m_suffix}"] = stats
  else:
    # Base case: Comparing individual leaf nodes
    leaf_results = get_leaf_dist(ref, hyp)
    for m_suffix, stats in leaf_results.items():
      all_metrics[m_suffix] = stats

  return all_metrics


class StabilityTask(task_lib.MSEBTask, abc.ABC):
  """Base class for profiling representational stability.

  This task profiles the invariance of an encoder by comparing clean audio
  embeddings to augmented variants. It measures how much the representational
  geometry shifts under noise (Drift) and how consistently the model responds
  to different noise samples (Instability).

  The task reports two primary scores per metric:
    1. Corpus_Mean: A global Micro-Average drift rate. It is calculated as the
       sum of all edit costs across the corpus divided by the sum of maximum
       possible costs (2*L for continuous, L for discrete).
    2. Mean_Local_IS: The Macro-Average of per-sample drift, where the standard
       deviation ('std') represents the Instability Score (IS). This identifies
       models that are unpredictable in their response to perturbation.
  """

  def __init__(
      self,
      num_augmentations: int = 5,
      config: Optional[utils.SpecAugmentConfig] = None,
  ):
    """Initializes the Stability Task.

    Args:
      num_augmentations: Number of perturbed variants to generate per sound.
      config: Configuration for SpecAugment (masking/warping parameters).
    """
    super().__init__()
    self.num_augmentations = num_augmentations
    self.config = config or utils.SpecAugmentConfig()

  @abc.abstractmethod
  def base_sounds(self) -> Iterable[types.Sound]:
    """Yields clean reference sounds from the dataset.

    Subclasses must implement this to define the evaluation corpus.
    """
    ...

  def sounds(self) -> Iterable[types.Sound]:
    """Yields Clean reference followed by N Augmented variants.

    Uses deterministic seeding based on the sound ID to ensure reproducibility
    across different models and runs.
    """
    for sound in self.base_sounds():
      # 1. Yield Clean reference
      yield sound

      # 2. Yield N Augmented variants
      for i in range(self.num_augmentations):
        seed = utils.get_deterministic_seed(sound.context.id, i)
        rng = np.random.default_rng(seed)
        aug_waveform = utils.apply_specaugment_to_waveform(
            sound.waveform, config=self.config, rng=rng
        )
        new_ctx = dataclasses.replace(
            sound.context, id=f"{sound.context.id}_aug_{i}"
        )
        yield types.Sound(waveform=aug_waveform, context=new_ctx)

  def compute_scores(
      self, embeddings: types.MultiModalEmbeddingCache
  ) -> dict[str, list[types.Score]]:
    """Calculates stability metrics using Micro and Macro averaging.

    This method implements a Micro-Average (sum-over-sum) normalization
    strategy to ensure that longer utterances contribute proportionally
    to the global error profile.

    Args:
      embeddings: The cache containing model outputs for clean and noisy audio.

    Returns:
      A dictionary mapping the 'stability' key to a list of MSEB Scores.
    """
    total_abs_dist: Dict[str, float] = {}
    total_ref_len: Dict[str, float] = {}
    local_normalized_drifts: Dict[str, List[float]] = {}

    for sound in self.base_sounds():
      base_id = sound.context.id
      clean = embeddings.get(base_id)
      if clean is None: continue

      # Collect all variants for this sample
      variants = [
          embeddings.get(f"{base_id}_aug_{i}")
          for i in range(self.num_augmentations)
      ]
      variants = [v for v in variants if v is not None]

      for v in variants:
        abs_results = _calculate_all_distances(clean, v)
        for m_name, res in abs_results.items():
          # Identify base metric type for normalization factor
          base_m = m_name.split("_")[-1]

          # Pull distance and reference length from metric results
          dist = res.get("raw_distance")
          ref_len = res.get("reference_length", 0.0)

          if dist is None or ref_len == 0: continue
          # 1. Accumulate for Micro-Average
          total_abs_dist[m_name] = total_abs_dist.get(m_name, 0.0) + dist
          total_ref_len[m_name] = total_ref_len.get(m_name, 0.0) + ref_len
          # 2. Accumulate for Macro-Average/IS
          # CED distance is bounded by 2.0 per frame on unit sphere
          norm_factor = 2.0 if base_m == "CED" else 1.0
          norm_drift = dist / (norm_factor * ref_len)
          local_normalized_drifts.setdefault(m_name, []).append(norm_drift)

    return {"stability": self._format_results(
        total_abs_dist, total_ref_len, local_normalized_drifts)}

  def _format_results(
      self,
      total_dist: Dict[str, float],
      total_len: Dict[str, float],
      local_drifts: Dict[str, List[float]]
  ) -> List[types.Score]:
    """Formats raw statistics into MSEB Score objects.

    Calculates both the Micro-averaged mean and the Macro-averaged
    Instability Score (IS).

    Args:
      total_dist: Mapping of metric name to sum of distances.
      total_len: Mapping of metric name to sum of reference lengths.
      local_drifts: Mapping of metric name to list of normalized drift values.

    Returns:
      A list of formatted MSEB Score objects.
    """
    final = []
    metric_meta = {
        "CED": (
            "Continuous Edit Distance. Measures the cost of transforming one"
            "vector sequence into another via insertions, deletions, and"
            "substitutions. Normalized by 2*L assuming a unit sphere where"
            "max L2 distance is 2.0."
        ),
        "UED": (
            "Unit Edit Distance. The normalized Levenshtein distance between"
            "discrete units (indices or words). Indicates categorical or"
            "semantic invariance. Normalized by L."
        ),
        "DTW": (
            "Dynamic Time Warping. Computes the optimal non-linear alignment"
            "cost between two sequences, measuring stability against temporal"
            "warping or stretching."
        ),
        "L2": (
            "Euclidean Distance (L2). A rigid, point-to-point distance metric."
            "Highly sensitive to precise temporal alignment and magnitude"
            "shifts."
        )
    }

    for m in sorted(total_dist.keys()):
      base_m = m.split("_")[-1]
      norm_factor = 2.0 if base_m == "CED" else 1.0
      description_base = metric_meta.get(base_m, f"Stability metric for {m}.")

      # 1. Global Micro-Average Drift Rate (Corpus Profile)
      global_drift = total_dist[m] / (norm_factor * total_len[m])
      final.append(types.Score(
          metric=f"Corpus_Mean_{m}",
          value=float(global_drift),
          description=(
              f"{description_base} Global Micro-Average"
              "across the entire corpus."
          ),
          min=0.0, max=1.0 if base_m in ["CED", "UED"] else float("inf")
      ))

      # 2. Mean Local Drift & Instability Score (Reliability Profile)
      local_mean = np.mean(local_drifts[m]) if local_drifts[m] else 0.0
      local_is = np.std(local_drifts[m]) if local_drifts[m] else 0.0
      final.append(types.Score(
          metric=f"Mean_Local_IS_{m}",
          value=float(local_mean),
          std=float(local_is),
          description=(
              f"Average local drift for {base_m}. The 'std' field"
              "represents the Instability Score (IS), measuring the"
              "predictability of model drift."
          ),
          min=0.0, max=1.0 if base_m in ["CED", "UED"] else float("inf")
      ))

    return final
