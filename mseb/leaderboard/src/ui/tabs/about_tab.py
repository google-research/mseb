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

"""About & Documentation Tab for the MSEB Gradio Leaderboard."""

from __future__ import annotations

import gradio as gr

CITATION_BUTTON_TEXT = r"""
@inproceedings{heigold2025mseb,
  title={{Massive Sound Embedding Benchmark (MSEB)}},
  author={Heigold, Georg and Variani, Ehsan and Bagby, Tom and Allauzen, Cyril and Ma, Ji and Kumar, Shankar and Riley, Michael},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
"""

ABOUT_MARKDOWN = """
# About Massive Sound Embedding Benchmark (MSEB)

The **Massive Sound Embedding Benchmark (MSEB)** is a unified, multilingual benchmark designed to rigorously evaluate the capabilities of speech, audio, and multimodal foundation models. MSEB standardizes evaluation across 9 distinct speech and acoustic tasks comprising over 275 datasets in dozens of languages and acoustic domains.

---

## MSEB Tasks

### 1. 🏷️ Classification (⬆️ Higher is better)
- **Objective:** Categorize spoken utterances or acoustic event recordings into target discrete classes (single-label multi-class or multi-label).
- **Primary Metric:** `Accuracy` (for multi-class) / `mAP` (Mean Average Precision for multi-label).
- **Datasets:** SVQ Multilingual Speaker Gender Classification (26 locales), Speech-MASSIVE Intent & Gender Classification (24 datasets across 12 languages), and Bioacoustic / Sound Event Classification (BirdSet HSN/NBP/POW, FSD50K).

### 2. 👥 Clustering (⬆️ Higher is better)
- **Objective:** Unsupervised grouping of acoustic embedding representations to discover latent speaker identities or acoustic clusters using MiniBatch K-Means.
- **Primary Metric:** `V-Measure` (harmonic mean of Homogeneity and Completeness).
- **Datasets:** SVQ Multilingual Speaker Clustering (26 locales) and Bioacoustic / Sound Event Clustering (BirdSet, FSD50K).

### 3. 🧠 Reasoning (⬆️ Higher is better)
- **Objective:** Spoken question answering and contextual text span identification given spoken audio inputs.
- **Primary Metric:** `GmeanF1` (Geometric Mean F1 score balancing answer vs no-answer questions: $\\sqrt{\\text{F1}_{\\text{No Answer}} \\cdot \\text{F1}_{\\text{Answer}}}$).
- **Datasets:** SVQ In-Language Span Reasoning (17 locales) and SVQ Cross-Language Span Reasoning (10 locales).

### 4. 🔀 Reranking (⬆️ Higher is better)
- **Objective:** Re-rank candidate textual hypotheses/transcripts given a spoken audio embedding to place the true transcript at rank 1.
- **Primary Metric:** `mAP` / `MAP` (Mean Average Precision) and `MRR`.
- **Datasets:** SVQ Multilingual Query Reranking across 26 languages and dialects.

### 5. 🔍 Retrieval (⬆️ Higher is better)
- **Objective:** Nearest-neighbor embedding search across large document, passage, and speech corpora (e.g. ScaNN indexing).
- **Primary Metric:** `MRR` (Mean Reciprocal Rank), `Recall@10`, `NDCG@10`.
- **Datasets:** 112 datasets including Document In-Language & Cross-Language Retrieval, Passage In-Language & Cross-Language Retrieval, Small Index evaluations, and Multimodal Audio-Visual Retrieval (Flickr8k, SpokenCOCO).

### 6. ✂️ Segmentation (⬆️ Higher is better)
- **Objective:** Detect precise temporal boundaries (start/end timestamps within tolerance $\\tau$) and classify salient spoken terms.
- **Primary Metric:** `TimestampsAndEmbeddingsAccuracy` (Overall Accuracy requiring both timestamp alignment and correct label identification).
- **Datasets:** SVQ Multilingual Salient Term Segmentation across 26 languages.

### 7. ✍️ Transcription (⬇️ Lower is better)
- **Objective:** Automatic speech recognition (ASR) converting spoken audio into verbatim text.
- **Primary Metric:** `WER` (Word Error Rate: $\\frac{S + D + I}{N}$).
- **Datasets:** SVQ Speech Transcription across 26 languages and dialects.

---

## SVQ Macro-Averaging Methodology

1. **Orientation Normalization:**
   - For higher-is-better metrics (Accuracy, VMeasure, GmeanF1, MAP, MRR), scores $[0, 1]$ are scaled to percentages $[0, 100]$.
   - For lower-is-better error metrics (WER, SER), scores are inverted: $\\max(0, 100 - \\text{WER}\\%)$.
   - For distance metrics (CED, FAD), scores are mapped smoothly: $\\frac{100}{1 + \\text{distance}}$.
2. **Task & Overall SVQ Macro-Averaging:**
   - For each model and task, the mean of evaluated SVQ dataset scores is computed.
   - The **Average ⬆️** score is computed as the unweighted macro-average (`nanmean`) across all evaluated tasks. Unevaluated tasks do not penalize or drop sparse models.

---

## Model Configurations & Subfolders

Models with multiple ASR backends or prompting strategies appear as distinct rows:
- `base_model` (e.g. `gemini-2.5-flash`): Direct audio input / default model architecture.
- `base_model (asr=truth)`: Audio cascade using ground-truth reference transcripts.
- `base_model (asr=whisper)`: Audio cascade using Whisper ASR transcripts.
- `base_model (asr=gpt-4o-transcribe)`: Audio cascade using GPT-4o transcripts.

---

## ✉️ Submission Instructions

To submit your model to the MSEB Leaderboard please contact mseb-dev@google.com.
"""


def create_about_tab() -> gr.TabItem:
  """Constructs the About & Documentation tab."""
  with gr.TabItem("About", id="tab-about"):
    gr.Markdown(ABOUT_MARKDOWN, elem_classes=["markdown-text"])

    with gr.Accordion("BibTeX Citation", open=True):
      gr.Textbox(
          value=CITATION_BUTTON_TEXT.strip(),
          label="Copy BibTeX Citation",
          lines=8,
          elem_id="citation-textbox",
          show_copy_button=True,
          interactive=False,
      )
