# MSEB Evaluation Metrics

This page documents the evaluation metrics reported across Massive Sound
Embedding Benchmark (MSEB) evaluators, task suites, and benchmark leaderboards.
Each `Score` object in MSEB records a `metric` identifier, a human-readable
`description`, a scalar `value`, an optional standard deviation `std`, and
`[min, max]` bounds. Unless noted otherwise, ratio metrics are computed in
$[0,1]$ by the evaluators and often displayed as percentages (`0-100%`) in
leaderboard tables.

## Transcription Metrics

Speech transcription tasks evaluate recognition accuracy primarily through a
family of corpus-level micro-averaged Levenshtein edit-distance rate metrics —
including Word Error Rate (`WER`) and Character Error Rate (`CER`) — alongside
utterance-level Sentence Error Rate (`SER`). In implementation, the error rates
are guarded against zero denominators by flooring the denominator at $1$ via
$\max(\text{denominator},1)$, however we omit this in the equations below for
simplicity.

### Word Error Rate (WER)

-   **Metric ID:** `WER` (also `WordErrorRate` in sequence segmentation)
-   **Range & Direction:** $[0,\infty)$ — **lower is better** (`0.0` is an
    exact match; values exceed `1.0` when insertions outnumber reference words).
-   **Definition:** Corpus-level micro-averaged word Levenshtein edit distance
    between normalized reference and hypothesis transcripts. For each utterance
    $u$ with normalized reference word sequence
    $r_u=(r_{u,1},\dots,r_{u,N_u})$ and normalized hypothesis word sequence
    $h_u=(h_{u,1},\dots,h_{u,M_u})$, word-level minimum edit distance
    (`jiwer.process_words`) yields exact-match word hits $C_u$, substitutions
    $S_u$, deletions $D_u$, and insertions $I_u$ (where
    $C_u+S_u+D_u=N_u$ is the normalized reference word count). Counts are
    summed across all utterances $u$ in the evaluation split before dividing
    (never averaging per-utterance or per-batch WERs):

$$
\text{WER} = \frac{\sum_u (S_u + D_u + I_u)}{\sum_u (C_u + S_u + D_u)}
$$

-   **Normalization & No-Response Handling:** Both raw reference
    ($r_{\text{raw},u}$) and hypothesis ($h_{\text{raw},u}$) strings pass
    through the task's text normalizer $\tau(\cdot)$ prior to word
    tokenization and alignment. `transcription_evaluator.text_transform(lang)`
    splits `lang` on underscores (`lang.split('_')[0].lower() == 'en'`),
    selecting `EnglishTextNormalizer` for `'en'` and underscore-delimited
    English tags such as `'en_us'` and `BasicTextNormalizer` otherwise (passing
    a hyphenated tag such as `'en-US'` falls through to `BasicTextNormalizer`).
    Utterances returning `LLM_NO_RESPONSE_STR` (`"NO_RESPONSE"`) are scored
    against an empty hypothesis (`""`), contributing `100%` deletion errors
    ($D_u=N_u$) and incrementing `NoResultRate`, whereas an ordinary empty
    transcript (`""`) is scored with $D_u=N_u$ and `0` `NoResultRate`.

### Sentence Error Rate (SER)

-   **Metric ID:** `SER`
-   **Range & Direction:** $[0,1]$ — **lower is better**.
-   **Definition:** Fraction of evaluated utterances ($U$) with at least one
    word error after text normalization ($S_u+D_u+I_u>0$) or a missing
    response (`"NO_RESPONSE"`):

$$
\text{SER} = \frac{1}{|U|} \sum_{u \in U} \mathbb{I}[S_u + D_u + I_u > 0]
$$

### Character Error Rate (CER)

-   **Metric ID:** `CER`
-   **Range & Direction:** $[0,\infty)$ — **lower is better**.
-   **Definition:** Corpus-level micro-averaged character Levenshtein edit
    distance (`jiwer.process_characters`) over normalized reference characters
    (`CharCount`), reported alongside `WER` for writing systems without
    whitespace word boundaries (`cmn_hans_cn`, `ja_jp`, and `ko_kr`):

$$
\text{CER} = \frac{\sum_u (S_u^{\text{char}} + D_u^{\text{char}} + I_u^{\text{char}})}{\sum_u (C_u^{\text{char}} + S_u^{\text{char}} + D_u^{\text{char}})}
$$

-   **Normalization:** Applies the task's base text normalizer, converts
    Traditional Chinese characters to Simplified Chinese via OpenCC (`t2s`) on
    `cmn_hans_cn`, and removes whitespace on `cmn_hans_cn` and `ja_jp` before
    character alignment.

## Retrieval Metrics

Retrieval evaluators rank corpus documents by dot-product similarity using exact
`BruteForceSearcher` by default (`allow_scann = False`; when approximate search
is explicitly enabled via `--allow_scann`, indices with $\ge{50{,}000}$
documents use ScaNN) and truncate predictions to `top_k` (default $k=10$).
Invalid or missing predictions receive `0.0` on all quality metrics below.

### Mean Reciprocal Rank (MRR)

-   **Metric ID:** `MRR`
-   **Range & Direction:** $[0,1]$ — **higher is better**.
-   **Definition:** Mean across queries $Q$ of the reciprocal `1`-indexed rank
    $\text{rank}_q$ of the highest-ranked relevant document within the top
    $k$ retrieved candidates (`0.0` if no relevant document appears in top
    $k$; primary score for SVQ passage and document retrieval):

$$
\text{MRR}@k = \frac{1}{|Q|} \sum_{q \in Q} \frac{\mathbb{I}[\text{rank}_q \le k]}{\text{rank}_q}
$$

### Exact Match (EM)

-   **Metric ID:** `EM`
-   **Range & Direction:** $[0,1]$ — **higher is better**.
-   **Definition:** Fraction of queries where the top-1 retrieved document
    ($\text{rank}_q=1$) is relevant (equivalent to `RecallAt1` and
    `Precision@1`).

### Mean Average Precision (MAP)

-   **Metric ID:** `MAP`
-   **Range & Direction:** $[0,1]$ — **higher is better**.
-   **Definition:** Mean across queries of binary-relevance Average Precision.
    For query $q$ with relevant document set $R_q$ and ranked unique
    retrieved list $(d_1,\dots,d_K)$, summing precision at each rank $r$
    where $d_r\in{R_q}$ (primary score for `Doppelganger` audio-to-audio
    retrieval and query reranking):

$$
\text{AP}_q = \frac{1}{|R_q|} \sum_{r=1}^{K} \mathbb{I}[d_r \in R_q] \cdot \text{Precision}@r, \qquad \text{MAP} = \frac{1}{|Q|} \sum_{q \in Q} \text{AP}_q
$$

### Normalized Discounted Cumulative Gain (NDCG)

-   **Metric ID:** `NDCG@10` (retrieval), `NDCG` (reranking at default $k=3$,
    and sequence segmentation)
-   **Range & Direction:** $[0,1]$ — **higher is better**.
-   **Evaluator-Specific Definitions:**
    -   **Retrieval (`NDCG@10`):** Single-target binary discount
        $1/\log_2(\text{rank}+1)$ when a single string `reference_id` matches
        at $1\le\text{rank}\le{10}$, and `0.0` otherwise (multi-target list
        subtasks such as `Doppelganger` category retrieval use `MAP` as their
        primary metric).
    -   **Reranking (`NDCG`, default $k=3$):** `scikit-learn` `ndcg_score`
        computed over binary candidate relevance labels ranked by predicted
        similarity score (primary score for salient-term reranking).
    -   **Sequence Segmentation (`NDCG`):** Position-wise sequence discount
        awarding $1/\log_2(i+2)$ at `0`-based index $i$ only when the
        $i$-th predicted segment term matches the $i$-th reference segment
        label, divided by `IDCG` over `len(reference_terms)` (primary score for
        salient-term segmentation).

$$
\text{DCG}@k = \sum_{r=1}^{k} \frac{\text{rel}_r}{\log_2(r + 1)}, \qquad \text{NDCG}@k = \frac{\text{DCG}@k}{\text{IDCG}@k}
$$

### Recall at K (RecallAtK)

-   **Metric ID:** `RecallAt10` (`RecallAt{k}`), `RecallAtInf`
-   **Range & Direction:** $[0,1]$ — **higher is better**.
-   **Definition:** Fraction of queries for which at least one relevant document
    appears within the top $k$ retrieved candidates. Note that
    `RetrievalEvaluator` truncates candidate lists to the configured `top_k`
    (default $k=10$) before scoring and names the companion metric
    `RecallAt{top_k}`, so `RecallAtInf` always equals `RecallAt{top_k}` for the
    configured `top_k` cutoff (raising `top_k`—for example, `RecallAt3065` on
    `Doppelganger`—moves both metrics together toward full-gallery recall).

## Reranking Metrics

Reranking evaluators score a per-example candidate text list against the query
embedding, reporting `MAP`, `NDCG` (at $k=3$), `MRR` (at $k=10$), top-1
`WER` against the ground-truth text, and `CER` below. Note two implementation
distinctions in `RerankingEvaluator`:

-   **String Normalization & Multi-Reference Scope:** Top-1 `WER` and `CER`
    normalize both `texts[0]` and the top-ranked candidate via `text_transform`,
    whereas `MRR` (checked against `texts[0]` only), `MAP`, and `NDCG` (checked
    for membership across all `texts`) compare raw unnormalized candidate
    strings.
-   **`MAP` Denominator:** Whereas `RetrievalEvaluator` divides $\text{AP}_q$
    by the total ground-truth relevant set size $\lvert{R_q}\rvert$,
    `RerankingEvaluator` delegates `MAP` to `scikit-learn`
    `average_precision_score` over `predicted_items` (dividing by relevant items
    present in the returned candidate list).

### Candidate Error Rate (CER)

-   **Metric ID:** `CER` (on reranking tasks)
-   **Range & Direction:** $[0,1]$ — **lower is better**.
-   **Definition:** Fraction of reranking examples whose top-ranked candidate
    has at least one word error ($S_u+D_u+I_u>0$) relative to the normalized
    reference text (analogous to Sentence Error Rate on the selected top-1
    candidate; distinct from transcription Character Error Rate).

## Reasoning Metrics

Span reasoning tasks select a text span or abstain with `'No Answer'` (when no
explicit threshold is configured, a 2-component 1D Gaussian Mixture Model is fit
over each candidate-bearing example's **top-1** span similarity score, and any
example whose top-1 score falls below the resulting Bayes decision boundary is
predicted as `'No Answer'`).

### Geometric Mean F1 (GmeanF1)

-   **Metric ID:** `GmeanF1`
-   **Range & Direction:** $[0,1]$ — **higher is better**.
-   **Definition:** Prevalence-weighted geometric mean combining mean token `F1`
    on answerable examples ($\text{F1}_{\text{ans}}$, with prevalence
    $w=N_{\text{ans}}/N_{\text{total}}$) and exact abstention accuracy on
    `'No Answer'` examples ($\text{F1}_{\text{no-ans}}$, weight $1-w$),
    preventing trivial all-`'No Answer'` baselines from scoring high (primary
    score for SVQ span reasoning):

$$
\text{GmeanF1} = \left(\text{F1}_{\text{ans}}\right)^{w} \cdot \left(\text{F1}_{\text{no-ans}}\right)^{1 - w}, \qquad w = \frac{N_{\text{ans}}}{N_{\text{total}}}
$$

### Token F1 (F1)

-   **Metric ID:** `F1`
-   **Range & Direction:** $[0,1]$ — **higher is better**.
-   **Definition:** Whitespace-tokenized bag-of-words token `F1` score
    (`reasoning_evaluator.compute_f1_score`) averaged across all examples.
    `target.split()` and `prediction.split()` are compared directly as multisets
    (`collections.Counter`) without lowercasing, punctuation stripping, or
    article removal. Empty/invalid predictions (`""`) score `0.0`, and if either
    target or prediction is `'No Answer'`, the per-example score is `1.0` on
    exact match and `0.0` otherwise.

## Classification Metrics

### Accuracy and Top-K Accuracy

-   **Metric ID:** `Accuracy`, `Top-5 Accuracy` (`Top-{k} Accuracy`)
-   **Range & Direction:** $[0,1]$ — **higher is better**.
-   **Definition:** Multi-class top-1 accuracy (fraction of examples whose
    highest-scoring class matches the reference label; primary score for intent
    and speaker-gender classification) and top-$k$ hit rate (fraction of
    examples where the reference label appears among the top $k$ predicted
    classes). Invalid (`""`) and no-response (`"NO_RESPONSE"`) text predictions
    map to dedicated one-hot sentinel label IDs so they count as incorrect top-1
    `Accuracy` predictions (note that top-$k$ selection over one-hot sentinel
    scores still fills the remaining $k-1$ slots with zero-tied class
    indices).

### Balanced Accuracy and Weighted Precision Recall F1

-   **Metric ID:** `Balanced Accuracy`, `Weighted Precision`, `Weighted Recall`,
    `Weighted F1-Score`
-   **Range & Direction:** $[0,1]$ — **higher is better**.
-   **Definition:** Standard `scikit-learn` multi-class metrics:
    `Balanced Accuracy` averages per-class recall equally across classes to
    account for label imbalance, while `Weighted Precision`, `Weighted Recall`,
    and `Weighted F1-Score` weight per-class scores by reference class support.

### Multi-Label Classification Metrics

Reported on multi-label audio event classification tasks (`BirdSet`, `FSD50K`):

-   **`mAP`** ($[0,1]$, **higher is better**): Macro-averaged Mean Average
    Precision across all target classes (threshold-free ranking metric; primary
    score for `BirdSet` and `FSD50K` classification).
-   **`Micro F1`** ($[0,1]$, **higher is better**): Global F1 score computed
    over pooled true positives, false positives, and false negatives after
    binarizing class scores at `threshold = 0.5`.
-   **`Macro F1`** ($[0,1]$, **higher is better**): Unweighted mean of
    per-class binary F1 scores at `threshold = 0.5`.
-   **`Hamming Loss`** ($[0,1]$, **lower is better**): Fraction of
    `(example, class)` binary label decisions predicted incorrectly.
-   **`Subset Accuracy`** ($[0,1]$, **higher is better**): Exact-match
    fraction of examples whose predicted label set matches the ground-truth set
    on every class.

## Clustering Metrics

### V-Measure (VMeasure)

-   **Metric ID:** `VMeasure`
-   **Range & Direction:** $[0,1]$ — **higher is better**.
-   **Definition:** Harmonic mean of clustering homogeneity
    $h=1-H(C\mid{K})/H(C)$ and completeness $c=1-H(K\mid{C})/H(K)$,
    evaluated on `MiniBatchKMeans` assignments with `n_clusters` equal to the
    number of ground-truth classes and fixed `random_state = 0` (primary score
    for all MSEB clustering tasks):

$$
\text{VMeasure} = \frac{2 \cdot h \cdot c}{h + c}
$$

## Segmentation Metrics

### Segmentation Hit Accuracy Metrics

Evaluated via three sequential passes over a single shared pool of unconsumed
predicted `(term, start_time, end_time)` segments and ground-truth reference
segments (`NumSegments`) using a timestamp tolerance $\pm\tau$ seconds
(default $\tau=0.05$ s), where in each pass every unmatched reference segment
claims the **first** unused predicted segment in list order satisfying that
pass's criterion:

-   **`TimestampsAndEmbeddingsAccuracy`** ($[0,1]$, **higher is better**):
    Fraction of reference segments matched in Pass 1 in **both** start/end
    timestamps ($\lvert{t_{\text{pred}}-t_{\text{ref}}}\rvert\le\tau$) and
    case-insensitive term content (`TimestampsAndEmbeddingsHits / NumSegments`).
-   **`TimestampsAccuracy`** ($[0,1]$, **higher is better**): Fraction of
    reference segments matched either jointly in Pass 1 or on start/end
    timestamps within $\pm\tau$ in Pass 2 regardless of label
    (`TimestampsHits / NumSegments`).
-   **`EmbeddingsAccuracy`** ($[0,1]$, **higher is better**): Fraction of
    reference segments matched either jointly in Pass 1 or on case-insensitive
    term content in Pass 3 among segments not already consumed in Passes 1–2
    (`EmbeddingsHits / NumSegments`).

Segmentation evaluators also report:

-   **Sequence `NDCG`** ($[0,1]$, **higher is better**): Position-wise
    sequence discount over ordered lowercased segment labels (see
    [`NDCG`](#normalized-discounted-cumulative-gain-ndcg)).
-   **Sequence `WordErrorRate`** ($[0,\infty)$, **lower is better**):
    Corpus-level word Levenshtein error rate (`jiwer.process_words`) computed
    after lowercasing and space-joining each example's ordered reference and
    predicted segment labels (splitting multi-word segment labels into
    individual words; returns `0.0` if total reference word count is $0$).
-   **Detection `mAP`** ($[0,1]$, **higher is better**): Dataset-level Average
    Precision ranked by segment confidence score and rescaled by
    `matched_positives / total_reference_segments` to penalize missed
    detections, where a true positive requires **both** start/end timestamps
    within $\pm\tau$ and a case-insensitive term match. Computed **only** for
    `SoundEmbedding` predictions carrying explicit `scores` (`TextPrediction` /
    LLM outputs omit confidence scores and receive `mAP = 0.0`).

## Reconstruction Metrics

### Frechet Audio Distance (FAD)

-   **Metric ID:** `FAD`
-   **Range & Direction:** $[0,\infty)$ — **lower is better**.
-   **Definition:** **Squared** Fréchet (2-Wasserstein) distance between
    multivariate Gaussians $(\mu_x,\Sigma_x)$ and $(\mu_y,\Sigma_y)$ fit to
    all pooled reference and reconstructed embedding frames across the dataset
    (regularizing covariances with $\varepsilon{I}$ inside
    `scipy.linalg.sqrtm` at $\varepsilon=10^{-6}$, taking the real part if
    numerical error yields tiny imaginary components, and clamping at zero via
    $\max(0,\text{score})$):

$$
\text{FAD} = \|\mu_x - \mu_y\|_2^2 + \operatorname{Tr}\!\left(\Sigma_x + \Sigma_y - 2(\Sigma_x \Sigma_y)^{1/2}\right)
$$

### Kernel Audio Distance (KAD)

-   **Metric ID:** `KAD`
-   **Range & Direction:** $[0,\infty)$ — **lower is better**.
-   **Definition:** Squared Maximum Mean Discrepancy ($\text{MMD}^2$
    U-statistic, scaled by `100` and clamped at zero via
    $\max(0,\text{score})$) between pooled reference and reconstructed
    embedding frames using a characteristic kernel (`gaussian` by default, with
    bandwidth set via the median pairwise Euclidean distance over the
    **reconstructed** frames). Unlike `FAD`, `KAD` makes no Gaussian assumption,
    and its underlying U-statistic estimator is unbiased prior to the final
    $\max(0,\cdot)$ clamp.

### Embedding Mean Squared Error (Embedding MSE)

-   **Metric ID:** `Embedding MSE`
-   **Range & Direction:** $[0,\infty)$ — **lower is better**.
-   **Definition:** Per-example frame-by-frame Mean Squared Error
    `mean((X - Y)^2)` between aligned original and reconstructed embedding
    tensors, averaged across all examples whose reference and reconstructed
    arrays have matching shapes (examples with mismatched shapes are skipped).

## Stability Metrics

Stability tasks compare each clean reference utterance ($T_{\text{ref}}>0$;
empty references with $T_{\text{ref}}=0$ are skipped) against $N$
deterministically seeded SpecAugment perturbations. Every base distance
`<Metric>` in `{CED, UED, DTW, L2}` (optionally prefixed with `<head>_` on
multi-head `SoundEmbeddingCollection` outputs) is emitted exclusively under two
score IDs:

-   **`Corpus_Mean_<Metric>`:** Global micro-averaged drift across all valid
    corpus pairs.
-   **`Mean_Local_IS_<Metric>`:** Macro-averaged per-utterance mean drift across
    perturbations (`value`), whose `std` field records the corpus mean of each
    utterance's **Instability Score** (within-utterance population standard
    deviation of normalized drift across its $N$ SpecAugment perturbations).

### Continuous Edit Distance (CED)

-   **Metric IDs:** `Corpus_Mean_CED`, `Mean_Local_IS_CED`
-   **Range & Direction:** $[0,\infty)$ — **lower is better**.
-   **Definition:** Sequence-aware Levenshtein edit distance generalized to
    continuous `[T, D]` frame embeddings ($T_{\text{ref}}>0$). Each sequence
    is scaled by its maximum frame L2 norm so all vectors lie inside the unit
    ball (bounding pairwise Euclidean substitution cost
    $\lVert{u_i-v_j}\rVert_2\le{2.0}$), aligned via dynamic programming with
    insertion and deletion penalties $w_{\text{ins}}=w_{\text{del}}=2.0$, and
    normalized by $2\cdot{T_{\text{ref}}}$:

$$
\text{CED}(U, V) = \frac{d_{\text{edit}}(\hat{U}, \hat{V};\, w_{\text{ins}}=2, w_{\text{del}}=2)}{2 \cdot T_{\text{ref}}}
$$

### Unit Edit Distance (UED)

-   **Metric IDs:** `Corpus_Mean_UED`, `Mean_Local_IS_UED`
-   **Range & Direction:** $[0,\infty)$ — **lower is better**.
-   **Definition:** Levenshtein edit distance $(S+D+I)/T_{\text{ref}}$
    ($T_{\text{ref}}>0$) between discrete `[T]` token ID sequences or
    whitespace-tokenized text predictions under perturbation.

### Dynamic Time Warping (DTW) and Lp Distance (L2)

-   **Metric IDs:** `Corpus_Mean_DTW`, `Mean_Local_IS_DTW`, `Corpus_Mean_L2`,
    `Mean_Local_IS_L2`
-   **Range & Direction:** $[0,\infty)$ — **lower is better**.
-   **Definition:** Companion continuous stability distances normalized by
    $T_{\text{ref}}$ ($T_{\text{ref}}>0$): `DTW` computes the minimum
    cumulative Euclidean alignment cost under non-linear time warping, while
    `L2` computes the Frobenius/Euclidean norm $\lVert{Z_1-Z_2}\rVert_2$ after
    zero-padding the shorter sequence to match the longer sequence's frame
    count.

## Brain Encoding Metrics

### Brain Encoding Correlation Metrics

Evaluated by downsampling frame embeddings to fMRI repetition-time (`TR`)
resolution, stacking Finite Impulse Response (`FIR`) delays, and fitting
per-voxel SVD ridge regression models (with per-voxel regularization $\alpha$
selected by cross-validation on training data) on training BOLD responses and
evaluating on held-out test responses:

-   **`MeanCorrelation`** ($[-1,1]$, **higher is better**): Mean voxel-wise
    Pearson correlation $r$ between predicted and observed fMRI BOLD time
    series on held-out test stimuli (`std` reports cross-voxel standard
    deviation).
-   **`MedianCorrelation`** ($[-1,1]$, **higher is better**): Median
    voxel-wise Pearson correlation $r$ across voxels on held-out test stimuli.
-   **`FractionSignificant`** ($[0,1]$, **higher is better**): Fraction of
    voxels achieving strictly positive held-out prediction correlation
    (`mean(r > 0)`; note that this thresholds directly at $r>0$ rather than
    running a $p<0.05$ statistical significance test, despite the `Score`
    description text).

## Reliability and Coverage Rates

-   **`NoResultRate` and `MissingResultRate`** ($[0,1]$, **lower is better**):
    Fraction of evaluation examples whose prediction equals
    `types.LLM_NO_RESPONSE_STR` (`"NO_RESPONSE"`, reserved for exhausted RPC
    retries or missing cache entries). `TranscriptionEvaluator`,
    `RetrievalEvaluator` (which converts missing cache keys to `"NO_RESPONSE"`),
    and `RerankingEvaluator` emit `NoResultRate`; `ReasoningEvaluator`,
    single-label `ClassificationEvaluator` (which skips missing cache keys), and
    `SegmentationEvaluator` emit `MissingResultRate`.
-   **`InvalidResultRate`** ($[0,1]$, **lower is better**): Fraction of
    structured-prediction examples (retrieval, reasoning, reranking,
    classification, and segmentation) whose prediction equals
    `types.LLM_INVALID_ANSWER_STR` (`""`, the empty string, or
    `InvalidAnswerListPrediction` in retrieval/reranking when an encoder output
    is empty or fails response parsing). In speech transcription, an empty
    string `""` is scored as a valid empty transcript rather than an invalid
    answer.

## Encoding Efficiency and Compute Metrics

Every MSEB task evaluation appends two representation-efficiency `Score` entries
from `EncodingStats` (`mseb/leaderboard.py`):

-   **`mean_encoding_size_bytes`** ($[0,\infty)$ bytes, **lower is more
    compact**): Mean in-memory representation size (`output.size_bytes`) across
    per-example outputs on the task: `embedding.size * embedding.dtype.itemsize`
    for single-array embeddings, `len(prediction.encode('utf-8'))` for
    `TextPrediction` outputs, and the sum of `size_bytes` across bundled heads
    for `SoundEmbeddingCollection`.
-   **`flops`** ($[0,\infty)$, **lower is cheaper**): Floating-point
    operations reported on the first cached example (`stats[0].flops` from
    `encoder.get_encode_flops(...)`, intended for constant fixed-window FLOP
    estimates; `0.0` when not implemented by the encoder).

Dedicated hardware throughput benchmarks additionally measure static-batch
inference speed on fixed-duration audio clips:

-   **Throughput (`audio s/s`):** Ratio of total input audio duration (in
    seconds) processed per wall-clock second of steady-state inference (**higher
    is better**).
-   **Average Latency (`ms`):** Mean wall-clock inference latency in
    milliseconds per static batch (**lower is better**).
