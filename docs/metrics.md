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
including Word Error Rate (`WER`), Character Error Rate (`CER`), Entity Word
Error Rate (`EntityWER` and `NonEntityWER`), and contextual biasing error rates
(`No-List WER`, `k-Distractor WER`, `BWER`, and `RareWER`) — alongside
utterance-level Sentence Error Rate (`SER`) and streaming latency metrics. In
implementation, entity-attributed rates (`EntityWER`, `BWER`, and `RareWER`)
guard against zero denominators by flooring the denominator at $1$ via
$\max(\text{denominator},1)$ (while stability evaluations skip empty
references with $T_{\text{ref}}=0$), sequence `WordErrorRate` in
`SegmentationEvaluator` returns `0.0` when total reference word count is $0$,
and standard `WER`/`CER` in `TranscriptionEvaluator` accept empty per-utterance
references (adding insertions with weight $0$) while dividing by total corpus
reference length without a zero guard (`compute_weighted_average_and_std` raises
`ZeroDivisionError` if an entire split has $0$ total reference
words/characters); we omit explicit zero-denominator guards from the equations
below for simplicity.

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

### Entity Word Error Rate (EntityWER)

-   **Metric ID:** `EntityWER` (companion metrics: `NonEntityWER` and
    `EntityWordCount`, where `EntityWordCount` equals the effective entity
    denominator weight $W_E$ below, including hypothesis-only entity
    substitutions rather than solely reference entity words; note that the base
    open-source `TranscriptionEvaluator` emits `WER`, `SER`, `NoResultRate`,
    `UtteranceCount`, and `WordCount`—plus `CER` and `CharCount` on CJK
    locales—while extended entity-aware transcription evaluators additionally
    emit `EntityWER`, `NonEntityWER`, and `EntityWordCount` whenever
    `TranscriptTruth` carries `entities` annotations such as SVQ salient terms;
    open-source `TranscriptTruth.entities` and entity-evaluator rollout is in
    progress)
-   **Range & Direction:** $[0,\infty)$ — **lower is better**.
-   **Symmetric Alignment Attribution:** Given an utterance's normalized
    reference word sequence $r=(r_1,\dots,r_N)$, normalized hypothesis word
    sequence $h=(h_1,\dots,h_M)$, and a per-utterance target set $E$ of
    normalized **single-word** entities (`EntityWER` transforms each element of
    `entities` without splitting on whitespace and matches single words in $r$
    and $h$, so multi-word entries in `entities` such as `"the who"` do not
    match individual words and count toward `NonEntityWER`, whereas the `BWER`
    scorer below explicitly splits bias phrases into single words first), a
    single utterance-wide word Levenshtein alignment is computed between $r$
    and $h$. Every word-level alignment operation is then partitioned
    **symmetrically** between entity statistics $(C_E,S_E,D_E,I_E)$ and
    non-entity statistics $(C_{\bar{E}},S_{\bar{E}},D_{\bar{E}},I_{\bar{E}})$
    based on whether **either** the aligned reference word **or** the aligned
    hypothesis word belongs to $E$:

| Alignment Operation | Condition for Entity Split ($E$) | Entity Count Updated | Otherwise ($\bar{E}$ / `NonEntity` Split) |
| :--- | :--- | :--- | :--- |
| **Equal** ($r_i=h_j$) | $r_i\in{E}$ (equivalently $h_j\in{E}$) | $C_E\leftarrow{C_E+1}$ | $C_{\bar{E}}\leftarrow{C_{\bar{E}}+1}$ |
| **Substitution** ($r_i\to{h_j}$, $r_i\ne{h_j}$) | $r_i\in{E}$ **or** $h_j\in{E}$ | $S_E\leftarrow{S_E+1}$ | $S_{\bar{E}}\leftarrow{S_{\bar{E}}+1}$ |
| **Deletion** ($r_i\to\varepsilon$) | $r_i\in{E}$ | $D_E\leftarrow{D_E+1}$ | $D_{\bar{E}}\leftarrow{D_{\bar{E}}+1}$ |
| **Insertion** ($\varepsilon\to{h_j}$) | $h_j\in{E}$ | $I_E\leftarrow{I_E+1}$ | $I_{\bar{E}}\leftarrow{I_{\bar{E}}+1}$ |

-   **Why Substitutions Use $r_i\in{E}$ or $h_j\in{E}$:** Earlier asymmetric
    formulations (such as `B-WER` in
    [Le et al., 2021](https://arxiv.org/abs/2104.02194)) checked $E$ on the
    hypothesis side for insertions ($h_j\in{E}$) **but not** for substitutions
    (checking only $r_i\in{E}$). Because a single substitution (cost $1$)
    and a deletion-plus-insertion (cost $2$, or tied under multi-word error
    spans) differ only in alignment pairing, ignoring $r_i\notin{E}$ with
    $h_j\in{E}$ substitutions makes asymmetric `B-WER` alignment-dependent and
    misses false-alarm entity words whenever they overwrite a reference word
    instead of inserting between words. Symmetric attribution ($r_i\in{E}$ or
    $h_j\in{E}$) counts **every** alignment error involving a word in $E$,
    whether missed from the reference or falsely emitted in the hypothesis.
-   **Corpus Aggregation & Equations:** Raw counts $(C_E,S_E,D_E,I_E)$ and
    $(C_{\bar{E}},S_{\bar{E}},D_{\bar{E}},I_{\bar{E}})$ are summed across all
    utterances in the dataset. Define the effective entity weight $W_E$ and
    non-entity weight $W_{\bar{E}}$:

$$
W_E = \sum_u (C_{E,u} + S_{E,u} + D_{E,u}), \qquad W_{\bar{E}} = \sum_u (C_{\bar{E},u} + S_{\bar{E},u} + D_{\bar{E},u})
$$

$$
\text{EntityWER}(E) = \frac{\sum_u (S_{E,u} + D_{E,u} + I_{E,u})}{W_E}, \qquad \text{NonEntityWER}(E) = \frac{\sum_u (S_{\bar{E},u} + D_{\bar{E},u} + I_{\bar{E},u})}{W_{\bar{E}}}
$$

-   **Denominator Note & Exact Decomposition:** Notice that $W_E$ equals the
    number of reference words in $E$ **plus** any hypothesis-only entity
    substitutions ($r_i\notin{E}$ and $h_j\in{E}$), while $W_{\bar{E}}$
    decreases by the same count, so $W_E+W_{\bar{E}}=N_{\text{ref}}$ holds
    identically. When $W_E=0$ (for example, an utterance group with zero
    reference entities and zero entity substitutions), the denominator floor
    ($\max(W_E,1)$) reports the raw count of false-alarm entity insertions
    $\sum_u{I_{E,u}}$. Whenever $W_E>0$ and $W_{\bar{E}}>0$ and all three
    metrics are scored under the same text normalizer $\tau(\cdot)$, overall
    corpus `WER` decomposes exactly as:

$$
\text{WER} = \frac{\text{EntityWER}(E) \cdot W_E + \text{NonEntityWER}(E) \cdot W_{\bar{E}}}{W_E + W_{\bar{E}}}
$$

### Contextual Biasing Metrics

Contextual biasing benchmarks evaluate whether a speech recognizer can use an
utterance-specific candidate phrase list $B_u$ (such as contact names, domain
terminology, or rare proper nouns) to improve rare-word recognition **without**
degrading general transcription quality or hallucinating unspoken distractors.

For each evaluation split (for example, Written
[LibriHeavy](https://github.com/k2-fsa/libriheavy) `dev`, `test-clean`, and
`test-other`—abbreviated `LS Dev`, `LS Test Clean`, and `LS Test Other`, using
re-segmented punctuated book transcripts rather than standard uppercase
LibriSpeech—as well as `SVQ en_us` or `Fleurs en_us`), contextual biasing
benchmark suites report **four raw metrics** across two decoding passes:

1.  **`<Dataset> (no-list)` (`WER`)** — decoded in a dedicated non-biasing pass
    with bias features disabled / an **empty** bias list ($B_u=\emptyset$;
    distinct from an **oracle** $k=0$ biasing pass $B_u=B_u^{+}$, which
    supplies spoken ground-truth rare words with zero distractors).
2.  **`<Dataset> k100` (`WER`)** — overall `WER` decoded with ground-truth bias
    phrases padded with unspoken distractors and capped at $L_{\max}=100$
    total entries at distractor budget $k=100$ ($k$-distractor condition).
3.  **`<Dataset> k100` (`BWER`)** — Biased Word Error Rate over **all** words in
    the final $k$-distractor candidate bias list (targets **and**
    distractors).
4.  **`<Dataset> k100` (`RareWER`)** — Rare Word Error Rate over **only** the
    ground-truth rare words in the reference transcript.

> **Open-Source Package Note:** `mseb` plumbs utterance context via
> `SoundWithTitleAndContext.context_text` and scores standard `WER` and `SER` in
> the base `TranscriptionEvaluator`; the four paired `no-list` and
> $k$-distractor metrics below specify the MSEB contextual biasing benchmark
> protocol (open-source task and evaluator rollout is in progress).

#### No-List Word Error Rate

-   **Condition / Label:** `<Dataset> (no-list)` `WER` (for example,
    `LS Test Other (no-list)` `WER`)
-   **Decoding Condition:** Each utterance $u$ is decoded in a dedicated
    non-biasing pass with bias features disabled and an **empty bias list**
    ($B_u=\emptyset$) — not to be confused with an oracle ($k=0$) biasing
    evaluation pass, where the model still receives spoken ground-truth rare
    words $B_u=B_u^{+}$ without distractors.
-   **Formula:** Standard corpus [`WER`](#word-error-rate-wer) over **all**
    normalized words in the dataset:

$$
\text{WER}_{\text{no-list}} = \frac{\sum_u (S_u^{(0)} + D_u^{(0)} + I_u^{(0)})}{\sum_u (C_u^{(0)} + S_u^{(0)} + D_u^{(0)})}
$$

-   **What It Measures:** Baseline un-biased recognition accuracy on the exact
    same utterances and text normalizer, verifying that contextual biasing
    support does not regress recognition when no context list is supplied.

#### Contextual Biasing k-Distractor WER

-   **Condition / Label:** `<Dataset> k100` `WER` (for example,
    `LS Test Other k100` `WER` at distractor budget $k=100$ and maximum
    bias-list capacity $L_{\max}=100$)
-   **Decoding Condition (`k100` Bias-List Protocol):** For a target distractor
    budget $k$, maximum bias-list capacity $L_{\max}$, and per-term subword
    limit $T_{\max}$ under the evaluated model's subword tokenizer (with
    $k=100$, $L_{\max}=100$, $T_{\max}=10$ subword tokens, and rarity
    threshold $\theta_{\text{rare}}=10$ over a lowercased unigram reference
    catalog in the standard `k100` suite, where out-of-vocabulary words have
    $\operatorname{freq}_{\text{catalog}}(w)=0$), candidate construction first
    collects unique spoken ground-truth rare words $B_u^{+}$
    ($0\le\operatorname{freq}_{\text{catalog}}(w)\le{10}$) in first-occurrence
    transcript order and samples
    $\min(k,\max(0,L_{\max}-\lvert{B_u^{+}}\rvert))$ unspoken single-word
    distractors $B_u^{-}$ uniformly without replacement from in-catalog rare
    words ($1\le\operatorname{freq}_{\text{catalog}}(w)\le{10}$) absent from
    the utterance transcript. When $\lvert{B_u^{+}}\rvert<{L_{\max}}$,
    $B_u^{+}\cup{B_u^{-}}$ has at most $L_{\max}$ entries and is shuffled;
    when $\lvert{B_u^{+}}\rvert\ge{L_{\max}}$, zero distractors are drawn and
    $B_u^{+}$ remains in first-occurrence transcript order without shuffling.
    Bias-list tokenization then filters out any term (target or distractor)
    exceeding $T_{\max}=10$ subword tokens and slices the remaining list to at
    most $L_{\max}=100$ entries, yielding the final decoded bias list $B_u$
    with $\lvert{B_u}\rvert\le{L_{\max}}=100$ (so when more than $L_{\max}$
    unique spoken rare words survive the $T_{\max}$-subword filter, only the
    first $L_{\max}$ in transcript order are retained and later spoken rare
    words are dropped, while terms exceeding $T_{\max}$ subwords are dropped
    regardless). When an utterance has zero spoken rare words
    ($B_u^{+}=\emptyset$), $B_u$ contains up to $k=100$ distractors (all
    sampled distractors with $\le{T_{\max}}$ subwords) so distractor
    hallucinations on target-free utterances count toward corpus `BWER` and
    `k100 WER`.
-   **Formula:** Standard corpus [`WER`](#word-error-rate-wer) over **all**
    normalized words in the utterance (both biased and unbiased words):

$$
\text{WER}_{k} = \frac{\sum_u (S_u^{(k)} + D_u^{(k)} + I_u^{(k)})}{\sum_u (C_u^{(k)} + S_u^{(k)} + D_u^{(k)})}
$$

-   **What It Measures:** Net end-to-end transcription accuracy when biasing is
    active. Comparing `<Dataset> k100` (`WER`) against `<Dataset> (no-list)`
    (`WER`) shows whether rare-word gains outweigh any collateral errors or
    distractor false alarms across the full transcript.

#### Biased Word Error Rate (BWER)

-   **Metric ID / Label:** `BWER` (for example, `LS Test Other k100` `BWER`;
    companion unbiased split: `UBWER`)
-   **Entity Set ($E_{\text{bias},u}$):** All normalized whitespace-split
    words appearing in **any** phrase of the utterance's final decoded bias list
    $B_u$ (after subword-length filtering and $L_{\max}$ truncation, so
    $B_u$ includes surviving spoken target phrases and sampled unspoken
    distractor phrases), normalized with the composed bias normalizer
    $\tau_{\text{bias}}=\tau\circ\operatorname{normalize\_text}$:

$$
E_{\text{bias}, u} = \bigcup_{p \in B_u} \operatorname{words}(\tau_{\text{bias}}(p)), \qquad \tau_{\text{bias}} = \tau \circ \operatorname{normalize\_text}
$$

-   **Formula:** Symmetric [`EntityWER`](#entity-word-error-rate-entitywer)
    evaluated on the $k$-distractor decoding pass with entity set
    $E=E_{\text{bias}}$, where bias phrases, reference transcripts, and
    hypotheses are all normalized with
    $\tau_{\text{bias}}=\tau\circ\operatorname{normalize\_text}$: here
    `normalize_text` applies Unicode NFC normalization, lowercases, folds
    typographic apostrophes to `'`, collapses single-letter dotted initialisms
    (e.g. `"U.S."` → `"us"`), strips inter-digit commas while preserving decimal
    points, and replaces all other punctuation and symbols (including hyphens,
    `$`, and `%` by default) with spaces except word-internal apostrophes,
    before applying the task normalizer $\tau(\cdot)$ (because `WER` and
    `RareWER` apply only $\tau(\cdot)$, dotted initialisms and `$`/`%` symbols
    can yield slightly different token counts between `BWER`/`UBWER` and
    `WER`/`RareWER`):

$$
\text{BWER} = \text{EntityWER}(E_{\text{bias}}) = \frac{\sum_u \left(S_{E_{\text{bias}}, u}^{(k)} + D_{E_{\text{bias}}, u}^{(k)} + I_{E_{\text{bias}}, u}^{(k)}\right)}{\sum_u \left(C_{E_{\text{bias}}, u}^{(k)} + S_{E_{\text{bias}}, u}^{(k)} + D_{E_{\text{bias}}, u}^{(k)}\right)}
$$

-   **Which Errors Are Counted in `BWER`:**
    -   **Missed spoken bias words ($r_i\in{E_{\text{bias}}}$):** Counted as
        deletions ($D_{E_{\text{bias}}}$) if omitted, or as substitutions
        ($S_{E_{\text{bias}}}$) if misrecognized as another word (whether the
        substituted hypothesis word is in $E_{\text{bias}}$ or not).
    -   **False-alarm distractor substitutions ($r_i\notin{E_{\text{bias}}}$
        and $h_j\in{E_{\text{bias}}}$):** **Counted** in
        $S_{E_{\text{bias}}}$ (and in the denominator weight
        $W_{E_{\text{bias}}}$) when the recognizer overwrites an ordinary
        reference word with a distractor word from $B_u$.
    -   **False-alarm distractor insertions ($h_j\in{E_{\text{bias}}}$):**
        **Counted** in $I_{E_{\text{bias}}}$ when the recognizer inserts a
        bias-list or distractor word with no aligned reference word.
    -   **Errors between two non-bias words ($r_i\notin{E_{\text{bias}}}$ and
        $h_j\notin{E_{\text{bias}}}$):** **Not counted** in `BWER`; these go
        to Unbiased Word Error Rate (`UBWER`, i.e.,
        $\text{NonEntityWER}(E_{\text{bias}})$).

#### Rare Word Error Rate (RareWER)

-   **Metric ID / Label:** `RareWER` (for example, `LS Test Other k100`
    `RareWER`; companion common-word split: `NonRareWER`)
-   **Entity Set ($E_{\text{rare},u}$):** Only the normalized **ground-truth
    rare words present in the utterance's normalized reference word sequence
    $r_u$** (defined in the `k100` protocol with rarity threshold
    $\theta_{\text{rare}}=10$ in a lowercased unigram catalog such as
    LibriHeavy, treating out-of-vocabulary words as
    $\operatorname{freq}_{\text{catalog}}(w)=0$, so
    $0\le\operatorname{freq}_{\text{catalog}}(w)\le{10}$). Unspoken distractor
    words from $B_u^{-}$ are **not** in $E_{\text{rare},u}$:

$$
E_{\text{rare}, u} = \{ w \in r_u : 0 \le \operatorname{freq}_{\text{catalog}}(w) \le \theta_{\text{rare}} \}
$$

-   **Formula:** Symmetric [`EntityWER`](#entity-word-error-rate-entitywer)
    evaluated on the $k$-distractor decoding pass (or on the `no-list` pass
    when reporting baseline `RareWER`) with entity set $E=E_{\text{rare}}$:

$$
\text{RareWER} = \text{EntityWER}(E_{\text{rare}}) = \frac{\sum_u \left(S_{E_{\text{rare}}, u}^{(k)} + D_{E_{\text{rare}}, u}^{(k)} + I_{E_{\text{rare}}, u}^{(k)}\right)}{\sum_u \left(C_{E_{\text{rare}}, u}^{(k)} + S_{E_{\text{rare}}, u}^{(k)} + D_{E_{\text{rare}}, u}^{(k)}\right)}
$$

-   **Which Errors Are Counted in `RareWER`:**
    -   **Missed or misrecognized reference rare words
        ($r_i\in{E_{\text{rare}}}$):** Counted as deletions
        ($D_{E_{\text{rare}}}$) or substitutions ($S_{E_{\text{rare}}}$).
    -   **Repeated or mis-aligned occurrences of the utterance's *own* reference
        rare words ($h_j\in{E_{\text{rare}}}$):** Counted as
        $I_{E_{\text{rare}}}$ (if inserted) or $S_{E_{\text{rare}}}$ (if
        substituted onto a non-rare word in the same utterance).
    -   **Hallucinated unspoken distractors ($h_j\in{B_u^{-}}$ and
        $h_j\notin{E_{\text{rare}}}$, aligned to $\varepsilon$ or to a
        common word $r_i\notin{E_{\text{rare}}}$):** **Not counted** in
        `RareWER` (counted in `NonRareWER`, `BWER`, and overall `k100 WER`).
        However, if a distractor word substitutes *for* a reference rare word
        ($r_i\in{E_{\text{rare}}}\to{h_j\in{B_u^{-}}}$), it **is** counted in
        $S_{E_{\text{rare}}}$ because $r_i\in{E_{\text{rare}}}$.

#### Contextual Biasing Error-Counting Summary and Worked Example

The table below contrasts which word-alignment events contribute to each of the
four contextual biasing metrics:

| Alignment Event | `no-list WER` ($B_u=\emptyset$) | `k100 WER` ($B_u\subseteq{B_u^{+}\cup{B_u^{-}}}$) | `k100 BWER` ($E=E_{\text{bias}}$) | `k100 RareWER` ($E=E_{\text{rare}}$) |
| :--- | :--- | :--- | :--- | :--- |
| Common ref word matched ($r_i=h_j$, neither in $B_u$ nor rare) | Denom ($+1$ $C$) | Denom ($+1$ $C$) | Ignored (`UBWER`) | Ignored (`NonRareWER`) |
| Target rare bias word matched ($r_i=h_j\in{E_{\text{rare}}}$) | Denom ($+1$ $C$ on no-list run) | Denom ($+1$ $C$) | Denom ($+1$ $C_E$) | Denom ($+1$ $C_E$) |
| Target rare bias word deleted ($r_i\in{E_{\text{rare}}}\to\varepsilon$) or substituted ($r_i\in{E_{\text{rare}}}\to{h_j\ne{r_i}}$) | Error & Denom (on no-list run) | Error ($+1$) & Denom ($+1$) | Error ($+1$) & Denom ($+1$) | Error ($+1$) & Denom ($+1$) |
| Unspoken distractor substituted onto common ref word ($r_i\notin{E_{\text{bias}}}\to{h_j\in{B_u^{-}}}$) | N/A | Error ($+1$ $S$) & Denom ($+1$) | **Error ($+1$ $S_E$) & Denom ($+1$ $W_E$)** | Ignored (`NonRareWER`) |
| Unspoken distractor inserted ($\varepsilon\to{h_j\in{B_u^{-}}}$) | N/A | Error ($+1$ $I$), Denom $+0$ | **Error ($+1$ $I_E$), Denom $+0$** | Ignored (`NonRareWER`) |
| Common word substituted/deleted/inserted ($r_i,h_j\notin{E_{\text{bias}}}$) | Error (on no-list run) | Error ($+1$) | Ignored (`UBWER`) | Ignored (`NonRareWER`) |

**Worked 1-Utterance Example:**

Suppose an utterance has normalized 6-word reference:

-   Normalized reference $r$: `"call doctor korchnoi at noon please"`
    ($N_{\text{ref}}=6$)
-   Ground-truth rare word set $E_{\text{rare}}$: `{"korchnoi"}` (`1`
    reference rare word)
-   `k100` bias list $B_u$ (`100` total entries) contains `"korchnoi"` plus
    `99` distractors including `"spassky"` and `"petrosian"`, so
    $E_{\text{bias}}$ = `{"korchnoi", "spassky", "petrosian", ...}`.

1.  **No-list pass ($B_u=\emptyset$):** Suppose the model outputs
    `"call doctor kortnoy at noon please"`.
    -   `"korchnoi"` → `"kortnoy"` is `1` substitution ($C=5,S=1,D=0,I=0$).
    -   **`no-list WER`** = $1/6=16.67\%$ (and baseline `RareWER` =
        $1/1=100.0\%$).
2.  **`k100` pass ($B_u$ provided):** Suppose the model recovers `"korchnoi"`
    from $B_u$, **but** overwrites `"noon"` with distractor `"spassky"` and
    inserts distractor `"petrosian"` at the end:
    `"call doctor korchnoi at spassky please petrosian"`.
    -   Word alignment has `4` common hits (`"call"`, `"doctor"`, `"at"`,
        `"please"`), `1` rare/bias hit (`"korchnoi"` = `"korchnoi"`), `1`
        substitution (`"noon"` → `"spassky"`), and `1` insertion
        ($\varepsilon\to$ `"petrosian"`).
    -   **`k100 WER`:** Total errors $S+D+I=1+0+1=2$ over $6$ reference
        words → $2/6=33.33\%$.
    -   **`k100 RareWER` ($E_{\text{rare}}$ = `{"korchnoi"}`):** `"korchnoi"`
        is an exact hit ($C_{E_{\text{rare}}}=1$), and neither `"noon"` →
        `"spassky"` nor $\varepsilon\to$ `"petrosian"` involves `"korchnoi"`
        ($S_{E_{\text{rare}}}=D_{E_{\text{rare}}}=I_{E_{\text{rare}}}=0$) →
        **$0/1=0.00\%$** (`100%` recall on the spoken rare word).
    -   **`k100 BWER` ($E_{\text{bias}}$):** `"korchnoi"` is an entity hit
        ($C_{E_{\text{bias}}}=1$), `"noon"` → `"spassky"` has
        $h_j\in{E_{\text{bias}}}$ so $S_{E_{\text{bias}}}=1$, and
        $\varepsilon\to$ `"petrosian"` has $h_j\in{E_{\text{bias}}}$ so
        $I_{E_{\text{bias}}}=1$ ($D_{E_{\text{bias}}}=0$). Here
        $W_{E_{\text{bias}}}=C_{E_{\text{bias}}}+S_{E_{\text{bias}}}+D_{E_{\text{bias}}}=1+1+0=2$,
        giving **$\text{BWER}=(1+0+1)/2=100.00\%$**, exposing the two
        distractor false alarms that `RareWER` intentionally excludes.

### Streaming Latency and Stability Metrics

Defined for streaming speech transcription benchmark evaluations when
`TextPrediction` carries timestamped `StreamingStats` partial or finalized
recognition events (note that the base open-source `TranscriptionEvaluator`
accepts `StreamingStats` on `TextPrediction` without computing latency scores):

-   **`WallTimeToFirstOutput`** ($[0,\infty)$ seconds, **lower is better**):
    Mean wall-clock elapsed time from stream start to the first non-empty
    partial or finalized hypothesis event.
-   **`ContentTimeToFirstOutput`** ($[0,\infty)$ seconds, **lower is
    better**): Mean audio content timestamp corresponding to the first non-empty
    partial or finalized hypothesis event.
-   **`FirstOutputPersistence`** ($[0,1]$, **higher is better**): Fraction of
    utterances whose normalized first emitted output remains a prefix of the
    model's normalized final prediction (measuring early-hypothesis stability
    against rewrites).
-   **`FirstOutputCorrectness`** ($[0,1]$, **higher is better**): Fraction of
    utterances whose normalized first emitted output is a prefix of the
    normalized ground-truth reference transcript.

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
    classes). Invalid (`""`) and no-response (`"NO_RESPONSE"`) predictions map
    to dedicated sentinel label IDs so they count as incorrect predictions.

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
-   **`Mean_Local_IS_<Metric>`:** Macro-averaged per-utterance drift (`value`),
    whose `std` field records the mean within-utterance **Instability Score**
    across augmentations.

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
