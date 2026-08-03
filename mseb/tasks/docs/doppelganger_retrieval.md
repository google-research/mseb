# Doppelganger retrieval

Doppelganger measures whether an audio embedding preserves identity and event
category across the real-synthetic boundary. The benchmark evaluates the fixed
3,065-pair test set in both directions:

- `DoppelgangerSyntheticToRealRetrieval` uses each synthetic twin as a query
  and the 3,065 real recordings as the gallery.
- `DoppelgangerRealToSyntheticRetrieval` uses each real recording as a query
  and the 3,065 synthetic twins as the gallery.

Each direction reports two subtasks. In `exact_source`, only the query's paired
counterpart is relevant. In `category`, every gallery item with the same UCS
event category as the query is relevant. The category subtask therefore asks a
different question from exact matching: it rewards an embedding for retrieving
the right kind of sound even when it does not recover the particular source
recording.

MSEB ranks the full gallery by its standard dot-product similarity. Full
rankings are retained so mean average precision (MAP) is exact for the
multi-relevant category task. MAP is the main score. For exact-source
retrieval, MAP equals reciprocal rank; `EM` gives top-1 exact-source recall.
For category retrieval, MAP averages precision at every same-category hit;
`EM` indicates whether the top-ranked result is in the correct category.

The split and pair identities are fixed by the versioned `mseb` Hugging Face
configuration. No training data is part of this task.
