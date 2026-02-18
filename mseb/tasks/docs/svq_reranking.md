# SVQ Reranking

The reranking task assesses a model's ability to refine a list of candidate
answers.

-   **Input:** A voice query and a set of candidate text answers (e.g., top-K
    results from a first-stage retrieval system).
-   **Goal:** Re-order the candidates so that the ground-truth answer appears at
    rank 1.
