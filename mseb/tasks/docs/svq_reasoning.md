# SVQ Reasoning (Span Retrieval)

The reasoning task requires identifying the exact span of text within a
Wikipedia article that answers a voice query.

-   **Task Format:** Given an audio query and a target document, the model must
    predict the start and end offsets of the answer span.
-   **In-Lang Reasoning:** Query and document share the same language.
-   **Cross-Lang Reasoning:** Query is in a non-English language; the document
    and target answer span are in English.
