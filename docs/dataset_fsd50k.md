# FSD50K

FSD50K is an open dataset of human-labeled sound events containing 51,197 audio
clips totaling 108.3 hours across 200 classes from the AudioSet ontology.

## Licensing & Commercial Subset (`v2.0.0`)

While FSD50K annotations are licensed under `CC BY 4.0`, individual Freesound
clips carry uploader-selected Creative Commons licenses recorded in
`metadata/eval_clips_info_FSD50K.json` and `metadata/dev_clips_info_FSD50K.json`
(included in both Zenodo `FSD50K.metadata` and Hugging Face `Fhrozen/FSD50k`).
Dataset revision `2.0.0` (`commercial_only=True`, used by
`FSD50KTestClassificationCommercial`, `FSD50KTestClusteringCommercial`, the
`Debug` tasks, and the default MSEB `retrieval` and `generative` tracks) checks
each clip's license on the fly (from the Parquet `license` column or
`metadata/{eval,dev}_clips_info_FSD50K.json`), retaining only commercial-friendly
licenses (`CC0 1.0`, `CC BY 3.0`, and `CC BY 4.0`) and excluding `CC BY-NC 3.0`
and `CC Sampling+ 1.0`:

-   **`test` (`eval`)**: 8,403 of 10,231 clips retained (1,828 dropped; all 200
    AudioSet classes remain represented).
-   **`validation` (`dev` `val`)**: 3,410 of 4,170 clips retained (760 dropped;
    all 200 AudioSet classes remain represented).

For direct comparison against published external FSD50K benchmarks, the full
10,231-clip test tasks (`FSD50KTestClassification` and `FSD50KTestClustering`,
dataset revision `1.0.0`, `commercial_only=False`) remain available.

## Task Description

The primary tasks are **Multi-label Sound Event Classification** and **Sound
Event Clustering**. Audio clips are labeled using 200 classes from the AudioSet
ontology.

## Reference

-   **Paper:**
    [FSD50K: an Open Dataset of Human-Labeled Sound Events](https://arxiv.org/abs/2010.00475)
-   **Zenodo Release & Metadata:**
    [10.5281/zenodo.4060432](https://doi.org/10.5281/zenodo.4060432)
-   **Hugging Face:**
    [Fhrozen/FSD50k](https://huggingface.co/datasets/Fhrozen/FSD50k)
