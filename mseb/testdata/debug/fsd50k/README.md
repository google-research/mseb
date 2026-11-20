# FSD50K Debug Subset

This directory contains a 4-example filtered subset of the **FSD50K** (`test` /
`eval`) dataset for MSEB debug task testing.

## License & Attribution

* **Dataset Curation & Annotations** (`labels/vocabulary.csv` and label/MID
  columns in `test.parquet`): Derived from **FSD50K** by Eduardo Fonseca, Xavier
  Favory, Jordi Pons, Frederic Font, and Xavier Serra (*"FSD50K: an Open Dataset
  of Human-Labeled Sound Events"*, IEEE/ACM TASLP, 2022,
  https://doi.org/10.5281/zenodo.4060432), licensed under **Creative Commons
  Attribution 4.0 International (CC BY 4.0)**
  (https://creativecommons.org/licenses/by/4.0/). Changes made: filtered `eval`
  split down to the 4 debug clips listed below and stored waveforms in Parquet
  format.
* **Audio Clips** (`waveform` column in `test.parquet`): Each Freesound clip is
  distributed under its uploader's permissive Creative Commons license as
  recorded in `FSD50K.metadata/eval_clips_info_FSD50K.json` (waveform stored in
  Parquet format):
  * `37199` — *elecriff3.wav* by `UncleSigmund`
    (https://freesound.org/s/37199/), **CC0 1.0 Universal**
    (http://creativecommons.org/publicdomain/zero/1.0/)
  * `175151` — *western-danger-stab-02.wav* by `MinigunFiend`
    (https://freesound.org/s/175151/), **CC0 1.0 Universal**
    (http://creativecommons.org/publicdomain/zero/1.0/)
  * `329838` — *Ambient guitar loop* by `deerstalking`
    (https://freesound.org/s/329838/), **CC BY 3.0**
    (http://creativecommons.org/licenses/by/3.0/)
  * `1277` — *arpeggio.wav* by `plagasul` (https://freesound.org/s/1277/),
    **CC BY 3.0** (http://creativecommons.org/licenses/by/3.0/)
