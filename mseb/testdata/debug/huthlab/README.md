# HuthLab fMRI Debug Subset

This directory contains a truncated fMRI response fixture
(`responses/UTS03_responses.jbl`) for the MSEB `HuthLabBrainEncodingDebug` task
(`UTS03`, 100 voxels, 10 TRs for training stories `itsabox` and
`odetostepfather`, and 50 TRs for test story `wheretheressmoke`).

## License & Audio Stimuli

* **fMRI Responses (`responses/UTS03_responses.jbl`)**: Derived from the HuthLab
  natural narrative fMRI dataset (OpenNeuro `ds003020`, LeBel et al.,
  *Scientific Data* 2023; Antonello et al., arXiv:2401.10150), dedicated to the
  public domain under **CC0 1.0 Universal**
  (https://creativecommons.org/publicdomain/zero/1.0/).
* **Audio Stimuli (`stimuli/*.wav`)**: Story audio files (*The Moth Radio Hour*)
  are **not** redistributed in this repository. When `stimuli/{story}.wav` is
  not present locally, `HuthLabDataset.get_sound` downloads only the required
  prefix of each debug story on-the-fly via HTTP `Range` requests from OpenNeuro
  (`https://s3.amazonaws.com/openneuro.org/ds003020/stimuli/`).
