# Doppelganger

Doppelganger is a benchmark for matching real sound effects to
audio-conditioned synthetic twins. The MSEB configuration contains 3,065
one-to-one pairs from the held-out test split, covering 34 sound events in the
Universal Category System taxonomy.

The real recordings come from FSD50K. The synthetic clips were generated from
those recordings with Stable Audio Open. MSEB reads the dedicated `mseb`
configuration from
[`elliottash/doppelganger`](https://huggingface.co/datasets/elliottash/doppelganger).
The configuration pins the audio revisions and records each real clip's
Freesound title, uploader, source URL, and license URL.

Licensing is clip-specific. Real audio includes CC0, CC BY 3.0, CC BY-NC 3.0,
and Sampling+ 1.0 clips. Synthetic audio is covered by the Stability AI
Community License. The metadata and generation records are MIT licensed.

Reference: [Doppelganger: Sound Effects and Their Synthetic
Twins](https://arxiv.org/abs/2607.04337).
