# Sortformer v2.1 parity with NeMo Python

**Status as of 2026-04-14:** Median filter commit (`5a75ffd0`) has been
dropped from the `telemuze-patches` branch by resetting to `7d247578`.
The `--median-window` CLI flag on `sortformer_bench` is also gone.
Default sherpa-onnx now matches the "No-median" column in the table
below. See the sweep for the numbers that justified the drop.

**Status as of 2026-04-11 (afternoon update — 6-recording sweep):**

The earlier `vid-chat-demo.pcm` measurements were comparing sherpa-onnx
against `scripts/sortformer_diag_ref.py`, which is a hand-rolled
reimplementation of NeMo's post-processing that incorrectly adds a
`median_window=11` smoothing step. **NeMo's real post-processing
pipeline (`ts_vad_post_processing` in `nemo/.../vad_utils.py`) does
not apply any median filter.** Once the reference is switched to
NeMo's actual function (fed with the raw sigmoids from
`forward_streaming`), three things become clear:

1. The sherpa-onnx "median filter fix" (commit `5a75ffd0`) was a
   misdiagnosis. It papered over a reference-side artefact; against
   real NeMo it is a **pessimisation** worth ~0.7 pp DER on every
   recording.
2. With `--median-window 1` (disable), sherpa-onnx matches NeMo to
   within 0.11–1.64% DER across six recordings spanning 6 min to
   82 min. For the best-aligned recordings the residual is purely
   boundary jitter and DER drops to essentially the floor.
3. One recording (`communitycall`) is still a mild outlier at 1.64%
   DER even with the fair comparison. Pred-level diff shows the
   divergence is **sparse but high-magnitude** (individual frames
   differ by up to 0.74 in sigmoid space) and **concentrated in
   specific windows later in the recording**. This is a different
   failure mode from boundary jitter — the same class of
   speaker-assignment divergence as the zh-sample finding, but lower
   amplitude.

The zh-sample finding from the earlier section is unchanged: that
recording still drops a whole marginal speaker because
sherpa-onnx's spk3 sigmoid never crosses 0.5. Cache-compression
divergence is still the best hypothesis for that one.

## 6-recording sweep (2026-04-11)

Reference: NeMo `forward_streaming` preds → NeMo's real
`ts_vad_post_processing` (the canonical pipeline, no median filter).
Hypothesis: `target/release/examples/sortformer_bench`, measured
both before and after the `5a75ffd0` drop. Scorer:
`sortformer_der.py` (collar=0). The "No-median" column is now the
shipping default.

| File | Duration | Default (med=11, old) | No-median (current) | Speakers |
| --- | ---: | ---: | ---: | :---: |
| `vid-chat-demo.pcm` | 6.1 min | 1.31% | **0.24%** | 2 |
| `DIY Farming QA.pcm` | 27 min | 1.29% | **0.11%** | 2 |
| `Jonas.pcm` | 25 min | 1.81% | **1.02%** | 2 (+1 @ 4s) |
| `demosession_2025-12-18.pcm` | 37 min | 1.26% | **0.36%** | 3 |
| `TFTechSupportJam…pcm` | 82 min | 1.29% | **0.59%** | 4 |
| `communitycall_jan_8_2026.pcm` | 60 min | 3.11% | **1.64%** | 4 |

Every row improves by 0.7–1.2 pp when the median filter is
disabled on the sherpa-onnx side. Confusion is ≤0.06% in all six
no-median runs — speaker identities agree throughout. The residual
is entirely missed + false-alarm, i.e. boundary disagreements.

### Evidence that the diag-ref reference was artefact-inducing

Running `scripts/sortformer_diag_ref.py` with and without `--no-median`
on vid-chat-demo:

| Reference | Hypothesis | DER |
| --- | --- | ---: |
| `diag-ref` (adds median=11 in Python) | sherpa med=11 | 0.49% |
| `diag-ref --no-median` | sherpa no-med | 0.24% |
| NeMo `ts_vad_post_processing` (real) | sherpa med=11 | **1.31%** |
| NeMo `ts_vad_post_processing` (real) | sherpa no-med | **0.24%** |

The **only** configuration that was stable across the reference
switch was "no median on either side". The previously-reported
0.49% number came from a comparison where both sides happened to
smooth the same way, which hid the fact that sherpa-onnx was now
smoothing where NeMo was not. Switch to NeMo-real, and the
default-mode DER jumps to 1.31%. Disable the sherpa-side median and
the DER drops back to 0.24%.

This also explains why the earlier H1 attribution ("missing median
filter: ~1.1 pp") landed on the wrong root cause: it was computed
against a reference that had an extra smoothing step neither side
of shipping NeMo actually runs.

**Action taken (2026-04-14):** dropped `5a75ffd0` from the
`telemuze-patches` branch via `git reset --hard 7d247578`. The
median filter code and `median_window` config field are gone
entirely; there is no toggle.

### The `communitycall` outlier

At 1.64% DER (no-median on both sides) `communitycall` is still
3–15× worse than the other five recordings. Dumping raw sigmoids
from both sides (via an env-var-gated hook added to
`offline-sortformer-diarization.cc` and reverted before the session
ended) and diffing them frame-by-frame reveals:

- **Overall agreement is tight.** Mean |diff| per speaker is 0.002–
  0.007, RMSE 0.022, and only 2.26% of frames have max|diff|>0.1.
- **But the disagreements are big when they happen.** The top-20
  diverging frames hit 0.43–0.74 in sigmoid space, which is large
  enough to flip a segment boundary or mis-route a new speaker.
- **They are concentrated.** Most 60-second windows have mean|diff|
  ≈ 0.001–0.003, but eight windows in the second half of the
  recording spike to 0.005–0.027. The divergence builds up over the
  duration — 0-20 min is mostly clean, 20-60 min accumulates hotspots.
- **The disagreements look like speaker-assignment flips, not mel
  drift.** Example: at frame 32712–32716 (~2617 s) NeMo has a clean
  spk3 activation (~0.95), sherpa-onnx splits it across spk0/spk1
  at ~0.5/0.5 with spk3 at ~0.2. None of the four sherpa values
  cross onset, so the segment is lost entirely. This is the same
  failure mode as the zh-sample cache-compression divergence, just
  at a different severity.

`DIYFarmingQA.pcm` (the best-aligned recording at 0.11% DER) has
max diffs of 0.25–0.34 and only 98 frames (0.48%) above 0.1, with
no concentration hotspots. The difference between it and
`communitycall` is qualitative, not quantitative: the latter has
distinct divergence events that the former simply doesn't have.

### What I did not investigate this session

- **Whether the divergence events correlate with cache-compression
  boundaries.** The hypothesis would be that compression cycles
  introduce small numerical differences that compound. The test
  would be to run sherpa-onnx with a much larger `spkcache_len`
  (or without compression at all on a test build) and re-measure
  `communitycall`. Not done.
- **Whether the mel preprocessor is the seed.** The earlier doc's
  H4 hypothesis still stands as the likely upstream cause. A
  byte-for-byte mel port remains the most principled single fix.
- **Whether NeMo's `diarize()` high-level API agrees with
  `forward_streaming` + `ts_vad_post_processing`.** My `nemo_real_pp.py`
  runner calls `ts_vad_post_processing` directly on
  `forward_streaming` output. If the top-level `diarize()` path
  applies any additional step I don't know about, the numbers
  above would shift by a small amount. Spot-checking one recording
  with `model.diarize()` would close this gap.

### Reproducing this session

Under `scripts/` and `/tmp/sortformer_parity/`:

```bash
# 1. Convert inputs once (if not already done):
ffmpeg -v error -y -i input.mp4 -f f32le -ac 1 -ar 16000 input.pcm

# 2. Dump NeMo preds:
.venv/bin/python scripts/sortformer_diag_ref.py \
  --pcm input.pcm \
  --nemo ~/.cache/huggingface/hub/diar_streaming_sortformer_4spk-v2.1.nemo \
  --dump-preds /tmp/input.preds.npy > /dev/null

# 3. Run sherpa-onnx (the shipping binary, now median-free):
target/release/examples/sortformer_bench \
  --model ~/.local/share/telemuze/models/diar_streaming_sortformer_4spk-v2.1.onnx \
  --pcm input.pcm > /tmp/input.sherpa.json

# 4. Apply NeMo's real post-processing and score:
.venv/bin/python /tmp/sortformer_parity/nemo_real_pp.py \
  --preds /tmp/input.preds.npy \
  --audio-duration $DURATION \
  --out /tmp/input.nemo_real_pp.json
.venv/bin/python scripts/sortformer_der.py \
  /tmp/input.nemo_real_pp.json /tmp/input.sherpa.json
```

---
## (Historical) Original investigation notes

Everything below this line predates the 2026-04-11 afternoon
sweep and is kept for context. Note that DER numbers here were
measured against the flawed `diag-ref` reference and should be
re-interpreted in light of the table above.

## Setup

- Model: `diar_streaming_sortformer_4spk-v2.1.onnx`, exported from
  `diar_streaming_sortformer_4spk-v2.1.nemo`
- Streaming params (native v2.1): `chunk_len=188, fifo_len=0,
  spkcache_len=188, spkcache_update_period=188`
- Post-processing: NeMo callhome config — `onset=0.641, offset=0.561,
  pad_onset=0.229, pad_offset=0.079, min_duration_on=0.511,
  min_duration_off=0.296, median_window=11`
- Reference: `scripts/sortformer_diag_ref.py` running NeMo PyTorch
  `forward_streaming`
- Hypothesis: `target/release/examples/sortformer_bench` running the
  sherpa-onnx C++ pipeline through our Rust binding
- Scorer: `scripts/sortformer_der.py` (pyannote.metrics, collar=0)

## Results

### `vid-chat-demo.pcm` (363.6s, 4 speakers, internal recording)

| Stage                              | DER       | Missed | False Alarm | Confusion | Segments |
| ---------------------------------- | --------- | ------ | ----------- | --------- | -------- |
| Sherpa-onnx, no median filter      | 1.58%     | 0.08s  | 5.30s       | 0.00s     | 60       |
| **Sherpa-onnx + median filter**    | **0.49%** | 0.16s  | 1.52s       | 0.00s     | 55       |

Post-fix the entire residual is boundary jitter — confusion is 0.00%
and segment count matches NeMo exactly.

### `0-four-speakers-zh.wav` (56.9s, 4 speakers, sherpa-onnx release sample)

Source: `https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav`

| Stage                           | DER       | Missed | False Alarm | Confusion | Segments |
| ------------------------------- | --------- | ------ | ----------- | --------- | -------- |
| **Sherpa-onnx + median filter** | **8.72%** | 2.63s  | 0.16s       | 0.00s     | 9        |

False alarm is a clean **0.50%** — the median filter fix behaves
identically well on this sample. Of the 9 segments sherpa-onnx emits,
7 match NeMo's 10 segments to sub-frame precision, and 2 more differ
only in the end time by exactly one model frame (80 ms). The entire
8.22% missed is **one missing speaker**: NeMo detects
`spk3: 47.93 → 50.56` (2.63s), sherpa-onnx misses it entirely.

#### Evidence: speaker assignment divergence, not mel drift

Dumping the raw sigmoid predictions from both pipelines on this
recording and comparing frame-by-frame produced a much sharper
picture than the vid-chat-demo mel-drift hypothesis would suggest:

| Speaker | Mean over full recording (NeMo / Sherpa) | Mean \|diff\| | Max \|diff\| | Frames with \|diff\| > 0.1 |
| ------- | ---------------------------------------- | ------------ | ----------- | ------------------------- |
| spk0    | 0.1805 / 0.1801                          | 0.0005       | 0.029       | 0                         |
| spk1    | 0.0982 / 0.0987                          | 0.0011       | 0.019       | 0                         |
| spk2    | 0.1591 / 0.1782                          | 0.019        | **0.335**   | 34                        |
| spk3    | 0.0340 / 0.0196                          | 0.015        | **0.330**   | 36                        |

**spk0 and spk1 match to essentially zero error.** That rules out
uniform mel drift, which would affect all four speakers symmetrically.
The divergence is contained to spk2 and spk3 and, for spk3, is a
single contiguous region — exactly the window where the new speaker
emerges.

At the spk3 window (frames 601-635, ≈ 48.08-50.80s):

| Side   | spk0  | spk1  | spk2  | spk3  | sum  |
| ------ | ----- | ----- | ----- | ----- | ---- |
| NeMo   | 0.002 | 0.015 | 0.225 | **0.635** | 0.876 |
| Sherpa | 0.001 | 0.027 | **0.500** | 0.354 | 0.882 |

Both sides agree *something* is being spoken at ~0.88 total
confidence. They disagree on **who**: NeMo correctly routes the
activity to spk3 (the emerging new speaker); sherpa-onnx attributes
most of it to spk2 (an already-known speaker). This is a speaker
**assignment** problem, not a detection one. Sherpa-onnx's spk3
sigmoid never crosses 0.5 in this window — no onset-threshold tweak
can recover the segment.

#### Root cause hypothesis

Speaker identity during streaming is determined by the attention
mechanism's lookup against the speaker cache (`spkcache`). The cache
is periodically **compressed** from ~376 to 188 entries by a
quality-scored top-k selection in `_compress_spkcache`. The exact
frames kept in the cache affect how the attention routes new activity:
if the cache's idea of "spk2" has diluted, a new speaker can get
pulled into that slot instead of a fresh one.

Both sherpa-onnx and parakeet-rs port this compression logic from
NeMo, and a direct line-by-line reading of all three
(`sortformer_modules.py:_compress_spkcache`, `parakeet-rs src/sortformer.rs:compress_spkcache`,
`sherpa-onnx offline-sortformer-diarization.cc:CompressSpkcache`)
doesn't surface an obvious bug. The divergence is likely a compounding
float-precision / ordering effect across several compression cycles,
triggered or exacerbated by tiny mel-level differences earlier in the
recording. This is consistent with the finding that spk0/spk1 are
identical: those speakers' cache entries were locked in during chunks
0-1 when state had not yet diverged; spk2 partially drifted by
chunk 3, and spk3's arrival at chunk 3 landed in a diverged cache.

#### What this means for the median-filter fix

The fix (sherpa-onnx commit `5a75ffd0`) is still correct and still
matters — without it, vid-chat-demo would be at 1.58% DER instead of
0.49%. On sub-minute recordings with late-emerging marginal speakers
it simply isn't enough on its own.

#### Paths forward (none taken in this session)

1. **Mel preprocessor byte-for-byte port** — replace
   `ExtractMelFeatures` in sherpa-onnx with a direct port of NeMo's
   `AudioToMelSpectrogramPreprocessor`. Eliminates the seed of the
   state divergence. ~1 day of work.
2. **Deterministic top-k tie-breaking in CompressSpkcache** — replace
   `std::nth_element` with a stable partial-sort that ties break by
   ascending frame index, exactly matching NeMo's
   `torch.topk(sorted=False)` + `torch.sort` pattern. Low risk if
   done carefully but may not actually fix this particular symptom
   since the ties are rare.
3. **Accept and document** — current state. The fix we have closes
   most of the gap on long recordings; short-sample edge cases where
   a borderline speaker gets swapped are a known limitation.

A one-session fix that doesn't risk regressing vid-chat-demo isn't
available. This document exists to save the next investigator the
3-4 hours it took to narrow the problem down to here.

## Attribution of the original 1.58% gap

Measured by dumping raw sigmoid predictions and mel features from both
pipelines with env-var-gated hooks in `offline-sortformer-diarization.cc`
(those dumps were removed after the isolation work; the commit removing
them is in the same sherpa-onnx fork branch as the fix).

- **H1 — missing median filter: ~1.1 pp.** NeMo's `median_window=11`
  smooths per-speaker sigmoids before binarization; sherpa-onnx had no
  equivalent. Disabling NeMo's median filter on the reference side
  dropped DER from 1.58% to 0.24%, pinning H1 as the dominant
  contributor. **Fixed** — see sherpa-onnx commit `5a75ffd0`.
- **H4 — mel-feature drift: ~0.49 pp residual.** Sherpa-onnx uses its
  own Hann window + preemphasis + log-guard in `ExtractMelFeatures`;
  NeMo uses `model.preprocessor` (torchaudio-backed). Mean |mel diff|
  is ~6.5e-3, with 28 frames showing >1 dB deviations in individual
  bins, distributed throughout the recording rather than concentrated
  at edges. This propagates into max per-frame sigmoid diffs of ~0.22
  (mean 1.5e-3), which cause small boundary shifts. Not fixed — the
  DER impact is small and porting NeMo's preprocessor byte-for-byte
  isn't justified for a 1.5s false-alarm budget over 6 minutes.
- **H2, H3, H5, H6** (frame-grid quantization, filter/merge ordering,
  ORT vs PyTorch inference drift, 80ms frame quantization floor) —
  each <0.1 pp, all absorbed into the H4 residual above.

Note: this attribution covers the vid-chat-demo residual specifically,
where the failure mode is sub-second boundary shifts. The separate
zh-sample finding above — where sherpa-onnx drops a marginal speaker
entirely — is a different failure mechanism (speaker cache compression
divergence, not boundary jitter). H4 is the most plausible seed for
that divergence too, but the downstream effect is assignment drift,
not boundary drift.

## Fix

Sherpa-onnx fork branch `telemuze-patches`, commit `5a75ffd0`:

- New `median_window` field on `OfflineSortformerDiarizationConfig`
  (default 11)
- `MedianFilterPerSpeaker()` helper in
  `sherpa-onnx/csrc/offline-sortformer-diarization.cc` — np.median
  semantics including even-window averaging at edges
- Applied in `Binarize()` at 80 ms model-frame resolution, before
  `repeat_interleave` upsample, matching NeMo's ordering
- Plumbed through the C API (`c-api.h` / `c-api.cc`) and Rust bindings
  (`sherpa-onnx-sys` + `sherpa-onnx`)
- Exposed as `--sortformer-median-window` command-line option
- Telemuze picks this up automatically via path-dep on the fork

## Diagnostic scripts

Under `scripts/`:

- `sortformer_diag_ref.py` — NeMo `forward_streaming` reference with
  `--dump-preds`, `--dump-mel`, `--no-median` flags. Runs at ~3.5x real
  time on CPU.
- `sortformer_diff_preds.py` — compares NeMo `.npy` preds against
  sherpa-onnx binary preds dump. Reports shape alignment, max/mean/rmse,
  and applies the 1e-4 branch gate.
- `sortformer_der.py` — thin pyannote.metrics DER scorer.

These were built for the isolation work but are kept as the canonical
way to measure regressions going forward. To reproduce the current
parity number:

```bash
# Hypothesis: sherpa-onnx via the telemuze bench binary
target/release/examples/sortformer_bench \
  --model ~/.local/share/telemuze/models/diar_streaming_sortformer_4spk-v2.1.onnx \
  --pcm ~/media/vid-chat-demo.pcm > /tmp/sherpa_segments.json

# Reference: NeMo PyTorch
.venv/bin/python scripts/sortformer_diag_ref.py \
  --pcm ~/media/vid-chat-demo.pcm \
  --nemo ~/.cache/huggingface/hub/diar_streaming_sortformer_4spk-v2.1.nemo \
  > /tmp/nemo_segments.json

# Score
.venv/bin/python scripts/sortformer_der.py \
  /tmp/nemo_segments.json /tmp/sherpa_segments.json
```

For the zh sample:

```bash
# Convert the wav to raw f32 PCM (sortformer_bench expects headerless)
curl -sSL -o /tmp/0-four-speakers-zh.wav \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav
ffmpeg -v error -y -i /tmp/0-four-speakers-zh.wav \
  -f f32le -ac 1 -ar 16000 /tmp/0-four-speakers-zh.pcm

target/release/examples/sortformer_bench \
  --model ~/.local/share/telemuze/models/diar_streaming_sortformer_4spk-v2.1.onnx \
  --pcm /tmp/0-four-speakers-zh.pcm > /tmp/zh_sherpa.json

.venv/bin/python scripts/sortformer_diag_ref.py \
  --pcm /tmp/0-four-speakers-zh.pcm \
  --nemo ~/.cache/huggingface/hub/diar_streaming_sortformer_4spk-v2.1.nemo \
  > /tmp/zh_nemo.json

.venv/bin/python scripts/sortformer_der.py /tmp/zh_nemo.json /tmp/zh_sherpa.json
```

Expected: vid-chat-demo at 0.49% DER, zh at 8.72% DER. If either drifts
by more than ~0.1 pp, something upstream has changed.

## When to revisit

- **Regression watch**: if `scripts/sortformer_der.py` on either
  recording drifts by more than ~0.1 pp from the numbers above,
  something has changed — likely mel extraction, an ONNX runtime
  upgrade, or a post-processing tweak.
- **Closing the vid-chat-demo residual (0.49%)**: only worth doing if
  the ~1.5s residual false alarm matters for a downstream consumer.
  The fix is the mel-preprocessor port.
- **Closing the zh-sample gap (8.72%)**: worth doing if we see
  real-world recordings where marginal speakers get swapped — this
  likely generalizes to any recording with a late-emerging speaker
  whose sigmoid sits near the onset threshold. The three candidate
  fixes are listed in the zh section above; the most principled one
  is the mel-preprocessor port (since it's the likely seed of the
  divergence), but deterministic top-k tie-breaking in
  `CompressSpkcache` is cheaper to try first.
