#!/usr/bin/env python3
"""NeMo Sortformer streaming reference for the DER-gap diagnostic.

Purpose-built for the sherpa-onnx vs NeMo Python comparison described in
the sortformer DER-gap plan. Runs forward_streaming with native v2.1 params,
optionally dumps raw sigmoid preds and mel features, and can disable the
median filter to isolate H1.

Usage:
    python sortformer_diag_ref.py --pcm audio.pcm \
        --nemo ~/.cache/.../diar_streaming_sortformer_4spk-v2.1.nemo \
        --dump-preds /tmp/nemo_preds.npy
"""
import argparse
import json
import logging
import sys

import numpy as np
import torch

SAMPLE_RATE = 16_000
FRAME_DURATION = 0.08
NUM_SPEAKERS = 4

CONFIGS = {
    "callhome": dict(
        onset=0.641, offset=0.561,
        pad_onset=0.229, pad_offset=0.079,
        min_duration_on=0.511, min_duration_off=0.296,
        median_window=11,
    ),
    "dihard3": dict(
        onset=0.56, offset=1.0,
        pad_onset=0.063, pad_offset=0.002,
        min_duration_on=0.007, min_duration_off=0.151,
        median_window=11,
    ),
}


def read_pcm(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = f.read()
    assert len(data) % 4 == 0, f"PCM byte length {len(data)} not multiple of 4"
    return np.frombuffer(data, dtype=np.float32).copy()


def median_filter(preds: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    n_frames, n_spk = preds.shape
    out = preds.copy()
    for spk in range(n_spk):
        for t in range(n_frames):
            s = max(0, t - half)
            e = min(n_frames, t + half + 1)
            out[t, spk] = np.median(preds[s:e, spk])
    return out


def binarize(preds: np.ndarray, cfg: dict, n_audio_samples: int) -> list:
    onset = cfg["onset"]
    offset = cfg["offset"]
    pad_on_s = int(cfg["pad_onset"] * SAMPLE_RATE)
    pad_off_s = int(cfg["pad_offset"] * SAMPLE_RATE)
    min_on_s = int(cfg["min_duration_on"] * SAMPLE_RATE)
    min_off_s = int(cfg["min_duration_off"] * SAMPLE_RATE)
    spf = int(FRAME_DURATION * SAMPLE_RATE)

    n_frames = preds.shape[0]
    segments = []

    for spk in range(NUM_SPEAKERS):
        in_seg = False
        seg_start = 0
        temp = []
        for t in range(n_frames):
            p = float(preds[t, spk])
            if p >= onset and not in_seg:
                in_seg = True
                seg_start = t
            elif p < offset and in_seg:
                in_seg = False
                start_s = max(0, seg_start * spf - pad_on_s)
                end_s = t * spf + pad_off_s
                if end_s - start_s >= min_on_s:
                    temp.append((start_s, end_s, spk))
        if in_seg:
            start_s = max(0, seg_start * spf - pad_on_s)
            end_s = n_frames * spf + pad_off_s
            if end_s - start_s >= min_on_s:
                temp.append((start_s, end_s, spk))

        if len(temp) > 1:
            merged = [list(temp[0])]
            for s, e, spk_ in temp[1:]:
                gap = max(0, s - merged[-1][1])
                if gap < min_off_s:
                    merged[-1][1] = e
                else:
                    merged.append([s, e, spk_])
            segments.extend(merged)
        else:
            segments.extend([list(t) for t in temp])

    segments.sort(key=lambda s: s[0])
    out = []
    for s, e, spk in segments:
        e = min(e, n_audio_samples)
        if e > s:
            out.append({
                "start": s / SAMPLE_RATE,
                "end": e / SAMPLE_RATE,
                "speaker": spk,
            })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pcm", required=True)
    p.add_argument("--nemo", required=True)
    p.add_argument("--chunk-len", type=int, default=188)
    p.add_argument("--fifo-len", type=int, default=0)
    p.add_argument("--spkcache-len", type=int, default=188)
    p.add_argument("--spkcache-update-period", type=int, default=188)
    p.add_argument("--config", choices=list(CONFIGS.keys()), default="callhome")
    p.add_argument("--dump-preds", default=None, help="Save raw sigmoid preds to .npy")
    p.add_argument("--dump-mel", default=None, help="Save mel features to .npy, shape (n_frames, 128)")
    p.add_argument("--no-median", action="store_true", help="Disable median filter (sets median_window=1)")
    args = p.parse_args()

    logging.disable(logging.WARNING)

    audio = read_pcm(args.pcm)
    print(f"diag-ref: loaded {len(audio)} samples ({len(audio)/SAMPLE_RATE:.1f}s)",
          file=sys.stderr)

    from nemo.collections.asr.models import SortformerEncLabelModel

    model = SortformerEncLabelModel.restore_from(restore_path=args.nemo, map_location="cpu")
    model.eval()
    model.sortformer_modules.chunk_len = args.chunk_len
    model.sortformer_modules.fifo_len = args.fifo_len
    model.sortformer_modules.spkcache_len = args.spkcache_len
    model.sortformer_modules.spkcache_update_period = args.spkcache_update_period

    print(
        f"diag-ref: chunk_len={args.chunk_len} fifo_len={args.fifo_len} "
        f"spkcache_len={args.spkcache_len} update_period={args.spkcache_update_period} "
        f"config={args.config} no_median={args.no_median}",
        file=sys.stderr,
    )

    audio_t = torch.from_numpy(audio).unsqueeze(0)
    length_t = torch.tensor([len(audio)], dtype=torch.long)

    with torch.no_grad():
        features, feat_lengths = model.preprocessor(
            input_signal=audio_t, length=length_t
        )
        if args.dump_mel:
            # preprocessor returns (batch, n_mels, n_frames); transpose to
            # (n_frames, n_mels) to match sherpa-onnx's row-major layout.
            mel_np = features[0].cpu().numpy().T.astype(np.float32, copy=False)
            np.save(args.dump_mel, mel_np)
            print(f"diag-ref: saved mel {mel_np.shape} -> {args.dump_mel}",
                  file=sys.stderr)

        preds = model.forward_streaming(
            processed_signal=features,
            processed_signal_length=feat_lengths,
        )

    preds_np = preds[0].cpu().numpy().astype(np.float32, copy=False)
    print(f"diag-ref: raw preds {preds_np.shape}", file=sys.stderr)

    if args.dump_preds:
        np.save(args.dump_preds, preds_np)
        print(f"diag-ref: saved preds -> {args.dump_preds}", file=sys.stderr)

    cfg = dict(CONFIGS[args.config])
    if args.no_median:
        cfg["median_window"] = 1
    if cfg["median_window"] > 1:
        preds_np = median_filter(preds_np, cfg["median_window"])

    segments = binarize(preds_np, cfg, n_audio_samples=len(audio))
    print(f"diag-ref: produced {len(segments)} segments", file=sys.stderr)

    print(json.dumps({"segments": segments}))


if __name__ == "__main__":
    main()
