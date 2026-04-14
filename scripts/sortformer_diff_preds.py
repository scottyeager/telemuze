#!/usr/bin/env python3
"""Diff raw sigmoid predictions from NeMo and sherpa-onnx Sortformer pipelines.

Expects:
    nemo_preds.npy  — float32 array of shape (n_frames, 4) produced by
                      sortformer_diag_ref.py --dump-preds
    sherpa_preds.bin — custom binary format produced by the
                      SORTFORMER_DUMP_PREDS env var in sherpa-onnx:
                          int32 n_frames
                          int32 n_speakers
                          float32[n_frames * n_speakers] row-major

Reports shape alignment, max/mean/rmse of |a-b|, and the first mismatching
frames at a 1e-3 threshold.
"""
import argparse
import sys

import numpy as np


def load_sherpa_bin(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(8), dtype=np.int32)
        n_frames, n_spk = int(header[0]), int(header[1])
        payload = np.frombuffer(f.read(), dtype=np.float32)
    expected = n_frames * n_spk
    if payload.size != expected:
        raise ValueError(
            f"{path}: header says {n_frames}x{n_spk} = {expected} floats, "
            f"got {payload.size}"
        )
    return payload.reshape(n_frames, n_spk).copy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("nemo_npy")
    p.add_argument("sherpa_bin")
    args = p.parse_args()

    a = np.load(args.nemo_npy).astype(np.float32, copy=False)
    b = load_sherpa_bin(args.sherpa_bin)

    print(f"nemo  : {a.shape} dtype={a.dtype}")
    print(f"sherpa: {b.shape} dtype={b.dtype}")

    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        print(f"ERROR: incompatible shapes", file=sys.stderr)
        sys.exit(1)

    if a.shape[0] != b.shape[0]:
        n = min(a.shape[0], b.shape[0])
        print(
            f"NOTE: frame count differs (nemo={a.shape[0]}, sherpa={b.shape[0]}); "
            f"truncating both to {n} for comparison"
        )
        a = a[:n]
        b = b[:n]

    diff = np.abs(a - b)
    max_d = float(diff.max())
    mean_d = float(diff.mean())
    rmse = float(np.sqrt(((a - b) ** 2).mean()))
    print(f"max|diff|  = {max_d:.6e}")
    print(f"mean|diff| = {mean_d:.6e}")
    print(f"rmse       = {rmse:.6e}")

    thresh = 1e-3
    bad = np.where(diff.max(axis=1) > thresh)[0]
    print(f"frames with any |diff| > {thresh}: {bad.size} / {diff.shape[0]}")
    for t in bad[:5]:
        print(
            f"  frame {int(t):5d}: nemo={a[t].tolist()}  sherpa={b[t].tolist()}"
        )

    branch_gate = 1e-4
    verdict = "Branch A (post-processing)" if max_d < branch_gate else "Branch B (upstream)"
    print(f"branch gate (max|diff| < {branch_gate}): {verdict}")


if __name__ == "__main__":
    main()
