#!/usr/bin/env python3
"""Thin DER scorer between two Sortformer segments JSON files.

Input format: {"segments": [{"start": f, "end": f, "speaker": i}, ...]}
Speaker labels are optimally mapped by pyannote.metrics.

Usage:
    python sortformer_der.py reference.json hypothesis.json [--collar 0.25]
"""
import argparse
import json
import warnings


def load_annotation(path: str):
    from pyannote.core import Annotation, Segment

    with open(path) as f:
        data = json.load(f)
    ann = Annotation()
    for i, seg in enumerate(data["segments"]):
        ann[Segment(float(seg["start"]), float(seg["end"])), i] = f"spk{seg['speaker']}"
    return ann


def main():
    p = argparse.ArgumentParser()
    p.add_argument("reference")
    p.add_argument("hypothesis")
    p.add_argument("--collar", type=float, default=0.0)
    p.add_argument("--skip-overlap", action="store_true")
    args = p.parse_args()

    from pyannote.metrics.diarization import DiarizationErrorRate

    ref = load_annotation(args.reference)
    hyp = load_annotation(args.hypothesis)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metric = DiarizationErrorRate(collar=args.collar, skip_overlap=args.skip_overlap)
        comp = metric(ref, hyp, detailed=True)

    der = comp["diarization error rate"]
    total = comp["total"]
    print(f"DER: {der * 100:.2f}%")
    if total > 0:
        print(f"  total ref:   {total:.2f}s")
        print(f"  missed:      {comp['missed detection']:.2f}s ({comp['missed detection']/total*100:.2f}%)")
        print(f"  false alarm: {comp['false alarm']:.2f}s ({comp['false alarm']/total*100:.2f}%)")
        print(f"  confusion:   {comp['confusion']:.2f}s ({comp['confusion']/total*100:.2f}%)")


if __name__ == "__main__":
    main()
