#!/usr/bin/env python3
# /// script
# dependencies = ["onnxruntime", "onnx"]
# ///
"""
Quantize sherpa-onnx transducer model files (encoder/decoder/joiner) to INT8.

Writes quantized files to a separate output directory with "-int8" suffix.

Usage:
  uv run scripts/quantize_model.py ~/.local/share/telemuze/models/sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000
"""

import shutil
import sys
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

ONNX_FILES = ["encoder.onnx", "decoder.onnx", "joiner.onnx"]
OTHER_FILES = ["tokens.txt", "bpe.vocab"]


def quantize_file(src: Path, dst: Path):
    print(f"Quantizing {src.name} → {dst.name} ...", flush=True)
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )
    src_mb = src.stat().st_size / 1_000_000
    dst_mb = dst.stat().st_size / 1_000_000
    print(f"  {src_mb:.0f} MB → {dst_mb:.0f} MB ({dst_mb/src_mb*100:.0f}%)")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <model-dir>")
        sys.exit(1)

    model_dir = Path(sys.argv[1])
    if not model_dir.is_dir():
        print(f"Error: {model_dir} is not a directory")
        sys.exit(1)

    out_dir = model_dir.parent / (model_dir.name + "-int8")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in ONNX_FILES:
        src = model_dir / name
        dst = out_dir / name.replace(".onnx", ".int8.onnx")
        if not src.exists():
            print(f"Skipping {name} — not found in source")
            continue
        if dst.exists():
            print(f"Skipping {name} — {dst.name} already exists in output")
            continue
        quantize_file(src, dst)

    for name in OTHER_FILES:
        src = model_dir / name
        dst = out_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"Copied {name}")

    print(f"\nDone. INT8 model ready in: {out_dir}")


if __name__ == "__main__":
    main()
