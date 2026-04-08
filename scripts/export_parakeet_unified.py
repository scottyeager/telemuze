#!/usr/bin/env python3
"""
Download and export parakeet-unified-en-0.6b to ONNX for sherpa-onnx offline use.

This is NVIDIA's unified FastConformer-RNNT model (600M params), jointly trained
for both offline and streaming modes. We export only the transducer branch for
offline use with sherpa-onnx's OfflineRecognizer.

HuggingFace: https://huggingface.co/nvidia/parakeet-unified-en-0.6b

Output defaults to ~/.local/share/telemuze/models/parakeet-unified-en-0.6b/

Usage:
    python scripts/export_parakeet_unified.py
    python scripts/export_parakeet_unified.py --output-dir /path/to/output
    python scripts/export_parakeet_unified.py --nemo-path /path/to/local.nemo
"""
import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Dict

import torch


DEFAULT_OUTPUT_DIR = (
    Path.home() / ".local" / "share" / "telemuze" / "models" / "parakeet-unified-en-0.6b"
)

HF_MODEL_NAME = "nvidia/parakeet-unified-en-0.6b"


def consolidate_and_add_metadata(filename: str, meta_data: Dict[str, str]):
    """Consolidate scattered external data into one .data file, fix opset, add metadata.

    For large models (>2GB weights), we must:
    1. Load the graph WITHOUT external data (to avoid >2GB protobuf)
    2. Use onnx.compose to consolidate external data files into one
    3. Fix opset and metadata on the lightweight graph proto
    """
    import onnx
    from onnx.external_data_helper import (
        convert_model_to_external_data,
        load_external_data_for_model,
    )

    filepath = os.path.abspath(filename)
    dirpath = os.path.dirname(filepath)
    basename = os.path.splitext(os.path.basename(filepath))[0]
    data_filename = basename + ".data"
    data_filepath = os.path.join(dirpath, data_filename)

    # Load the full model (graph + weights) into memory
    # Use onnx's external data loading which handles scattered files
    model = onnx.load(filepath, load_external_data=True)

    # Ensure ai.onnx opset is present (NeMo 2.8 may omit it)
    has_onnx_domain = any(
        op.domain == "" or op.domain == "ai.onnx"
        for op in model.opset_import
    )
    if not has_onnx_domain:
        opset = model.opset_import.add()
        opset.domain = ""
        opset.version = 17

    # Replace metadata
    while len(model.metadata_props):
        model.metadata_props.pop()
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    # Remove old data file if it exists
    if os.path.exists(data_filepath):
        os.remove(data_filepath)

    # Save with all tensors in a single external data file.
    # size_threshold=0 ensures ALL tensors go external (critical for >2GB models).
    convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=data_filename,
        size_threshold=0,
    )
    onnx.save(model, filepath)
    del model
    gc.collect()

    # Clean up scattered external data files left by NeMo export
    keep_names = {os.path.basename(filepath), data_filename, "tokens.txt", "bpe.vocab"}
    for f in os.listdir(dirpath):
        full = os.path.join(dirpath, f)
        if os.path.isfile(full) and f not in keep_names \
                and not f.endswith((".onnx", ".int8.onnx", ".data")):
            os.remove(full)
            print(f"  Cleaned up: {f}")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Download and export parakeet-unified-en-0.6b to ONNX"
    )
    parser.add_argument(
        "--nemo-path",
        default=None,
        help="Path to local .nemo checkpoint (downloads from HuggingFace if omitted)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip int8 quantization (export full-precision only)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    import nemo.collections.asr as nemo_asr

    if args.nemo_path:
        print(f"Step 1: Loading model from {args.nemo_path}...")
        asr_model = nemo_asr.models.ASRModel.restore_from(args.nemo_path)
    else:
        print(f"Step 1: Downloading model from HuggingFace ({HF_MODEL_NAME})...")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=HF_MODEL_NAME)
    print("Model loaded.")

    # Save tokens
    with open(output_dir / "tokens.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(asr_model.joint.vocabulary):
            f.write(f"{s} {i}\n")
        f.write(f"<blk> {i+1}\n")
    print(f"Saved tokens.txt ({i+2} tokens)")

    asr_model.eval()

    # Extract model parameters for metadata
    feat_dim = asr_model.cfg.preprocessor.features if hasattr(asr_model.cfg.preprocessor, "features") else 128

    normalize_type = asr_model.cfg.preprocessor.normalize
    if normalize_type == "NA":
        normalize_type = ""

    meta_data = {
        "vocab_size": asr_model.decoder.vocab_size,
        "normalize_type": normalize_type,
        "pred_rnn_layers": asr_model.decoder.pred_rnn_layers,
        "pred_hidden": asr_model.decoder.pred_hidden,
        "subsampling_factor": 8,
        "feat_dim": feat_dim,
        "model_type": "EncDecHybridRNNTCTCBPEModel",
        "version": "1",
        "model_author": "NeMo",
        "url": f"https://huggingface.co/{HF_MODEL_NAME}",
        "comment": "Only the transducer branch is exported",
    }
    print("Meta data:", meta_data)

    # Export transducer branch (no cache_support — this is for offline use)
    print("\nStep 2a: Exporting encoder.onnx...")
    sys.stdout.flush()
    asr_model.encoder.export(str(output_dir / "encoder.onnx"))
    gc.collect()
    print("Encoder exported.")

    print("\nStep 2b: Exporting decoder.onnx...")
    sys.stdout.flush()
    asr_model.decoder.export(str(output_dir / "decoder.onnx"))
    gc.collect()
    print("Decoder exported.")

    print("\nStep 2c: Exporting joiner.onnx...")
    sys.stdout.flush()
    asr_model.joint.export(str(output_dir / "joiner.onnx"))
    gc.collect()
    print("Joiner exported.")

    del asr_model
    gc.collect()
    print("\nModel released from memory.")

    # Consolidate external data, fix opset, add metadata
    for m in ["encoder", "decoder", "joiner"]:
        print(f"\nStep 3: Consolidating and fixing {m}.onnx...")
        sys.stdout.flush()
        md = meta_data if m == "encoder" else {}
        consolidate_and_add_metadata(str(output_dir / f"{m}.onnx"), md)
        print(f"{m}.onnx consolidated.")

    # Quantize — load model ourselves to avoid quantize_dynamic's broken
    # shape-inference path which copies to a temp dir without external data.
    if not args.no_quantize:
        import onnx
        from onnxruntime.quantization import QuantType, quantize_dynamic

        for m in ["encoder", "decoder", "joiner"]:
            print(f"\nStep 4: Quantizing {m}.onnx -> {m}.int8.onnx...")
            sys.stdout.flush()
            model_path = str(output_dir / f"{m}.onnx")
            int8_path = str(output_dir / f"{m}.int8.onnx")

            # Load the model with external data resolved
            model = onnx.load(model_path, load_external_data=True)
            quantize_dynamic(
                model_input=model,
                model_output=int8_path,
                weight_type=QuantType.QUInt8,
            )
            del model
            gc.collect()
            print(f"{m}.int8.onnx created.")

    print(f"\nDone! Files exported to {output_dir}")


if __name__ == "__main__":
    main()
