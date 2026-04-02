#!/usr/bin/env python3
"""
Download and export parakeet_realtime_eou_120m-v1 to ONNX for sherpa-onnx.

This is NVIDIA's streaming end-of-utterance model (120M params, FastConformer-RNNT).
It emits an <EOU> token when it detects the speaker has finished an utterance,
providing semantic turn-taking rather than silence-based endpoint detection.

HuggingFace: https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1

Output defaults to ~/.local/share/telemuze/models/parakeet-realtime-eou-120m-v1/

Usage:
    python scripts/export_parakeet_eou.py
    python scripts/export_parakeet_eou.py --output-dir /path/to/output
    python scripts/export_parakeet_eou.py --nemo-path /path/to/local.nemo
"""
import argparse
import gc
import sys
from pathlib import Path
from typing import Dict

import torch


DEFAULT_OUTPUT_DIR = (
    Path.home() / ".local" / "share" / "telemuze" / "models" / "parakeet-realtime-eou-120m-v1"
)

HF_MODEL_NAME = "nvidia/parakeet_realtime_eou_120m-v1"


def add_meta_data(filename: str, meta_data: Dict[str, str]):
    """Add meta data to an ONNX model. It is changed in-place."""
    import onnx
    import os

    model = onnx.load(filename)

    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    external_filename = filename.split(".onnx")[0]
    data_file = external_filename + ".data"

    if os.path.exists(data_file):
        os.remove(data_file)

    onnx.save(
        model,
        filename,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_filename + ".data",
    )
    del model
    gc.collect()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Download and export parakeet_realtime_eou_120m-v1 to ONNX"
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

    # Save tokens — includes <EOU> token in vocabulary
    with open(output_dir / "tokens.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(asr_model.joint.vocabulary):
            f.write(f"{s} {i}\n")
        f.write(f"<blk> {i+1}\n")
    print("Saved tokens.txt")

    # Report the EOU token
    vocab = asr_model.joint.vocabulary
    eou_candidates = [(i, s) for i, s in enumerate(vocab) if "eou" in s.lower() or "EOU" in s]
    if eou_candidates:
        print(f"EOU token(s) found: {eou_candidates}")
    else:
        print("WARNING: No <EOU> token found in vocabulary!")

    asr_model.eval()

    # Extract streaming config — this model has a two-phase chunking scheme
    # (first chunk differs from steady-state). sherpa-onnx uses the last entry
    # which is the steady-state chunk size.
    assert asr_model.encoder.streaming_cfg is not None, "Model does not have streaming config"
    cfg = asr_model.encoder.streaming_cfg

    chunk_sizes = cfg.chunk_size if isinstance(cfg.chunk_size, list) else [cfg.chunk_size]
    cache_sizes = cfg.pre_encode_cache_size if isinstance(cfg.pre_encode_cache_size, list) else [cfg.pre_encode_cache_size]

    chunk_size = chunk_sizes[-1]
    pre_encode_cache_size = cache_sizes[-1]
    window_size = chunk_size + pre_encode_cache_size

    print(f"Chunk sizes (frames): {chunk_sizes} — using steady-state: {chunk_size} ({chunk_size * 10}ms)")
    print(f"pre_encode_cache_size={pre_encode_cache_size}, window_size={window_size}")

    chunk_shift = chunk_size

    cache_last_channel_dim1 = len(asr_model.encoder.layers)
    cache_last_channel_dim2 = cfg.last_channel_cache_size
    cache_last_channel_dim3 = asr_model.encoder.d_model

    cache_last_time_dim1 = len(asr_model.encoder.layers)
    cache_last_time_dim2 = asr_model.encoder.d_model
    cache_last_time_dim3 = asr_model.encoder.conv_context_size[0]

    normalize_type = asr_model.cfg.preprocessor.normalize
    if normalize_type == "NA":
        normalize_type = ""

    feat_dim = asr_model.cfg.preprocessor.features if hasattr(asr_model.cfg.preprocessor, "features") else 80

    meta_data = {
        "vocab_size": asr_model.decoder.vocab_size,
        "window_size": window_size,
        "chunk_shift": chunk_shift,
        "normalize_type": normalize_type,
        "cache_last_channel_dim1": cache_last_channel_dim1,
        "cache_last_channel_dim2": cache_last_channel_dim2,
        "cache_last_channel_dim3": cache_last_channel_dim3,
        "cache_last_time_dim1": cache_last_time_dim1,
        "cache_last_time_dim2": cache_last_time_dim2,
        "cache_last_time_dim3": cache_last_time_dim3,
        "pred_rnn_layers": asr_model.decoder.pred_rnn_layers,
        "pred_hidden": asr_model.decoder.pred_hidden,
        "subsampling_factor": 8,
        "feat_dim": feat_dim,
        "model_type": "EncDecHybridRNNTCTCBPEModel",
        "version": "1",
        "model_author": "NeMo",
        "url": f"https://huggingface.co/{HF_MODEL_NAME}",
        "comment": "Streaming EOU model — transducer branch, emits <EOU> token",
    }
    print("Meta data:", meta_data)

    # Export each component
    asr_model.set_export_config({"cache_support": True})

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

    # Add metadata to encoder
    print("\nStep 3: Adding metadata to encoder.onnx...")
    sys.stdout.flush()
    add_meta_data(str(output_dir / "encoder.onnx"), meta_data)
    print("Metadata added.")

    # Quantize
    if not args.no_quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        for m in ["encoder", "decoder", "joiner"]:
            print(f"\nStep 4: Quantizing {m}.onnx -> {m}.int8.onnx...")
            sys.stdout.flush()
            quantize_dynamic(
                model_input=str(output_dir / f"{m}.onnx"),
                model_output=str(output_dir / f"{m}.int8.onnx"),
                weight_type=QuantType.QUInt8,
            )
            gc.collect()
            print(f"{m}.int8.onnx created.")

    print(f"\nDone! Files exported to {output_dir}")
    print("\nTo use with sherpa-onnx, look for the <EOU> token in decoded output")
    print("to detect end-of-utterance boundaries.")


if __name__ == "__main__":
    main()
