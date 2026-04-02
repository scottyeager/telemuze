#!/usr/bin/env python3
"""
Step 2: Add sherpa-onnx metadata to encoder and quantize all models.
Run after the NeMo export has produced encoder.onnx, decoder.onnx, joiner.onnx.

Output defaults to ~/.local/share/telemuze/models/nemotron-speech-streaming-en-0.6b/
"""
import argparse
import gc
import os
from pathlib import Path
from typing import Dict

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


DEFAULT_OUTPUT_DIR = (
    Path.home() / ".local" / "share" / "telemuze" / "models" / "nemotron-speech-streaming-en-0.6b"
)


def add_meta_data(filename: str, meta_data: Dict[str, str]):
    """Add meta data to an ONNX model, consolidating external data."""
    print(f"  Loading {filename}...")
    model = onnx.load(filename)

    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    external_filename = filename.split(".onnx")[0]
    data_file = external_filename + ".data"

    # Remove existing external data file to avoid FileExistsError
    if os.path.exists(data_file):
        print(f"  Removing existing {data_file}...")
        os.remove(data_file)

    print(f"  Saving with consolidated external data -> {data_file}...")
    onnx.save(
        model,
        filename,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_filename + ".data",
    )
    del model
    gc.collect()
    print(f"  Done with {filename}")


def main():
    parser = argparse.ArgumentParser(description="Add sherpa-onnx metadata and quantize models")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory containing exported ONNX models (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=1120,
        help="Chunk size in ms that was used during export (default: 1120). "
             "Must match the --chunk-ms used with export_onnx.py.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    chunk_frames = args.chunk_ms // 10
    # pre_encode_cache_size for nemotron is 9 frames for all chunk sizes
    pre_encode_cache = 9
    window_size = chunk_frames + pre_encode_cache
    print(f"Working directory: {output_dir}")
    print(f"Chunk: {args.chunk_ms}ms ({chunk_frames} frames), window_size={window_size}")

    meta_data = {
        "vocab_size": 1024,
        "window_size": window_size,
        "chunk_shift": chunk_frames,
        "normalize_type": "",
        "cache_last_channel_dim1": 24,
        "cache_last_channel_dim2": 70,
        "cache_last_channel_dim3": 1024,
        "cache_last_time_dim1": 24,
        "cache_last_time_dim2": 1024,
        "cache_last_time_dim3": 8,
        "pred_rnn_layers": 2,
        "pred_hidden": 640,
        "subsampling_factor": 8,
        "feat_dim": 128,
        "model_type": "EncDecHybridRNNTCTCBPEModel",
        "version": "1",
        "model_author": "NeMo",
        "url": "https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b",
        "comment": "Only the transducer branch is exported",
    }

    print("Step 1: Adding metadata to encoder.onnx...")
    add_meta_data(str(output_dir / "encoder.onnx"), meta_data)

    print("\nStep 2: Quantizing models (one at a time to save memory)...")
    for m in ["encoder", "decoder", "joiner"]:
        print(f"  Quantizing {m}.onnx -> {m}.int8.onnx...")
        quantize_dynamic(
            model_input=str(output_dir / f"{m}.onnx"),
            model_output=str(output_dir / f"{m}.int8.onnx"),
            weight_type=QuantType.QUInt8,
        )
        gc.collect()
        print(f"  {m}.int8.onnx created.")

    print("\nAll done!")
    print("Meta data:", meta_data)


if __name__ == "__main__":
    main()
